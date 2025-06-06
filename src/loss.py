import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, num_qubits, downscaling=0.25):
        super(Loss, self).__init__()
        
        self.num_qubits = num_qubits
        self.max_trace = 2 ** num_qubits
        self.downscaling = downscaling
        
        # Prepare single-qubit Pauli templates on CPU
        self.H_cpu = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / torch.sqrt(torch.tensor(2, dtype=torch.complex64))
        self.S_dag_cpu = torch.tensor([[1, 0], [0, -1j]], dtype=torch.complex64)
        self.YH_cpu = self.H_cpu @ self.S_dag_cpu

        self.trace_penalty_weight = 1.0
        self.pos_semi_def_penalty_weight = 1.0

    def tensor_product(self, *matrices):
        """Compute tensor product of multiple matrices (all should share the same device)."""
        result = matrices[0]
        for mat in matrices[1:]:
            result = torch.kron(result, mat)
        return result

    def get_pauli_matrix(self, index, device):
        """
        Return the single-qubit rotation for index ∈ {0: X, 1: Y, 2: Z},
        moved onto `device`.
        """
        if index == 0:
            return self.H_cpu.to(device)
        elif index == 1:
            return self.YH_cpu.to(device)
        else:
            # Identity for Z
            return torch.eye(2, dtype=torch.complex64, device=device)

    def apply_rotation(self, rho, basis_indices):
        """
        Rotate each density matrix `rho[i]` into the basis specified by `basis_indices[i]`.
        - rho: (batch_size, D, D) on some device
        - basis_indices: (batch_size, num_qubits) ints ∈ {0,1,2}
        Returns: (batch_size, D, D) rotated density matrices, on the same device.
        """
        batch_size, D, _ = rho.shape
        n = self.num_qubits
        device = rho.device

        # Build per-qubit, per-sample single-qubit rotations on `device`
        per_qubit_rotations = []
        for q in range(n):
            idx_q = basis_indices[:, q]  # shape: (batch_size,)
            rot_q = torch.zeros((batch_size, 2, 2), dtype=torch.complex64, device=device)

            # Mask for X
            mask_X = (idx_q == 0)
            if mask_X.any():
                rot_q[mask_X] = self.get_pauli_matrix(0, device)

            # Mask for Y
            mask_Y = (idx_q == 1)
            if mask_Y.any():
                rot_q[mask_Y] = self.get_pauli_matrix(1, device)

            # Mask for Z (identity)
            mask_Z = (idx_q == 2)
            if mask_Z.any():
                rot_q[mask_Z] = self.get_pauli_matrix(2, device)

            per_qubit_rotations.append(rot_q)

        # Construct the full n-qubit rotation U for each sample
        U_batch = torch.zeros((batch_size, D, D), dtype=torch.complex64, device=device)
        for i in range(batch_size):
            full = None
            for q in range(n):
                mat_q = per_qubit_rotations[q][i]  # (2,2)
                full = mat_q if (full is None) else torch.kron(full, mat_q)
            U_batch[i] = full

        U_dagger = U_batch.conj().transpose(-2, -1)  # (batch_size, D, D)
        temp = torch.bmm(U_batch, rho)              # (batch_size, D, D)
        rho_rotated = torch.bmm(temp, U_dagger)     # (batch_size, D, D)

        return rho_rotated

    def get_lower_triangular_flattened(self, rho):
        """
        Extract lower‐triangular entries from rho ∈ (batch_size, D, D),
        flatten real+imag parts into a single vector per sample.
        """
        batch_size, dim, _ = rho.shape
        device = rho.device
        indices = torch.tril_indices(dim, dim, device=device)
        lower_real = rho.real[:, indices[0], indices[1]]
        lower_imag = rho.imag[:, indices[0], indices[1]]
        return torch.cat([lower_real, lower_imag], dim=-1)

    def reconstruct_density_matrix_from_lower(self, flattened_vectors):
        """Reconstruct a symmetric matrix from the lower triangular part."""
        batch_size, flattened_length = flattened_vectors.size()
        d = int(((-1 + (1 + 8 * flattened_length)**0.5) / 2).real)  # Solve for dimension size
        indices = torch.tril_indices(d, d, device=flattened_vectors.device)

        matrix = torch.zeros(batch_size, d, d, device=flattened_vectors.device)
        matrix[:, indices[0], indices[1]] = flattened_vectors
        matrix[:, indices[1], indices[0]] = flattened_vectors  # Ensure symmetry

        return matrix

    def get_state_prediction(self, predictor, initial_state, time):
        """
        Use `predictor` to output a flattened lower‐triangular density matrix,
        reconstruct it, exponentiate to get U = exp(-i·downscaling·time·rho_pred),
        then apply U to `initial_state`.

        - predictor: model on correct device
        - initial_state: (batch_size, D, D) on same device
        - time: (batch_size,) on same device
        Returns: evolved_state ∈ (batch_size, D, D) on the same device.
        """
        device = next(predictor.parameters()).device
        batch_size = initial_state.shape[0]

        # 1) Predictor → flattened lower‐triangular (batch_size, L)
        predictor_output_flat = predictor(batch_size=batch_size)
        # 2) Reconstruct full Hermitian (batch_size, D, D)
        predictor_output = self.reconstruct_density_matrix_from_lower(predictor_output_flat)
        predictor_output = predictor_output.to(device)

        # 3) Build H_evo = -i * downscaling * time * predictor_output
        time_expanded = time.view(batch_size, 1, 1).expand_as(predictor_output)
        H_evo = -1j * time_expanded * predictor_output * self.downscaling

        # 4) Exponentiate to get U on `device`
        U = torch.matrix_exp(H_evo)

        # 5) Evolve: ρ' = U · initial_state · U†
        initial_state = initial_state.to(device)
        U_dagger = U.conj().transpose(-2, -1)
        temp = torch.matmul(U, initial_state)
        evolved_state = torch.matmul(temp, U_dagger)
        return evolved_state

    def forward(self, predictor, time, input_state, target_indices, basis_indices):
        """
        - predictor: the Predictor model already on correct device
        - time: (batch_size,) on same device
        - input_state: (batch_size, D, D) on same device
        - target_indices: (batch_size,) ints for measurement outcome
        - basis_indices: (batch_size, num_qubits) ints for X/Y/Z
        """
        device = next(predictor.parameters()).device

        # 1) Predicted evolved state: (batch_size, D, D)
        predicted_state = self.get_state_prediction(predictor, input_state, time)

        # 2) Rotate into measurement basis
        predicted_state_rotated = self.apply_rotation(predicted_state, basis_indices)

        # 3) Compute measurement probabilities = |diag(ρ_rotated)|^2
        probs = torch.abs(torch.diagonal(predicted_state_rotated, dim1=-2, dim2=-1)) ** 2
        probs = probs / probs.sum(dim=1, keepdim=True)

        # 4) Gather target probabilities
        batch_idx = torch.arange(probs.size(0), device=device)
        target_probs = probs[batch_idx, target_indices]

        # 5) Clamp and compute negative log-likelihood
        eps = 1e-12
        target_probs = torch.clamp(target_probs, min=eps)
        loss = -torch.sum(torch.log(target_probs)) / target_probs.size(0)
        loss = loss / self.num_qubits

        return loss
