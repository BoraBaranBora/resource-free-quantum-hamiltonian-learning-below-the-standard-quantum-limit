import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, num_qubits, downscaling=0.25):
        super(Loss, self).__init__()
        
        self.num_qubits = num_qubits
        self.max_trace = 2 ** num_qubits
        self.downscaling = downscaling
        
        # Initialize Pauli matrices and other necessary constants
        self.H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / torch.sqrt(torch.tensor(2, dtype=torch.complex64))
        self.S_dagger = torch.tensor([[1, 0], [0, -1j]], dtype=torch.complex64)
        self.YH = torch.matmul(self.H, self.S_dagger)

        self.trace_penalty_weight = 1.0  # Adjust as needed
        self.pos_semi_def_penalty_weight = 1.0  # Adjust as needed

    def tensor_product(self, *matrices):
        """Compute tensor product of multiple matrices."""
        result = matrices[0]
        for mat in matrices[1:]:
            result = torch.kron(result, mat)
        return result

    def apply_rotation(self, rho, basis_indices):
        """Apply rotations based on basis indices."""
        batch_size = rho.size(0)
        rotation_operators = [
            self.tensor_product(*[self.get_pauli_matrix(basis_indices[i][j]) for j in range(self.num_qubits)])
            for i in range(batch_size)
        ]
        rotation_operators = torch.stack(rotation_operators)
        U_dagger = torch.conj(rotation_operators.transpose(-2, -1))
        rho_rotated = torch.matmul(rotation_operators, torch.matmul(rho, U_dagger))
        return rho_rotated


    def get_pauli_matrix(self, index):
        """Get the Pauli matrix (X, Y, or I) for a given index."""
        pauli_matrices = [
            self.H,  # X
            self.YH,  # Y
            torch.tensor([[1, 0], [0, 1]], dtype=torch.complex64),  # Identity
        ]
        return pauli_matrices[index]
    
    def get_lower_triangular_flattened(self, rho):
        """Extract the lower triangular part of a matrix and flatten it."""
        batch_size, dim, _ = rho.shape
        indices = torch.tril_indices(dim, dim, device=rho.device)
        lower_tri = rho[:, indices[0], indices[1]]
        return lower_tri

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
        """Get the predicted state using the model."""
        batch_size = initial_state.shape[0]
        predictor_output_flat = predictor(batch_size=batch_size)
        predictor_output = self.reconstruct_density_matrix_from_lower(predictor_output_flat)

        time_expanded = time.view(-1, 1, 1).expand_as(predictor_output)
        #time_expanded = time.expand_as(predictor_output)
        hamiltonian_evolution = -1j * time_expanded * predictor_output * self.downscaling
        unitary_operator = torch.matrix_exp(hamiltonian_evolution)
        evolved_state = torch.matmul(torch.matmul(unitary_operator, initial_state), unitary_operator.conj().transpose(-2, -1))
        return evolved_state

    def forward(self, predictor, time, input_state, target_indices, basis_indices):
        """Compute the loss."""
        predicted_state = self.get_state_prediction(predictor, input_state, time)
        predicted_state_rotated = self.apply_rotation(predicted_state, basis_indices)
        
        # Calculate probabilities (diagonal elements squared)
        probabilities = torch.abs(torch.diagonal(predicted_state_rotated, dim1=-2, dim2=-1))**2
        probabilities /= torch.sum(probabilities, dim=1, keepdim=True)  # Normalize

        # Calculate target probabilities
        batch_indices = torch.arange(probabilities.size(0), device=probabilities.device)
        target_probabilities = probabilities[batch_indices, target_indices]

        # Clamp probabilities to avoid log of zero
        epsilon = 1e-12
        target_probabilities = torch.clamp(target_probabilities, min=epsilon)

        # Compute negative log-likelihood
        loss = -torch.sum(torch.log(target_probabilities)) / len(target_probabilities) 
        
        loss = loss/self.num_qubits#
        
        return loss
    