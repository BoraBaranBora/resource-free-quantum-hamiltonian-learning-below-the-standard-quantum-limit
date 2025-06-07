import torch
import numpy as np
from torch.utils.data import Dataset

class DataGen(Dataset):
    def __init__(
        self,
        times: list,
        num_measurements: int,
        shots: int,
        num_qubits: int,
        initial_state_indices=None,
        seed=1234,
        spreadings=2,
        perturbation_depth=2,
        hamiltonian=None
    ):
        self.times = np.array(times, dtype=np.float64).flatten()
        self.num_measurements = num_measurements
        self.shots = shots
        self.num_qubits = num_qubits
        self.seed = seed
        self.spreadings = spreadings
        self.perturbation_depth = perturbation_depth
        self.initial_state_indices = initial_state_indices

        # Store the Hamiltonian (should already be on the correct device)
        self.hamiltonian = hamiltonian
        if self.hamiltonian is None:
            raise ValueError("Hamiltonian must be provided for DataGen.")
        self.device = self.hamiltonian.device

        # Predefine Pauli‐like rotations on CPU; we will move them to `self.device` when needed
        self._H_cpu = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / torch.sqrt(torch.tensor(2, dtype=torch.complex64))
        self._S_dagger_cpu = torch.tensor([[1, 0], [0, -1j]], dtype=torch.complex64)
        self._YH_cpu = self._H_cpu @ self._S_dagger_cpu

        np.random.seed(self.seed)  # For reproducibility

    def get_pauli_matrix(self, basis):
        """
        Return the single‐qubit “rotation” for basis ∈ {X, Y, Z}, moved onto `self.device`.
        - X → H (on cosθ=π/4) 
        - Y → YH 
        - Z → identity
        """
        if basis == 'X':
            return self._H_cpu.to(self.device)
        elif basis == 'Y':
            return self._YH_cpu.to(self.device)
        elif basis == 'Z':
            return torch.tensor([[1, 0], [0, 1]], dtype=torch.complex64, device=self.device)

    def tensor_product(self, *matrices):
        """
        Kron‐multiply a list of 2×2 matrices (all should live on `self.device`).
        """
        result = matrices[0]
        for mat in matrices[1:]:
            result = torch.kron(result, mat)
        return result

    def measure_in_basis(self, rho, measurement_bases):
        """
        Rotate density matrix `rho` into the chosen single‐qubit bases, then
        compute measurement outcome indices via multinomial sampling. All work
        is done on `self.device`.
        """
        # Build list of 2×2 rotations (each on device)
        rotation_ops = [self.get_pauli_matrix(b) for b in measurement_bases]
        composite_rotation = self.tensor_product(*rotation_ops)
        composite_dagger = composite_rotation.conj().transpose(-2, -1)

        # rho should already be on the correct device
        rotated_rho = composite_rotation @ rho @ composite_dagger

        # Probabilities = diagonal of rotated_rho
        probs = torch.abs(torch.diag(rotated_rho)) ** 2
        probs = probs / probs.sum()

        # Sample `self.shots` outcomes
        outcome_indices = torch.multinomial(probs, self.shots, replacement=True)
        return outcome_indices

    def prepare_initial_state_density_matrix(self, index):
        """
        Create |index⟩⟨index| on `self.device`.
        """
        dim = 2 ** self.num_qubits
        state_vector = torch.zeros(dim, dtype=torch.complex64, device=self.device)
        state_vector[index] = 1.0
        density_matrix = state_vector.unsqueeze(-1) @ state_vector.conj().unsqueeze(0)
        return density_matrix

    def get_lower_triangular_flattened(self, rho):
        """
        Extract the lower triangular part of complex matrix `rho`, flatten real+imag parts.
        """
        dim = rho.shape[0]
        indices = torch.tril_indices(dim, dim, device=rho.device)
        rho_real = rho.real
        rho_imag = rho.imag
        lower_real = rho_real[indices[0], indices[1]]
        lower_imag = rho_imag[indices[0], indices[1]]
        return torch.cat([lower_real, lower_imag], dim=-1)

    def apply_random_gates(self, state, repetitions=0):
        """
        Apply random Haar‐distributed single‐qubit unitaries `self.perturbation_depth` times.
        Operates on a density matrix `state` already on `self.device`.
        """
        repetitions = self.perturbation_depth
        num_qubits = int(np.log2(state.shape[0]))
        device = self.device

        def random_single_qubit_unitary():
            """Generate one random single‐qubit unitary on CPU, then move to `device`."""
            theta = np.arccos(1 - 2 * np.random.uniform(0, 1))
            phi = np.random.uniform(0, 2 * np.pi)
            lambd = np.random.uniform(0, 2 * np.pi)

            RY_cpu = torch.tensor([
                [np.cos(theta / 2), -np.sin(theta / 2)],
                [np.sin(theta / 2),  np.cos(theta / 2)]
            ], dtype=torch.complex64)
            RZ_phi_cpu = torch.tensor([
                [np.exp(-1j * phi / 2), 0],
                [0, np.exp(1j * phi / 2)]
            ], dtype=torch.complex64)
            RZ_lambda_cpu = torch.tensor([
                [np.exp(-1j * lambd / 2), 0],
                [0, np.exp(1j * lambd / 2)]
            ], dtype=torch.complex64)

            # U = RZ(λ) · RY(θ) · RZ(φ)
            U_cpu = RZ_lambda_cpu @ RY_cpu @ RZ_phi_cpu
            return U_cpu.to(device)

        for _ in range(repetitions):
            for i in range(num_qubits):
                # Build full n‐qubit gate
                U_single = random_single_qubit_unitary()  # 2×2 on `device`
                full_gate = None

                # Tensor‐product structure: identity ⊗ … ⊗ U_single ⊗ … ⊗ identity
                for q in range(num_qubits):
                    if q == i:
                        mat = U_single
                    else:
                        mat = torch.eye(2, dtype=torch.complex64, device=device)
                    full_gate = mat if (full_gate is None) else torch.kron(full_gate, mat)

                # Apply to density matrix: ρ → U ρ U†
                state = full_gate @ state @ full_gate.conj().transpose(-2, -1)

        return state

    def generate_dataset(self):
        """
        - For each initial‐state index, apply `spreadings` random‐gate sequences.
        - For each time in `self.times`, compute evolved density matrix.
        - Measure `self.num_measurements` times in a random basis.
        - Return (targets, times, basis_indices, initial_states).
        """
        times_list = []
        basis_list = []
        targets_list = []
        initial_states_list = []

        # Pre‐sample measurement bases (so each shot set uses the same basis for all times)
        measurement_bases_samples = [
            np.random.choice(['X', 'Y', 'Z'], size=self.num_qubits)
            for _ in range(self.num_measurements)
        ]

        for index in self.initial_state_indices:
            # 1) Build the pure‐state density matrix |index⟩⟨index| on `self.device`
            pure_dm = self.prepare_initial_state_density_matrix(index)

            for _ in range(self.spreadings):
                # 2) Apply random gates to get “spread” initial state
                spread_state = self.apply_random_gates(pure_dm.clone())

                for time in self.times:
                    # 3) Evolve via Hamiltonian (on correct device)
                    evolved_state = self.evolve_state_with_hamiltonian(spread_state, time)

                    for m in range(self.num_measurements):
                        measurement_bases = measurement_bases_samples[m]
                        outcomes = self.measure_in_basis(evolved_state, measurement_bases)

                        # Record each “shot” outcome
                        for shot_idx in outcomes:
                            times_list.append(time)
                            # Map ['X','Y','Z'] → [0,1,2]
                            basis_indices = [
                                0 if b == 'X' else 1 if b == 'Y' else 2
                                for b in measurement_bases
                            ]
                            basis_list.append(basis_indices)
                            targets_list.append(int(shot_idx))
                            initial_states_list.append(evolved_state)

        # Convert lists into tensors
        times_tensor = torch.tensor(times_list, dtype=torch.float32, device=self.device)
        basis_tensor = torch.tensor(basis_list, dtype=torch.long, device=self.device)
        targets_tensor = torch.tensor(targets_list, dtype=torch.long, device=self.device)
        initial_states_tensor = torch.stack(initial_states_list).to(self.device)

        return targets_tensor, times_tensor, basis_tensor, initial_states_tensor

    def get_initial_state_indices(self):
        return self.initial_state_indices

    def evolve_state_with_hamiltonian(self, initial_state, time):
        """
        Evolve the state `initial_state` (density‐matrix) by U = exp(-i H t).
        Both `H` and `initial_state` live on `self.device`.
        """
        H = self.hamiltonian  # Already on device
        device = H.device

        # Ensure initial_state is on correct device
        initial_state = initial_state.to(device)

        # Compute U = exp(-i H t)
        U = torch.matrix_exp(-1j * H * time)

        # Evolve:  ρ(t) = U ρ(0) U†
        evolved_state = U @ initial_state @ U.conj().transpose(-2, -1)
        return evolved_state



