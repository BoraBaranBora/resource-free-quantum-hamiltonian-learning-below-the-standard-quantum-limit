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
            initial_state = self.prepare_initial_state_density_matrix(index)

            for _ in range(self.spreadings):
                # 2) Apply random gates to get “spread” initial state
                spread_state = self.apply_random_gates(initial_state.clone())

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
                            initial_states_list.append(initial_state)

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







class DataGen(Dataset):
    def __init__(self, times: list, num_measurements: int, shots: int, num_qubits: int, initial_state_indices = None, seed=1234, spreadings=1, perturbation_depth=2, hamiltonian=None):
        self.times = np.array(times[:], dtype=np.float64).flatten()
        self.num_measurements = num_measurements
        self.shots = shots  # Number of times each measurement is performed
        self.num_qubits = num_qubits
        self.seed = seed
        self.spreadings = spreadings
        self.perturbation_depth = perturbation_depth
        self.initial_state_indices = initial_state_indices
        self.H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / torch.sqrt(torch.tensor(2, dtype=torch.complex64))
        self.S_dagger = torch.tensor([[1, 0], [0, -1j]], dtype=torch.complex64)
        self.YH = torch.matmul(self.H, self.S_dagger)
        self.hamiltonian = hamiltonian
        np.random.seed(self.seed)  # Set numpy random seed for reproducibility

    def get_pauli_matrix(self, basis):
        if basis == 'X':
            return self.H
        elif basis == 'Y':
            return self.YH
        elif basis == 'Z':
            return torch.tensor([[1, 0], [0, 1+ 0j]], dtype=torch.complex64)

    def tensor_product(self, *matrices):
        result = matrices[0]
        for mat in matrices[1:]:
            result = torch.kron(result, mat)
        return result

    def measure_in_basis(self, rho, measurement_bases):
        rotation_operators = [self.get_pauli_matrix(basis) for basis in measurement_bases]
        composite_rotation = self.tensor_product(*rotation_operators)
        composite_dagger = torch.conj(torch.transpose(composite_rotation, 0, 1))

        rotated_rho = torch.matmul(torch.matmul(composite_rotation, rho), composite_dagger)
        probs = abs(torch.diag(rotated_rho))**2
        probs /= torch.sum(probs)

        measurement_outcomes_indices = torch.multinomial(probs, self.shots, replacement=True)
        return measurement_outcomes_indices

    def prepare_initial_state_density_matrix(self, index):
        state_vector = torch.zeros(2 ** self.num_qubits, dtype=torch.complex64)
        state_vector[index] = 1.0  # System entirely in the |index> state

        # Create the density matrix rho = |psi><psi|
        density_matrix = torch.outer(state_vector, torch.conj(state_vector))
        return density_matrix
    
    def get_lower_triangular_flattened(self, rho):
        """
        Extracts the lower triangular part of a complex density matrix (including the diagonal),
        separates the real and imaginary parts, and flattens them into a single real-valued tensor.
        """
        # Get the lower triangular indices
        dim = rho.shape[0]
        indices = torch.tril_indices(dim, dim)
        
        # Extract the lower triangular part for real and imaginary parts separately
        rho_real = rho.real
        rho_imag = rho.imag
        lower_tri_real = rho_real[indices[0], indices[1]]
        lower_tri_imag = rho_imag[indices[0], indices[1]]
        
        # Flatten and concatenate the real and imaginary parts
        lower_tri_flattened = torch.cat([lower_tri_real, lower_tri_imag], dim=-1)
        return lower_tri_flattened

    
    def apply_random_gates(self, state, repetitions=0):
        """
        Apply random single-qubit unitaries (according to Haar measure) to each qubit multiple times.

        Parameters:
        state (torch.Tensor): Initial state to apply the gates to.
        repetitions (int): Number of times to apply random unitaries.

        Returns:
        torch.Tensor: State after applying the random gates.
        """
        repetitions = self.perturbation_depth
        num_qubits = int(np.log2(state.shape[0]))
        
        def random_single_qubit_unitary():
            """ Generate a random single-qubit unitary matrix according to Haar measure. """
            #theta = np.random.uniform(0, np.pi)
            
            theta = np.arccos(1 - 2 * np.random.uniform(0, 1))
            
            phi = np.random.uniform(0, 2 * np.pi)
            lambd = np.random.uniform(0, 2 * np.pi)
            
            # Define rotation matrices
            RY = torch.tensor([[np.cos(theta / 2), -np.sin(theta / 2)], 
                            [np.sin(theta / 2), np.cos(theta / 2)]], dtype=torch.complex64)
            RZ_phi = torch.tensor([[np.exp(-1j * phi / 2), 0], 
                                [0, np.exp(1j * phi / 2)]], dtype=torch.complex64)
            RZ_lambda = torch.tensor([[np.exp(-1j * lambd / 2), 0], 
                                    [0, np.exp(1j * lambd / 2)]], dtype=torch.complex64)
            
            # The Haar unitary is RZ(λ) * RY(θ) * RZ(φ)
            return RZ_lambda @ RY @ RZ_phi

        for _ in range(repetitions):  # Apply the sequence multiple times
            for i in range(num_qubits):
                # Generate a random single-qubit unitary matrix
                matrix = random_single_qubit_unitary()

                # Construct the full gate for the multi-qubit state
                full_gate = 1
                for j in range(num_qubits):
                    if j == i:
                        full_gate = torch.kron(full_gate, matrix) if type(full_gate) != int else matrix
                    else:
                        full_gate = torch.kron(full_gate, torch.eye(2, dtype=torch.complex64)) if type(full_gate) != int else torch.eye(2, dtype=torch.complex64)

                # Apply the gate to the state (left and right multiplication for density matrix)
                state = full_gate @ state @ full_gate.conj().T
        
        return state

    def generate_dataset(self):
        times_list = []
        basis_list = []
        targets_list = []
        initial_states_list = []
        
        measurement_bases_samples = [np.random.choice(['X', 'Y', 'Z'], size=self.num_qubits) for _ in range(self.num_measurements)]

        for index in self.initial_state_indices:
            for _ in range(self.spreadings):
                basis_state = torch.zeros(2 ** self.num_qubits, dtype=torch.complex64)
                basis_state[index] = 1.0
                basis_state = torch.outer(basis_state, basis_state.conj())
                scattered_state = self.apply_random_gates(basis_state, repetitions=self.perturbation_depth)
                for time in self.times:
                    initial_state = scattered_state
                    evolved_state = self.evolve_state_with_hamiltonian(initial_state, time)
                    for m in range(self.num_measurements):
                        #measurement_bases = np.random.choice(['X', 'Y', 'Z'], size=self.num_qubits)
                        measurement_bases = measurement_bases_samples[m]
                        measurement_outcomes = self.measure_in_basis(evolved_state, measurement_bases)
                        # Extend lists to include data for each shot
                        for shot_index in measurement_outcomes:
                            times_list.append(time)  # Each shot has the same time stamp
                            basis_indices = [0 if b == 'X' else 1 if b == 'Y' else 2 for b in measurement_bases]
                            basis_list.append(basis_indices)  # Each shot has the same measurement basis configuration
                            targets_list.append(shot_index.item())  # Append the index of the measurement outcome
                            initial_states_list.append(initial_state)

        # Convert lists to PyTorch tensors to create the dataset
        times_tensor = torch.tensor(times_list, dtype=torch.float32)
        basis_tensor = torch.tensor(basis_list, dtype=torch.long)
        targets_tensor = torch.tensor(targets_list, dtype=torch.long)  # Targets are the indices of measurement outcomes
        initial_states_tensor = torch.stack(initial_states_list)
        
        print(f'initial_states_tensor shape: {initial_states_tensor.size()}')

        return targets_tensor, times_tensor, basis_tensor, initial_states_tensor

    def get_initial_state_indices(self):
        return self.initial_state_indices
    
    def evolve_state_with_hamiltonian(self, initial_state, time):
        """Evolve the state using the provided Hamiltonian."""
        if self.hamiltonian is None:
            raise ValueError("Hamiltonian must be provided for time evolution.")

        # Assuming we use the formula U = exp(-i H t), where H is the Hamiltonian
        H = self.hamiltonian  # Hamiltonian matrix
        U = torch.matrix_exp(-1j * H * time)  # Unitary evolution operator
        evolved_state = U @ initial_state @ U.conj().transpose(-2, -1)  # Apply the unitary evolution
        return evolved_state