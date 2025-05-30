import torch
import numpy as np
from torch.utils.data import Dataset

class DataGen(Dataset):
    def __init__(self, times: list, num_measurements: int, shots: int, num_qubits: int, initial_state_indices = None, seed=1234, perturbations=2, perturbation_depth=2, hamiltonian=None):
        self.times = np.array(times[1:], dtype=np.float64).flatten()
        self.num_measurements = num_measurements
        self.shots = shots  # Number of times each measurement is performed
        self.num_qubits = num_qubits
        self.seed = seed
        self.perturbations = perturbations
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
            for _ in range(self.perturbations):
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