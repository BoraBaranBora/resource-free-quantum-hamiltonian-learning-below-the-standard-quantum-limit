import torch
import numpy as np

def apply_kron(term):
    """Apply Kronecker product over a list of matrices."""
    result = term[0]
    for mat in term[1:]:
        result = torch.kron(result, mat) #  the first of the term list has to be on the right, to be mathematically correct in the end
    return result


def generate_hamiltonian_parameters(family, num_qubits, coupling_type="random", h_field_type="random", 
                                    coupling_std=0.1, h_field_std=0.1, include_transverse_field=True,
                                    include_higher_order=0, **kwargs):
    """
    Generate Hamiltonian parameters for different families with various configurations.
    
    Args:
    - family: String representing the Hamiltonian family (e.g., 'Ising', 'Heisenberg', 'Kitaev')
    - num_qubits: Number of qubits in the system
    - coupling_type: Type of coupling ('random', 'uniform', 'normal')
    - h_field_type: Type of transverse field ('random', 'uniform', 'normal')
    - coupling_std: Standard deviation for normal distribution of couplings (only if coupling_type='normal')
    - h_field_std: Standard deviation for normal distribution of transverse field (only if h_field_type='normal')
    - include_transverse_field: Whether to include the transverse field (boolean)
    - include_higher_order: Max order of interaction terms to include (1, 2, or 3)

    Returns:
    - Hamiltonian parameters (dictionary containing 'J' and 'h')
    """
    # Generate couplings
    if coupling_type == "random":
        J = np.random.uniform(-1, 1, size=(num_qubits, 3))  # Random couplings for Jx, Jy, Jz
    elif coupling_type == "uniform_random":
        uniform_value = np.random.uniform(-1, 1)  # Single random value
        J = np.full((num_qubits, 3), uniform_value)  # Same value for all qubits
    elif coupling_type == "normal":
        J = np.random.normal(1, coupling_std, size=(num_qubits, 3))  # Normal distribution for Jx, Jy, Jz
    elif coupling_type == "anisotropic":
        J_values = np.random.uniform(-1, 1, size=3)  # Different values for Jx, Jy, Jz
        J = np.tile(J_values, (num_qubits, 1))  # Repeat across all qubits
    elif coupling_type == "anisotropic_normal":
        # Each axis (Jx, Jy, Jz) has a different mean and normal distribution
        J_means = np.random.uniform(-1.0, 1.0, size=3)  # Random mean values for Jx, Jy, Jz
        J = np.random.normal(J_means, coupling_std, size=(num_qubits, 3))  # Normal distribution about these means

    # Generate transverse field
    h = None
    if include_transverse_field:
        if h_field_type == "random":
            h = np.random.uniform(-1, 1, size=num_qubits)   # Local transverse field values
        elif h_field_type == "uniform_random":
            uniform_value = np.random.uniform(-1, 1) 
            h = np.full(num_qubits, uniform_value)  # Uniform transverse field values
        elif h_field_type == "normal":
            h = np.random.normal(1, h_field_std, size=num_qubits)  # Normal distribution for transverse field
        elif h_field_type == "standard":
            h = np.ones(num_qubits)  # Normal distribution for transverse field

    return {
        'J': np.array(J),
        'h': np.array(h) if include_transverse_field else None}

def generate_hamiltonian(family, num_qubits, include_transverse_field=True, include_higher_order=0, **params):
    """
    Generate a Hamiltonian matrix based on the specified family and parameters.
    
    Args:
    - family: Hamiltonian family type (e.g., 'Ising', 'Heisenberg', 'Kitaev')
    - num_qubits: Number of qubits in the system
    - include_transverse_field: Whether to include the transverse field (boolean)
    - params: Hamiltonian-specific parameters (e.g., J, h, include_higher_order)

    Returns:
    - Hamiltonian matrix (torch.Tensor)
    """
    dim = 2 ** num_qubits
    H = torch.zeros((dim, dim), dtype=torch.complex64)

    # Define the Pauli matrices explicitly
    sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)  # X
    sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)  # Y
    sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)  # Z
    identity = torch.eye(2, dtype=torch.complex64)  # I
    #identity = torch.tensor([[1 + 0j, 0 + 0j], [0 + 0j, 1 + 0j]], dtype=torch.complex64)


    # Retrieve the Hamiltonian parameters
    J = np.array(params['J'])  # Shape: (num_qubits, 3)
    h = np.array(params['h']) if include_transverse_field and 'h' in params else None
    include_higher_order = params.get('include_higher_order', 0)

    # First-order terms
    if family == 'Heisenberg':
        for i in range(num_qubits - 1):
            term_x = [identity] * num_qubits
            term_y = [identity] * num_qubits
            term_z = [identity] * num_qubits
            term_x[i] = sigma_x
            term_x[i + 1] = sigma_x
            term_y[i] = sigma_y
            term_y[i + 1] = sigma_y
            term_z[i] = sigma_z
            term_z[i + 1] = sigma_z
            H -= J[i, 0] * apply_kron(term_x)
            H -= J[i, 1] * apply_kron(term_y)
            H -= J[i, 2] * apply_kron(term_z)

    # Add transverse field if included
    if include_transverse_field and h is not None:
        for i in range(num_qubits):
            term = [identity] * num_qubits
            term[i] = sigma_x
            H += h[i] * apply_kron(term)

    return H
