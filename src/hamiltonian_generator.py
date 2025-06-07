import torch
import numpy as np
import re

def apply_kron(term):
    """Apply Kronecker product over a list of 2×2 matrices (Pauli/identity)."""
    result = term[0]
    for mat in term[1:]:
        result = torch.kron(result, mat)
    return result


def generate_hamiltonian_parameters(
    family: str,
    num_qubits: int,
    coupling_type: str = "anisotropic_normal",
    h_field_type: str = "random",
    coupling_std: float = 0.1,
    h_field_std: float = 0.1,
    **kwargs
) -> dict:
    """
    Generate Hamiltonian parameters for “base” families like 'Heisenberg' or 'XYZ'
    plus a numeric suffix indicating higher‐order (e.g. 'XYZ2', 'Heisenberg3', ...).

    Returns a dict with:
      - 'base_family': either 'Heisenberg' or 'XYZ'
      - 'include_higher_order': integer parsed from suffix (0 if none)
      - 'J': np.array shape (num_qubits,3)   ← NN couplings
      - 'h_x': np.array shape (num_qubits,)  ← on‐site X‐field
      - 'h_y', 'h_z': np.array shape (num_qubits,) or None  ← on‐site Y/Z‐fields if requested
      - 'K': np.array shape (num_qubits-2,3) or None  ← second‐order couplings (i,i+2)
      - 'L': np.array shape (num_qubits-2,3) or None  ← third‐order couplings (i,i+1,i+2)
    """
    # ――― 1) Parse “family” string for a base name + numeric suffix ―――
    m = re.fullmatch(r"([A-Za-z]+?)(\d+)$", family)
    if m:
        base_name = m.group(1)
        include_higher_order = int(m.group(2))
    else:
        base_name = family
        include_higher_order = 0

    # Only “Heisenberg” or “XYZ” are supported as bases
    if base_name not in {"Heisenberg", "XYZ"}:
        raise ValueError(f"Unknown base family '{base_name}'; only 'Heisenberg' or 'XYZ' are allowed.")

    # ――― 2) Generate first‐order (NN) couplings J[i,(x,y,z)] ―――
    if coupling_type == "random":
        J = np.random.uniform(-1, 1, size=(num_qubits, 3))
    elif coupling_type == "uniform_random":
        uni = np.random.uniform(-1, 1)
        J = np.full((num_qubits, 3), uni)
    elif coupling_type == "normal":
        J = np.random.normal(1.0, coupling_std, size=(num_qubits, 3))
    elif coupling_type == "anisotropic":
        base = np.random.uniform(-1, 1, size=3)
        J = np.tile(base, (num_qubits, 1))
    elif coupling_type == "anisotropic_normal":
        means = np.random.uniform(-1.0, 1.0, size=3)
        J = np.random.normal(means, coupling_std, size=(num_qubits, 3))
    else:
        raise ValueError(f"Unknown coupling_type = {coupling_type}")

    # ――― 3) Generate on‐site X‐field h_x[i] ALWAYS (so we can always add at least σ⁽ˣ⁾ terms) ―――
    if h_field_type == "random":
        h_x = np.random.uniform(-1, 1, size=(num_qubits,))
    elif h_field_type == "uniform_random":
        uni = np.random.uniform(-1, 1)
        h_x = np.full((num_qubits,), uni)
    elif h_field_type == "normal":
        h_x = np.random.normal(1.0, h_field_std, size=(num_qubits,))
    elif h_field_type == "standard":
        h_x = np.ones((num_qubits,))
    else:
        raise ValueError(f"Unknown h_field_type = {h_field_type}")

    # ――― 4) If include_higher_order ≥ 2, also generate h_y, h_z for full on‐site field ―――
    h_y = None
    h_z = None
    if include_higher_order >= 2:
        # We simply mimic the same “h_field_type” logic for Y and Z
        if h_field_type == "random":
            h_y = np.random.uniform(-1, 1, size=(num_qubits,))
            h_z = np.random.uniform(-1, 1, size=(num_qubits,))
        elif h_field_type == "uniform_random":
            uni = np.random.uniform(-1, 1)
            h_y = np.full((num_qubits,), uni)
            h_z = np.full((num_qubits,), uni)
        elif h_field_type == "normal":
            h_y = np.random.normal(1.0, h_field_std, size=(num_qubits,))
            h_z = np.random.normal(1.0, h_field_std, size=(num_qubits,))
        elif h_field_type == "standard":
            h_y = np.ones((num_qubits,))
            h_z = np.ones((num_qubits,))
        else:
            # (Should never reach here because we already handled unknown types above)
            pass

    # ――― 5) Generate K (second‐order, i↔i+2) if include_higher_order ≥ 2 ―――
    K = None
    if include_higher_order >= 2:
        if coupling_type in {"random", "uniform_random"}:
            K = np.random.uniform(-1, 1, size=(num_qubits - 2, 3))
        elif coupling_type == "normal":
            K = np.random.normal(1.0, coupling_std, size=(num_qubits - 2, 3))
        elif coupling_type == "anisotropic":
            baseK = np.random.uniform(-1, 1, size=3)
            K = np.tile(baseK, (num_qubits - 2, 1))
        elif coupling_type == "anisotropic_normal":
            meansK = np.random.uniform(-1.0, 1.0, size=3)
            K = np.random.normal(meansK, coupling_std, size=(num_qubits - 2, 3))
        else:
            # fallback to uniform
            K = np.random.uniform(-1, 1, size=(num_qubits - 2, 3))

    # ――― 6) Placeholder for L (third‐order, i,i+1,i+2) if include_higher_order ≥ 3 ―――
    L = None
    if include_higher_order >= 3:
        # Example: one XXX, YYY, ZZZ coefficient per triple
        L = np.random.uniform(-1, 1, size=(num_qubits - 2, 3))

    return {
        "base_family": base_name,
        "include_higher_order": include_higher_order,
        "J": np.array(J),
        "h_x": np.array(h_x),
        "h_y": np.array(h_y) if (h_y is not None) else None,
        "h_z": np.array(h_z) if (h_z is not None) else None,
        "K": np.array(K) if (K is not None) else None,
        "L": np.array(L) if (L is not None) else None
    }


def generate_hamiltonian(
    family: str,
    num_qubits: int,
    device: torch.device,
    **params
) -> torch.Tensor:
    """
    Build the Heisenberg/XYZ Hamiltonian up to the order encoded in `family`.
    If family="XYZ2", then include NN (i,i+1) + NNN (i,i+2) + full on‐site {X,Y,Z} fields.
    If family="XYZ3", also include a 3‐body (i,i+1,i+2) triple‐product, etc.

    The parameter dict returned by `generate_hamiltonian_parameters` must be passed here.

    Returns:
      H ∈ ℂ^(2^num_qubits × 2^num_qubits)
    """
    # ――― 1) Re‐parse family to discover base name + numeric suffix ―――
    m = re.fullmatch(r"([A-Za-z]+?)(\d+)$", family)
    if m:
        base_name = m.group(1)
        include_higher_order = int(m.group(2))
    else:
        base_name = family
        include_higher_order = 0

    if base_name not in {"Heisenberg", "XYZ"}:
        raise ValueError(f"Unknown base family '{base_name}' in generate_hamiltonian.")

    dim = 2 ** num_qubits
    H = torch.zeros((dim, dim), dtype=torch.complex64, device=device)
    
    # Pauli matrices on the correct device
    sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
    sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
    sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)
    identity = torch.eye(2, dtype=torch.complex64, device=device)

    # ――― 2) Extract J, h_x, h_y, h_z, K, L from params ―――
    J   = np.array(params["J"])                        # shape (num_qubits, 3)
    h_x = np.array(params.get("h_x"))                   # shape (num_qubits,)
    h_y = np.array(params.get("h_y")) if params.get("h_y") is not None else None
    h_z = np.array(params.get("h_z")) if params.get("h_z") is not None else None
    K   = np.array(params.get("K")) if params.get("K") is not None else None
    L   = np.array(params.get("L")) if params.get("L") is not None else None
    

    # ――― 3) Nearest‐neighbor Heisenberg/XYZ (i,i+1) block ―――
    for i in range(num_qubits - 1):
        tx = [identity] * num_qubits
        tx[i] = sigma_x
        tx[i + 1] = sigma_x

        ty = [identity] * num_qubits
        ty[i] = sigma_y
        ty[i + 1] = sigma_y

        tz = [identity] * num_qubits
        tz[i] = sigma_z
        tz[i + 1] = sigma_z

        H -= J[i, 0] * apply_kron(tx)
        H -= J[i, 1] * apply_kron(ty)
        H -= J[i, 2] * apply_kron(tz)

    # ――― 4) On‐site X‐field ∑ₙ h_x[n]·σₙˣ ALWAYS (if h_x is not None) ―――
    if h_x is not None:
        for i in range(num_qubits):
            term = [identity] * num_qubits
            term[i] = sigma_x
            H += h_x[i] * apply_kron(term)

    # ――― 5) If include_higher_order ≥ 2, also add on‐site Y & Z fields ―――
    if include_higher_order >= 2:
        if h_y is not None:
            for i in range(num_qubits):
                term = [identity] * num_qubits
                term[i] = sigma_y
                H += h_y[i] * apply_kron(term)
        if h_z is not None:
            for i in range(num_qubits):
                term = [identity] * num_qubits
                term[i] = sigma_z
                H += h_z[i] * apply_kron(term)

    # ――― 6) Next‐nearest‐neighbor (i,i+2) if include_higher_order ≥ 2 ―――
    if (include_higher_order >= 2) and (K is not None):
        for i in range(num_qubits - 2):
            txx = [identity] * num_qubits
            txx[i] = sigma_x
            txx[i + 2] = sigma_x

            tyy = [identity] * num_qubits
            tyy[i] = sigma_y
            tyy[i + 2] = sigma_y

            tzz = [identity] * num_qubits
            tzz[i] = sigma_z
            tzz[i + 2] = sigma_z

            H -= K[i, 0] * apply_kron(txx)
            H -= K[i, 1] * apply_kron(tyy)
            H -= K[i, 2] * apply_kron(tzz)

    # ――― 7) Third‐order “XYZ3” triple‐product if include_higher_order ≥ 3 ―――
    if (include_higher_order >= 3) and (L is not None):
        for i in range(num_qubits - 2):
            txxx = [identity] * num_qubits
            txxx[i] = sigma_x
            txxx[i + 1] = sigma_x
            txxx[i + 2] = sigma_x

            tyyy = [identity] * num_qubits
            tyyy[i] = sigma_y
            tyyy[i + 1] = sigma_y
            tyyy[i + 2] = sigma_y

            tzzz = [identity] * num_qubits
            tzzz[i] = sigma_z
            tzzz[i + 1] = sigma_z
            tzzz[i + 2] = sigma_z

            H -= L[i, 0] * apply_kron(txxx)
            H -= L[i, 1] * apply_kron(tyyy)
            H -= L[i, 2] * apply_kron(tzzz)

    return H
