
import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import scoreatpercentile
from collections import defaultdict

from sklearn.linear_model import LinearRegression

from predictor import Predictor
from hamiltonian_generator import generate_hamiltonian

# Detect device once for all operations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_times(alpha, N, delta_t):
    return [delta_t * (k**alpha) for k in range(1, N+1)]


def reconstruct_density_matrix_from_lower(flattened_vector: torch.Tensor) -> torch.Tensor:
    """
    Given a flattened lower‐triangular vector (including diagonal) for a d×d symmetric matrix,
    rebuild the full d×d real matrix.
    """
    flattened_length = flattened_vector.numel()
    d = int((-1 + (1 + 8 * flattened_length) ** 0.5) / 2)
    indices = torch.tril_indices(d, d, device=flattened_vector.device)

    real_matrix = torch.zeros((d, d),
                              device=flattened_vector.device,
                              dtype=flattened_vector.dtype)
    real_matrix[indices[0], indices[1]] = flattened_vector
    real_matrix[indices[1], indices[0]] = flattened_vector
    return real_matrix


def collect_recovery_errors_from_data(
    base_run_dir: str,
    scaling_param: str = "times",      # "times" or "spreading"
    group_by: str = "spreading",          # "alpha", "times", or "spreading"

) -> dict:
    """
    Walk through each “combo” subdirectory under base_run_dir. For each Hamiltonian:
      - Load config.json and hamiltonians.json
      - Reconstruct predictor, generate true Hamiltonian, compute error
      - Use scaling_param (either "times" or "spreading") as the top‐level key (tuple of values)
      - Use group_by (either "alpha", "times", or "spreading") as the third element in each tuple

    scaling_param: str, must be "times" or "spreading". The code will read meta_params[scaling_param],
                   convert it to a tuple, and use that as the dict key.
    group_by:      str, must be "alpha", "times", or "spreading", and must differ from scaling_param.
                   This value becomes the third element in each (error, true_family, group_key).
    Returns:
      errors_by_scaling: { scaling_tuple → [ (error, true_family, group_key) ] }
    """
    if scaling_param not in {"times", "spreading"}:
        raise ValueError("scaling_param must be 'times' or 'spreading'")
    if group_by not in {"alpha", "times", "spreading", "num_qubits"}:
        raise ValueError("group_by must be one of 'alpha', 'times', 'spreading', or 'num_qubits'")
    if group_by == scaling_param:
        raise ValueError("group_by must differ from scaling_param")


    errors_by_scaling = {}

    for combo_folder in os.listdir(base_run_dir):
        combo_path = os.path.join(base_run_dir, combo_folder)
        if not os.path.isdir(combo_path):
            continue

        config_path       = os.path.join(combo_path, "config.json")
        hamiltonians_path = os.path.join(combo_path, "hamiltonians.json")
        if not (os.path.exists(config_path) and os.path.exists(hamiltonians_path)):
            print(f"Skipping {combo_folder}: missing config.json or hamiltonians.json")
            continue

        with open(config_path, "r") as f:
            meta_params = json.load(f)

                
        # ─── Extract alpha ───
        alpha = meta_params.get("alpha", None)
        
        # ─── Extract times, falling back to generate_times if missing ───
        raw_times = meta_params.get("times", None)
        
        if isinstance(raw_times, (list, tuple)) and len(raw_times) > 0:
            time_values = list(raw_times)
        else:
            # fallback: build times=[0.0, δt*1**α, …, δt*N**α]
            steps = meta_params.get("steps", None)
            delta_t = meta_params.get("delta_t", None)

            print(f"No valid 'times' in {combo_folder}; generating with α={alpha:.3f}, steps={steps}, δt={delta_t}")
            time_values = generate_times(alpha, steps, delta_t)

        time_tuple = tuple(time_values)

        # ─── Extract spreading (for "spreading" grouping or grouping by spreading) ───
        raw_spreading = meta_params.get("spreading", None)
        if raw_spreading is None:
            spreading_values = []
        elif isinstance(raw_spreading, (list, tuple)):
            if len(raw_spreading) == 0:
                print(f"Skipping {combo_folder}: config.json has empty 'spreading' list.")
                continue
            spreading_values = list(raw_spreading)
        else:
            spreading_values = [raw_spreading]
        spreading_tuple = tuple(spreading_values)



        num_qubits = meta_params.get("num_qubits", 5)

        # Reconstruct activation function
        raw_act = meta_params.get("activation", "Tanh")
        if "'" in raw_act:
            inner = raw_act.split("'")[1]
            activation_name = inner.split(".")[-1]
        elif "." in raw_act:
            activation_name = raw_act.split(".")[-1]
        else:
            activation_name = raw_act

        if not hasattr(nn, activation_name):
            raise RuntimeError(f"Cannot find activation '{activation_name}' in torch.nn")
        activation_fn = getattr(nn, activation_name)

        # Build predictor dimensions
        d = 2 ** num_qubits
        tri_len = d * (d + 1) // 2
        input_size = tri_len
        output_size = tri_len
        hidden_layers = meta_params.get("hidden_layers", [200, 200, 200])

        # ─── Decide the scaling_list and scaling_tuple ───
        if scaling_param == "times":
            scaling_list = time_values       # iterate over time‐stamps if needed
        else:  # scaling_param == "spreading"
            scaling_list = spreading_values

        if len(scaling_list) == 0:
            print(f"Skipping {combo_folder}: no values found for '{scaling_param}'.")
            continue

        scaling_tuple = tuple(scaling_list)
        errors_by_scaling.setdefault(scaling_tuple, [])

        # ─── Read the list of Hamiltonians ───
        with open(hamiltonians_path, "r") as f:
            hamiltonians_list = json.load(f)

        # ─── Loop over each Hamiltonian info dict ───
        for ham_info in hamiltonians_list:
            true_family    = ham_info["family"]
            ham_parameters = ham_info["params"]
            codified_name  = ham_info["name"]

            # Embedding filename is embedding_<codename>.pth
            model_fname = f"embedding_{codified_name}.pth"
            model_path  = os.path.join(combo_path, model_fname)
            if not os.path.isfile(model_path):
                # No saved embedding → skip
                continue

            # Reconstruct the same Predictor used at training
            predictor = Predictor(
                input_size    = input_size,
                output_size   = output_size,
                hidden_layers = hidden_layers,
                activation_fn = activation_fn,
                ignore_input  = True,
                device        = device
            )
            #predictor.load_state_dict(torch.load(model_path))
            #predictor.eval()
            
            state = torch.load(model_path, map_location=device)
            predictor.load_state_dict(state)

            # Generate the “true” Hamiltonian matrix once
            original_ham =  generate_hamiltonian(
                true_family,
                num_qubits,
                device=device,
                **ham_parameters
            ).to(torch.complex64).to(device)
            
            with torch.no_grad():
                out_flat = predictor(batch_size=1).squeeze(0)
                rec_matrix = reconstruct_density_matrix_from_lower(out_flat).to(torch.complex64) / 1.0 # rescale, because loss had factor of 0.25
                error = torch.mean((rec_matrix - original_ham).abs()).item()#/(2**num_qubits)
                

            # ─── Decide the group_key based on group_by ───
            if group_by == "alpha":
                group_key = alpha
            elif group_by == "spreading":
                group_key = spreading_tuple
            elif group_by == "times":
                group_key = time_tuple
            elif group_by == "num_qubits":
                group_key = num_qubits
            else:
                raise ValueError(f"Unknown group_by value: {group_by}")


            # Append (error, true_family, group_key) under scaling_tuple
            errors_by_scaling[scaling_tuple].append((error, true_family, group_key))

    return errors_by_scaling



def compute_betas_from_errors(
    errors_by_time: dict,
    scaling_param: str = 'spreading',
    include_families: list = None,
    exclude_x_scale: set = None,
    exclude_above_one: bool = False,
    verbose: bool = False,
):
    """
    Computes power-law exponents (betas) for each family and key.
    Model: error ≈ a * (sum(time_stamps))^b

    Returns:
        dict: family → (keys_sorted, betas_array, beta_errs_array)
    """
    if exclude_x_scale is None:
        exclude_x_scale = set()

    def _power(x, a, b):
        return a * np.power(x, -b)

    all_keys = sorted({key for errs in errors_by_time.values() for (_, _, key) in errs})
    all_fams = sorted({
        fam for errs in errors_by_time.values()
             for (_, fam, _) in errs
             if include_families is None or fam in include_families
    })

    results = {}

    for fam in all_fams:
        fit_data = {k: {"x": [], "y": []} for k in all_keys}

        for time_tuple, recs in errors_by_time.items():
            ssum = round(sum(time_tuple), 8)
            if ssum in exclude_x_scale:
                continue

            for err, fam0, key in recs:
                if fam0 != fam:
                    continue

                if scaling_param == 'times' and exclude_above_one:
                    if err >= 50.0:
                        continue

                if scaling_param == 'spreading':
                    keep = err < 10.0 if ssum >= 50 else True
                else:
                    keep = err < 10.0 if ssum >= 2 else True

                if not keep:
                    continue

                fit_data[key]["x"].append(ssum)
                fit_data[key]["y"].append(err)

        betas, beta_errs = [], []
        for key in all_keys:
            fx = np.array(fit_data[key]["x"], float)
            fy = np.array(fit_data[key]["y"], float)

            if fx.size < 2 or fy.size < 2:
                betas.append(np.nan)
                beta_errs.append(np.nan)
                continue

            # Simple percentile filtering (optional)
            q0 = scoreatpercentile(fy, 0)
            q50 = scoreatpercentile(fy, 100)
            filt_mask = (fy >= q0) & (fy <= q50)
            fx, fy = fx[filt_mask], fy[filt_mask]

            idx = np.argsort(fx)
            fx, fy = fx[idx], fy[idx]

            try:
                (_, b_fit), pcov = curve_fit(_power, fx, fy, p0=(1.0, -0.5))
                sigma_b = np.sqrt(pcov[1, 1]) if pcov.shape == (2, 2) else np.nan
            except Exception as e:
                if verbose:
                    print(f"[WARN] Curve fit failed for family={fam}, key={key}: {e}")
                b_fit, sigma_b = np.nan, np.nan

            if verbose:
                print(f"[DEBUG] {fam} | key={key}: beta={b_fit:.3f} ± {sigma_b:.3f} (N={len(fx)})")

            betas.append(float(b_fit))
            beta_errs.append(float(sigma_b))

        results[fam] = (
            np.array(all_keys,   dtype=float),
            np.array(betas,      dtype=float),
            np.array(beta_errs,  dtype=float),
        )

    return results


