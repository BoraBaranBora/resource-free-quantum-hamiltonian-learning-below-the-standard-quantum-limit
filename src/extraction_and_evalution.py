
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
    group_by: str = "spreading"          # "alpha", "times", or "spreading"
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
    if group_by not in {"alpha", "times", "spreading"}:
        raise ValueError("group_by must be 'alpha', 'times', or 'spreading'")
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

        # ─── Extract times (for "times" grouping or grouping by time) ───
        #raw_times = meta_params.get("times", None)
        #if raw_times is None or not isinstance(raw_times, (list, tuple)) or len(raw_times) == 0:
        #    print(f"Skipping {combo_folder}: config.json has no valid 'times' list.")
        #    continue
        #time_values = list(raw_times)       # e.g. [0.0, 0.5, 1.0]
        #time_tuple  = tuple(time_values)
                
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
                error = torch.mean((rec_matrix - original_ham).abs()).item()

            # ─── Decide the group_key based on group_by ───
            if group_by == "alpha":
                group_key = alpha
            elif group_by == "spreading":
                group_key = spreading_tuple
            else:  # group_by == "times"
                group_key = time_tuple

            # Append (error, true_family, group_key) under scaling_tuple
            errors_by_scaling[scaling_tuple].append((error, true_family, group_key))

    return errors_by_scaling




def compute_betas_from_errors(
    errors_by_time: dict,
    scaling_param='spreading',
    include_families: list = None,
    exclude_x_scale: set = None,
    exclude_above_one = False,
):
    """
    Given:
      errors_by_time: {
        time_stamps_tuple → [ (error, true_family, key) ]
      }
    (where `key` is either a spreading value or an α value),
    this function computes, for each unique `key`, the power‐law exponent b
    in the fit err ≈ a * (sum(time_stamps))^b.

    Steps:
      1) For each (time_stamps, error_list) pair, compute ssum = round(sum(time_stamps), 8).
      2) Bucket all error values by their `key`.
      3) For each key:
           • For each ssum (total experiment time), gather all errs < 1.0.
           • Compute q0 = 0th percentile of that list, q50 = 50th percentile.
             Keep only values e with q0 ≤ e ≤ q50.
           • Append (ssum, e) for each remaining e into fit_x and fit_y.
      4) Perform curve_fit on (fit_x, fit_y) with model err = a * x^b.
      5) Return (keys_sorted, betas_array, beta_errs_array).

    Returns:
      keys_sorted:  sorted list of unique keys
      betas:        np.ndarray of fitted exponents b, in the same order
      beta_errs:    np.ndarray of uncertainties σ_b, in the same order
    """
    # (1) Identify all unique keys
    all_keys = sorted({ key for errs in errors_by_time.values() for (_, _, key) in errs })

    # Prepare a container for filtered data per key
    fit_data = { k: {"x": [], "y": []} for k in all_keys }

    # (2) Iterate over each time_stamps → error list
    for time_tuple, err_list in errors_by_time.items():
        ssum = round(sum(time_tuple), 8)

        # Build a temporary bucket of errs per key at this ssum
        bucket = {}
        for err, fam, key in err_list:
            if include_families is not None and fam not in include_families:
                continue
            bucket.setdefault(key, []).append(err)

        for key, errs_at_key in bucket.items():
            if exclude_x_scale and (ssum in exclude_x_scale):
                continue

            # (3a) Prefilter: keep only err < 1.0
            if scaling_param == 'times':
                if exclude_above_one:
                    pref = [e for e in errs_at_key if e < 1.0]
                    if len(pref) < 2:
                        continue
                elif ssum < 2:
                    pref = errs_at_key.copy()
                else:
                    pref = [e for e in errs_at_key if e < 0.1]
            else:
                #pref = errs_at_key.copy()
                if ssum < 50:
                    pref = errs_at_key.copy()
                else:
                    pref = [e for e in errs_at_key if e < 0.1]

            if len(pref) < 2:
                continue

            # (3b) Compute 0th and 50th percentiles, then percentile‐filter
            q25 = np.percentile(pref, 0)   # same as min(pref)
            q75 = np.percentile(pref, 50)  # median(pref)
            filt = [e for e in pref if (q25 <= e <= q75)]
            #if len(filt) < 2:
            #    continue

            # (3c) Append (ssum, e) for each e in filt
            fit_data[key]["x"].extend([ssum] * len(filt))
            fit_data[key]["y"].extend(filt)

    # (4) For each key, do a power‐law fit: err = a * x^b
    def _power(x, a, b):
        return a * np.power(x, b)

    betas = []
    beta_errs = []

    for key in all_keys:
        fx = np.array(fit_data[key]["x"], dtype=float)
        fy = np.array(fit_data[key]["y"], dtype=float)
        
        if fx.ndim == 0 or fy.ndim == 0:
            print(f" DEBUG: key={key!r} → fx.ndim={fx.ndim}, fx={fx}, fy.ndim={fy.ndim}, fy={fy}")


        if fx.size < 2 or fy.size < 2:
            betas.append(np.nan)
            beta_errs.append(np.nan)
            continue

        # Sort by x to stabilize the fit
        idx = np.argsort(fx)
        fx = fx[idx]
        fy = fy[idx]

        try:
            (a_fit, b_fit), pcov = curve_fit(_power, fx, fy, p0=(1.0, -0.5))
            # Uncertainty in b is sqrt of the corresponding diagonal element
            sigma_b = np.sqrt(pcov[1, 1]) if pcov.shape == (2, 2) else np.nan
        except Exception:
            b_fit = np.nan
            sigma_b = np.nan

        betas.append(float(b_fit))
        beta_errs.append(float(sigma_b))

    return np.array(all_keys,   dtype=float), \
           np.array(betas,      dtype=float), \
           np.array(beta_errs,  dtype=float)





def compute_betas_from_errors(
    errors_by_time: dict,
    scaling_param='spreading',
    include_families: list = None,
    exclude_x_scale: set = None,
    exclude_above_one: bool = False,
):
    # ... [snip docstring and setup] ...
    all_keys = sorted({ key for errs in errors_by_time.values() for (_, _, key) in errs })
    fit_data = { k: {"x": [], "y": []} for k in all_keys }

    def _power(x, a, b):
        return a * np.power(x, b)

    # (2) Iterate
    for time_tuple, err_list in errors_by_time.items():
        ssum = round(sum(time_tuple), 8)
        if exclude_x_scale and ssum in exclude_x_scale:
            continue

        # bucket by key
        bucket = {}
        for err, fam, key in err_list:
            if include_families and fam not in include_families:
                continue
            bucket.setdefault(key, []).append(err)

        for key, errs_at_key in bucket.items():
            # ——— Replace your old “prefilter” with this ———
            if exclude_above_one:
                # apply the same thresholds as plot_errors_by_spreadings
                thresh = 1.25 if ssum == 0.1 else 1.0
                pref = [e for e in errs_at_key if e < thresh]
            else:
                # keep all errs, then let percentile do the work
                pref = errs_at_key.copy()

            if len(pref) < 2:
                continue

            # (3b) percentile filter
            q0 = np.percentile(pref, 0)
            q50 = np.percentile(pref, 50)
            filt = [e for e in pref if q0 <= e <= q50]
            if len(filt) < 2:
                continue

            # (3c) collect for fit
            fit_data[key]["x"].extend([ssum]*len(filt))
            fit_data[key]["y"].extend(filt)

    # (4) Fit per key
    betas, beta_errs = [], []
    for key in all_keys:
        fx = np.array(fit_data[key]["x"], float)
        fy = np.array(fit_data[key]["y"], float)
        if fx.size < 2 or fy.size < 2:
            betas.append(np.nan); beta_errs.append(np.nan)
            continue

        idx = np.argsort(fx)
        fx, fy = fx[idx], fy[idx]
        try:
            (_, b_fit), pcov = curve_fit(_power, fx, fy, p0=(1.0, -0.5))
            sigma_b = np.sqrt(pcov[1, 1]) if pcov.shape == (2, 2) else np.nan
        except:
            b_fit = sigma_b = np.nan

        betas.append(float(b_fit))
        beta_errs.append(float(sigma_b))

    return (
        np.array(all_keys,   dtype=float),
        np.array(betas,      dtype=float),
        np.array(beta_errs,  dtype=float),
    )





def compute_betas_from_errors(
    errors_by_time: dict,
    scaling_param: str = 'spreading',
    include_families: list = None,
    exclude_x_scale: set = None,
    exclude_above_one: bool = False,
):
    """
    Returns a dict:
      family → (keys_sorted, betas_array, beta_errs_array)

    where for each family we fit, for each unique `key`, the model:
      err ≈ a * (sum(time_stamps))^b
    using only that family's data.
    """
    # 1) Gather the full set of keys and families
    all_keys = sorted({ key for errs in errors_by_time.values() for (_, _, key) in errs })
    all_fams = sorted({
        fam
        for errs in errors_by_time.values()
        for (_, fam, _) in errs
        if include_families is None or fam in include_families
    })

    # Prepare output container
    results = {}

    # Power‐law model
    def _power(x, a, b):
        return a * np.power(x, b)

    # For each family, build its own fit_data[key] → {"x":[], "y":[]}
    for fam in all_fams:
        # initialize per‐key buckets
        fit_data = { k: {"x": [], "y": []} for k in all_keys }

        # collect only this family's errors
        for time_tuple, err_list in errors_by_time.items():
            ssum = round(sum(time_tuple), 8)

            # skip excluded x
            if exclude_x_scale and ssum in exclude_x_scale:
                continue

            for err, fam0, key in err_list:
                if fam0 != fam:
                    continue
                # prefilter by scaling_param & exclude_above_one
                if exclude_above_one:
                    threshold = 1.25 if ssum == 0.1 else 1.0
                    if err >= threshold:
                        continue
                if scaling_param == 'times':
                    if ssum < 2:
                        pref = [err]
                    else:
                        pref = [err] if err < 0.1 else []
                else:
                    if ssum < 50:
                        pref = [err]
                    else:
                        pref = [err] if err < 0.1 else []
                if not pref:
                    continue

                # percentile filter on this single‐element list is trivial,
                # but for consistency let's treat uniformly:
                q0 = np.percentile(pref, 0)
                q50 = np.percentile(pref, 50)
                filt = [e for e in pref if q0 <= e <= q50]
                if not filt:
                    continue

                # store
                fit_data[key]["x"].extend([ssum] * len(filt))
                fit_data[key]["y"].extend(filt)

        # now for this family, do the fits over each key
        betas = []
        beta_errs = []
        for key in all_keys:
            fx = np.array(fit_data[key]["x"], dtype=float)
            fy = np.array(fit_data[key]["y"], dtype=float)

            if fx.size < 2 or fy.size < 2:
                betas.append(np.nan)
                beta_errs.append(np.nan)
                continue

            # sort to stabilize the fit
            idx = np.argsort(fx)
            fx, fy = fx[idx], fy[idx]

            try:
                (_, b_fit), pcov = curve_fit(_power, fx, fy, p0=(1.0, -0.5))
                sigma_b = np.sqrt(pcov[1, 1]) if pcov.shape == (2, 2) else np.nan
            except Exception:
                b_fit = np.nan
                sigma_b = np.nan

            betas.append(float(b_fit))
            beta_errs.append(float(sigma_b))

        results[fam] = (
            np.array(all_keys,   dtype=float),
            np.array(betas,      dtype=float),
            np.array(beta_errs,  dtype=float),
        )

    return results


import numpy as np
from scipy.optimize import curve_fit

def compute_betas_from_errors(
    errors_by_time: dict,
    scaling_param: str = 'spreading',
    include_families: list = None,
    exclude_x_scale: set = None,
    exclude_above_one: bool = False,
):
    """
    Returns a dict:
      family → (keys_sorted, betas_array, beta_errs_array)

    where for each family we fit, for each unique `key`, the model:
      err ≈ a * (sum(time_stamps))^b
    using only that family's data.
    """
    # 1) Identify all keys and families
    all_keys = sorted({ key for errs in errors_by_time.values() for (_, _, key) in errs })
    all_fams = sorted({
        fam for errs in errors_by_time.values()
               for (_, fam, _) in errs
        if include_families is None or fam in include_families
    })

    # power-law model
    def _power(x, a, b):
        return a * np.power(x, b)

    results = {}

    # 2) Loop per family
    for fam in all_fams:
        # collect this family's data
        fit_data = { k: {"x": [], "y": []} for k in all_keys }

        for time_tuple, recs in errors_by_time.items():
            ssum = round(sum(time_tuple), 8)
            if exclude_x_scale and ssum in exclude_x_scale:
                continue

            # gather only this family's errors
            errs_for_key = {}
            for err, fam0, key in recs:
                if fam0 != fam:
                    continue
                errs_for_key.setdefault(key, []).append(err)

            # apply prefilter + percentile for each key
            for key, errs_list in errs_for_key.items():
                
                if scaling_param == 'times':
                    if exclude_above_one:
                        if ssum==0.1:
                            pref = [e for e in errs_list if e < 1.25]
                        else:
                            pref = [e for e in errs_list if e < 1.0]
                        if len(pref) < 2:
                            continue
                    elif ssum < 2:
                        pref = errs_list.copy()
                    else:
                        pref = [e for e in errs_list if e < 0.1]
                else:
                    #pref = errs_list.copy()
                    if ssum < 50:
                        pref = errs_list.copy()
                    else:
                        pref = [e for e in errs_list if e < 0.1]

                if len(pref) < 2:
                    continue

                # (c) percentile filter
                q0 = np.percentile(pref, 0)
                q50 = np.percentile(pref, 50)
                filt = [e for e in pref if q0 <= e <= q50]
                #if len(filt) < 2:
                #    continue

                # store for fitting
                fit_data[key]["x"].extend([ssum] * len(filt))
                fit_data[key]["y"].extend(filt)

        # (3) Fit per key for this family
        betas, beta_errs = [], []
        for key in all_keys:
            fx = np.array(fit_data[key]["x"], float)
            fy = np.array(fit_data[key]["y"], float)
            if fx.size < 2 or fy.size < 2:
                betas.append(np.nan)
                beta_errs.append(np.nan)
                continue

            idx = np.argsort(fx)
            fx, fy = fx[idx], fy[idx]

            try:
                (_, b_fit), pcov = curve_fit(_power, fx, fy, p0=(1.0, -0.5))
                sigma_b = np.sqrt(pcov[1, 1]) if pcov.shape == (2, 2) else np.nan
            except:
                b_fit = sigma_b = np.nan

            betas.append(float(b_fit))
            beta_errs.append(float(sigma_b))

        results[fam] = (
            np.array(all_keys,   dtype=float),
            np.array(betas,      dtype=float),
            np.array(beta_errs,  dtype=float),
        )

    return results
