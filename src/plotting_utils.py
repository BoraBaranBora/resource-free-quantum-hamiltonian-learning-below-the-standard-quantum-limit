# src/plotting_utils.py

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

# ──────────────────────────────────────────────────────────────────────────────
#  Helper / utility functions
# ──────────────────────────────────────────────────────────────────────────────

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


def calculate_relative_errors(
    base_run_dir: str,
    scaling_param: str = "times",      # "times" or "perturb"
    group_by: str = "perturb"          # "alpha", "times", or "perturb"
) -> dict:
    """
    Walk through each “combo” subdirectory under base_run_dir. For each Hamiltonian:
      - Load config.json and hamiltonians.json
      - Reconstruct predictor, generate true Hamiltonian, compute rel_error
      - Use scaling_param (either "times" or "perturb") as the top‐level key (tuple of values)
      - Use group_by (either "alpha", "times", or "perturb") as the third element in each tuple

    scaling_param: str, must be "times" or "perturb". The code will read meta_params[scaling_param],
                   convert it to a tuple, and use that as the dict key.
    group_by:      str, must be "alpha", "times", or "perturb", and must differ from scaling_param.
                   This value becomes the third element in each (rel_error, true_family, group_key).
    Returns:
      errors_by_scaling: { scaling_tuple → [ (relative_error, true_family, group_key) ] }
    """
    if scaling_param not in {"times", "perturb"}:
        raise ValueError("scaling_param must be 'times' or 'perturb'")
    if group_by not in {"alpha", "times", "perturb"}:
        raise ValueError("group_by must be 'alpha', 'times', or 'perturb'")
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
        raw_times = meta_params.get("times", None)
        if raw_times is None or not isinstance(raw_times, (list, tuple)) or len(raw_times) == 0:
            print(f"Skipping {combo_folder}: config.json has no valid 'times' list.")
            continue
        time_values = list(raw_times)       # e.g. [0.0, 0.5, 1.0]
        time_tuple  = tuple(time_values)

        # ─── Extract perturb (for "perturb" grouping or grouping by perturb) ───
        raw_perturb = meta_params.get("perturb", None)
        if raw_perturb is None:
            perturb_values = []
        elif isinstance(raw_perturb, (list, tuple)):
            if len(raw_perturb) == 0:
                print(f"Skipping {combo_folder}: config.json has empty 'perturb' list.")
                continue
            perturb_values = list(raw_perturb)
        else:
            perturb_values = [raw_perturb]
        perturb_tuple = tuple(perturb_values)

        # ─── Extract alpha ───
        alpha = meta_params.get("alpha", None)

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
        else:  # scaling_param == "perturb"
            scaling_list = perturb_values

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
                ignore_input  = True
            )
            predictor.load_state_dict(torch.load(model_path))
            predictor.eval()

            # Generate the “true” Hamiltonian matrix once
            original_ham = generate_hamiltonian(true_family, num_qubits, **ham_parameters)

            with torch.no_grad():
                out_flat = predictor(batch_size=1).squeeze(0)
                rec_matrix = reconstruct_density_matrix_from_lower(out_flat).to(torch.complex64) / 4.0
                rel_error = torch.mean((rec_matrix - original_ham).abs()).item()

            # ─── Decide the group_key based on group_by ───
            if group_by == "alpha":
                group_key = alpha
            elif group_by == "perturb":
                group_key = perturb_tuple
            else:  # group_by == "times"
                group_key = time_tuple

            # Append (rel_error, true_family, group_key) under scaling_tuple
            errors_by_scaling[scaling_tuple].append((rel_error, true_family, group_key))

    return errors_by_scaling


# ──────────────────────────────────────────────────────────────────────────────
#  Plotting routines
# ──────────────────────────────────────────────────────────────────────────────

def plot_relative_errors_by_perturbation(
    errors_by_time: dict,
    include_families: list = None,
    exclude_x_scale: set = None,
    label_prefix: str = "P"       # <<< new: pass "P" (default) or "α" for Figure 2
):
    """
    Plot “relative_error vs. sum_of_time_stamps”, coloring/fitting by the third‐tuple value.
    By default, we label curves with f"{label_prefix}={value}".  If you call with
      label_prefix="α", then legends will read “α=….”
    """
    def power_law(x, a, b):
        return a * x**b

    plt.figure(figsize=(10, 7))
    plt.xscale('log')
    plt.yscale('log')

    # Gather all unique third‐tuple values (either perturbations or alphas)
    unique_keys = sorted({key for _, errs in errors_by_time.items() for _, _, key in errs})
    cmap = plt.cm.viridis(np.linspace(0, 1, len(unique_keys)))
    color_map = {k: cmap[i] for i, k in enumerate(unique_keys)}

    fit_data = {}

    for time_stamps, errors in sorted(errors_by_time.items(), key=lambda x: round(sum(x[0]), 8)):
        ssum = round(sum(time_stamps), 8)
        bucket = {}
        for rel_err, fam, key in errors:
            if (include_families is not None) and (fam not in include_families):
                continue
            bucket.setdefault(key, []).append(rel_err)

        for key, rez in bucket.items():
            if exclude_x_scale and (ssum in exclude_x_scale):
                continue

            # (1) Scatter all raw points
            label_str = f"{label_prefix}={key:.2f}" if label_prefix == "α" else f"{label_prefix}={key}"
            already_in_legend = label_str in plt.gca().get_legend_handles_labels()[1]

            plt.scatter(
                [ssum] * len(rez),
                rez,
                color=color_map[key],
                edgecolor='black',
                alpha=0.5,
                s=80,
                label=label_str if not already_in_legend else None
            )

            # (2) Prefilter for fitting: keep only rel_err < 1.0
            pref = [e for e in rez if e < 1.0]
            if len(pref) < 2:
                continue

            q25 = scoreatpercentile(pref, 0)
            q75 = scoreatpercentile(pref, 50)
            filt = [e for e in pref if (q25 <= e <= q75)]

            fit_data.setdefault(key, {"x": [], "y": []})
            fit_data[key]["x"].extend([ssum] * len(filt))
            fit_data[key]["y"].extend(filt)

    # Fit & plot one curve per key
    for key, data in fit_data.items():
        fx = np.array(data["x"])
        fy = np.array(data["y"])
        if len(fx) < 2 or len(fy) < 2:
            continue

        popt, pcov = curve_fit(power_law, fx, fy, p0=(1, -0.5))
        a, b = popt
        a_err, b_err = np.sqrt(np.diag(pcov))

        def round_sig(val, err):
            sig_digit = -int(np.floor(np.log10(err)))
            return round(val, sig_digit), round(err, sig_digit)

        a_r, a_err_r = round_sig(a, a_err)
        b_r, b_err_r = round_sig(b, b_err)

        x_fit = np.linspace(min(fx), max(fx), 100)
        y_fit = power_law(x_fit, a, b)
        label_str = f"{label_prefix}={key:.2f}" if label_prefix == "α" else f"{label_prefix}={key}"
        plt.plot(
            x_fit,
            y_fit,
            '--',
            color=color_map[key],
            label=f"{label_str} fit: y=({a_r}±{a_err_r}) x^{b_r}±{b_err_r}"
        )

    plt.xlabel("Sum of Time Stamps (log)", fontsize=16)
    plt.ylabel("Error (log)", fontsize=16)
    title_prefix = "Error vs Sum of Time Stamps"
    if label_prefix == "α":
        plt.title(f"{title_prefix} (grouped by α)", fontsize=18)
    else:
        plt.title(f"{title_prefix} (grouped by perturbation)", fontsize=18)

    plt.grid(True, which='major', linestyle='-', linewidth=0.5, color='black', alpha=0.7)
    plt.grid(True, which='minor', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_beta_trends(
    keys: np.ndarray,
    betas: np.ndarray,
    beta_errs: np.ndarray = None,
    label_prefix: str = "P"
):
    """
    Plot fitted β values (learning‐rate exponents) against their corresponding keys
    (either perturbation values or α values) on a log‐scale x‐axis.

    Parameters:
    -----------
    keys : np.ndarray
        Array of keys (e.g. perturbation values or α values). Can be any shape; will be flattened.
    betas : np.ndarray
        Array of fitted β exponents corresponding to each key. Can be any shape; will be flattened.
    beta_errs : np.ndarray or None
        (Optional) Array of uncertainties σ_b for each fitted β. If provided,
        error bars will be drawn. Can be any shape; will be flattened.
    label_prefix : str
        If plotting vs. α, pass "α" so that axes and titles label appropriately;
        otherwise (default) it is interpreted as “perturbation.”

    Behavior:
    ---------
    - Converts inputs to 1D numpy arrays.
    - Scatter the (key, β) points in black with a thin edge.
    - Optionally draw vertical error bars at each point if beta_errs is given.
    - Connect the points in ascending‐key order with a gray line to show trend.
    - Draw a horizontal reference line at β = –0.75 (theoretical prediction)
      and a shaded band of width ±1/8 around it.
    - Use log‐scale for the x‐axis.
    """
    # (0) Ensure inputs are 1D numpy arrays
    keys = np.asarray(keys, dtype=float).ravel()
    betas = np.asarray(betas, dtype=float).ravel()
    if beta_errs is not None:
        beta_errs = np.asarray(beta_errs, dtype=float).ravel()

    # (1) Sort by keys so that the connecting line is monotonic
    idx = np.argsort(keys)
    keys_sorted = keys[idx]
    betas_sorted = betas[idx]
    if beta_errs is not None:
        beta_errs_sorted = beta_errs[idx]
    else:
        beta_errs_sorted = None

    plt.figure(figsize=(10, 7))

    # (2) Scatter or errorbar plot of (key, beta)
    if beta_errs_sorted is not None:
        plt.errorbar(
            keys_sorted,
            betas_sorted,
            yerr=beta_errs_sorted,
            fmt='o',
            color='black',
            ecolor='black',
            elinewidth=1,
            capsize=4,
            alpha=0.8,
            markersize=8,
            markeredgecolor='k',
            markeredgewidth=0.5,
            label="Fitted β ± σ"
        )
    else:
        plt.scatter(
            keys_sorted,
            betas_sorted,
            c='black',
            s=80,
            alpha=0.8,
            edgecolor='k',
            linewidth=0.5,
            label="Fitted β"
        )

    # (3) Trend line connecting the points
    plt.plot(
        keys_sorted,
        betas_sorted,
        '-', color='gray', alpha=0.6,
        label="Data trend"
    )

    # (4) Theoretical β = –0.75 and shaded band ±1/8
    theoretical_beta = -0.75
    half_width = 1 / 8
    plt.axhline(
        y=theoretical_beta,
        color='green',
        linestyle=':',
        linewidth=3,
        label=r"Theoretical β = –0.75"
    )
    plt.axhspan(
        theoretical_beta - half_width,
        theoretical_beta + half_width,
        color='green',
        alpha=0.2,
        label=r"O(m_t⁻¹) band around –0.75"
    )

    # (5) Axis scales and labels
    #plt.xscale('log')

    if label_prefix == "α":
        xlabel = "α (log)"
        plt.title("Learning Rate β vs. α", fontsize=16)
    else:
        xlabel = "Perturbation (log)"
        plt.title("Learning Rate β vs. Perturbation", fontsize=16)

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel("Learning Rate β", fontsize=14)

    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()
   

def compute_betas_from_errors(
    errors_by_time: dict,
    scaling_param='perturb',
    include_families: list = None,
    exclude_x_scale: set = None,
):
    """
    Given:
      errors_by_time: {
        time_stamps_tuple → [ (rel_error, true_family, key) ]
      }
    (where `key` is either a perturbation value or an α value),
    this function computes, for each unique `key`, the power‐law exponent b
    in the fit err ≈ a * (sum(time_stamps))^b using exactly the same
    prefiltering logic as plot_relative_errors_by_perturbation().

    Steps:
      1) For each (time_stamps, error_list) pair, compute ssum = round(sum(time_stamps), 8).
      2) Bucket all rel_error values by their `key`.
      3) For each key:
           • For each ssum, gather all rel_errs < 1.0.
           • Compute q25 = 0th percentile of that list, q75 = 50th percentile.
             Keep only values e with q25 ≤ e ≤ q75.
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

        # Build a temporary bucket of rel_errs per key at this ssum
        bucket = {}
        for rel_err, fam, key in err_list:
            if include_families is not None and fam not in include_families:
                continue
            bucket.setdefault(key, []).append(rel_err)

        for key, rel_errs_at_key in bucket.items():
            if exclude_x_scale and (ssum in exclude_x_scale):
                continue

            # (3a) Prefilter: keep only rel_err < 1.0
            if scaling_param == 'times':
                if ssum < 2:
                    pref = rel_errs_at_key.copy()
                else:
                    pref = [e for e in rel_errs_at_key if e < 0.1]
            else:
                #pref = rel_errs_at_key.copy()
                if ssum < 50:
                    pref = rel_errs_at_key.copy()
                else:
                    pref = [e for e in rel_errs_at_key if e < 0.1]

            if len(pref) < 2:
                continue

            # (3b) Compute 0th and 50th percentiles, then percentile‐filter
            q25 = np.percentile(pref, 0)   # same as min(pref)
            q75 = np.percentile(pref, 50)  # median(pref)
            filt = [e for e in pref if (q25 <= e <= q75)]
            if len(filt) < 2:
                continue

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


# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────


def plot_relative_errors_for_outer(
    errors_by_scaling: dict,
    scaling_param: str,
    group_by: str,
    outer_value,
    include_families: list = None,
    exclude_x_scale: set = None,
    show_theory: bool = True
):
    """
    Generalized “Error vs. sum(inner_tuple)” plot for a single outer_value, 
    with labels automatically derived from scaling_param and group_by.

    Parameters
    ----------
    errors_by_scaling : dict
        { inner_tuple → [ (rel_err, family, group_key), … ] }
        where `inner_tuple` was determined by scaling_param, and `group_key` by group_by.

    scaling_param : str
        Either "times" or "perturb". Indicates which meta‐parameter was used as the dict key.
        This affects the x‐axis label: 
          - "times" → "Sum of Time Stamps (log)"
          - "perturb" → "Sum of Perturbation (log)"

    group_by : str
        One of "alpha", "times", or "perturb". Indicates which meta‐parameter is used as the 
        “outer” grouping. Affects the plot title:
          - "alpha" → "α"
          - "times" → "Time Stamps"
          - "perturb" → "Perturbation"

    outer_value : scalar or tuple
        The specific group_key value to plot. All triplets whose third element == outer_value
        will be included.

    include_families : list or None
        If provided, only errors whose `family` is in this list are included.

    exclude_x_scale : set or None
        If provided, any ssum = round(sum(inner_tuple), 8) in this set will be scattered
        (all raw errs in gray) but excluded from fitting.

    show_theory : bool
        If True, overlay SQL (x^(-0.5)) and Heisenberg (x^(-1)) reference curves.

    Behavior
    --------
    1) Filters triplets in errors_by_scaling where group_key == outer_value.
    2) Groups those filtered errors by inner_tuple.
    3) For each inner_tuple (in ascending order of sum(inner_tuple)):
         a) Compute ssum = round(sum(inner_tuple), 8).
         b) If ssum ∈ exclude_x_scale: scatter raw errs (gray) & skip fitting.
         c) Else:
            • If ssum < 2: pref = errs.
            • Else: pref = [e for e in errs if e < 0.1].
            • If len(pref) < 2: scatter raw errs & skip fitting.
            • Else:
                – q0 = 0th percentile of pref, q50 = 50th percentile of pref.
                – filt = [e for e in pref if q0 ≤ e ≤ q50].
                – If len(filt) < 2: scatter raw errs & skip fitting.
                – Else:
                    • Scatter raw errs (gray), collect (ssum, e) for each e in filt into fit_x, fit_y.
    4) If len(fit_x) ≥ 2:
         • Fit fit_y ≈ a·(fit_x)^b via curve_fit.
         • Compute σ_a, σ_b from covariance.
         • Plot best‐fit line “y = (a±σ_a)·x^(b±σ_b)” in red dashed.
    5) If show_theory: overlay SQL (∝ x^(-0.5)) and Heisenberg (∝ x^(-1)) reference curves,
       normalized to pass through the first fitted point.

    Example usage
    -------------
    >>> # Suppose calculate_relative_errors(..., scaling_param="perturb", group_by="alpha")
    >>> errs_by_perturb = {
    ...     (0.1, 0.2): [ (0.05, "XY", 0.5), (0.08, "Heisenberg", 0.5), … ],
    ...     (0.2, 0.3): [ … ],
    ... }
    >>> plot_relative_errors_for_outer(
    ...     errors_by_scaling=errs_by_perturb,
    ...     scaling_param="perturb",
    ...     group_by="alpha",
    ...     outer_value=0.5,
    ...     include_families=None,
    ...     exclude_x_scale=None,
    ...     show_theory=True
    ... )
    """
    # Validate arguments
    if scaling_param not in {"times", "perturb"}:
        raise ValueError("scaling_param must be 'times' or 'perturb'")
    if group_by not in {"alpha", "times", "perturb"}:
        raise ValueError("group_by must be 'alpha', 'times', or 'perturb'")
    if group_by == scaling_param:
        raise ValueError("group_by must differ from scaling_param")

    # Determine axis labels from scaling_param and group_by
    inner_label = "Time Stamps" if scaling_param == "times" else "Perturbation"
    outer_label = {
        "alpha": "α",
        "times": "Time Stamps",
        "perturb": "Perturbation"
    }[group_by]

    # (1) Filter triplets where group_key == outer_value
    filtered = []
    for inner_tuple, triplets in errors_by_scaling.items():
        for (rel_err, family, group_key) in triplets:
            if group_key == outer_value:
                filtered.append((rel_err, family, inner_tuple))

    if not filtered:
        print(f"No data for {outer_label} = {outer_value}")
        return

    # (2) Prepare figure
    plt.figure(figsize=(8, 7))
    plt.xscale("log")
    plt.yscale("log")

    # (3) Group by inner_tuple
    inner_groups = {}
    for (rel_err, family, inner_tuple) in filtered:
        if (include_families is not None) and (family not in include_families):
            continue
        inner_groups.setdefault(inner_tuple, []).append(rel_err)

    fit_x = []
    fit_y = []

    # (4) Iterate over sorted inner_tuples
    for inner_tuple, errs in sorted(
        inner_groups.items(),
        key=lambda item: round(sum(item[0]), 8)
    ):
        ssum = round(sum(inner_tuple), 8)

        # (4a) Exclude from fitting if requested
        if (exclude_x_scale is not None) and (ssum in exclude_x_scale):
            plt.scatter(
                [ssum] * len(errs),
                errs,
                color="gray",
                edgecolor="black",
                alpha=0.5,
                s=80,
                label=None
            )
            continue

        # (4b) Custom prefilter
        #if ssum < 2:
        #    pref = errs.copy()
        #else:
        #    pref = [e for e in errs if e < 0.1]
            
        if scaling_param == 'times':
            if ssum < 2:
                pref = errs.copy()
            else:
                pref = [e for e in errs if e < 0.1]
        else:
            #pref = [e for e in errs if e < 2.0]
            #pref = errs.copy()
            if ssum < 50:
                pref = errs.copy()
            else:
                pref = [e for e in errs if e < 0.1]


        if len(pref) < 2:
            plt.scatter(
                [ssum] * len(errs),
                errs,
                color="gray",
                edgecolor="black",
                alpha=0.5,
                s=80,
                label=None
            )
            continue

        # (4c) Percentile‐filter on pref
        q0  = scoreatpercentile(pref, 0)
        q50 = scoreatpercentile(pref, 50)
        filt = [e for e in pref if (q0 <= e <= q50)]
        if len(filt) < 2:
            plt.scatter(
                [ssum] * len(errs),
                errs,
                color="gray",
                edgecolor="black",
                alpha=0.5,
                s=80,
                label=None
            )
            continue

        # (4d-1) Scatter all raw errs in gray
        plt.scatter(
            [ssum] * len(errs),
            errs,
            color="gray",
            edgecolor="black",
            alpha=0.5,
            s=80,
            label=None
        )

        # (4d-2) Append filtered errs for fitting
        fit_x.extend([ssum] * len(filt))
        fit_y.extend(filt)

    # (5) Fit a power‐law if we have ≥2 points
    def _power_law(x, a, b):
        return a * np.power(x, b)

    if len(fit_x) >= 2 and len(fit_y) >= 2:
        fx = np.array(fit_x, dtype=float)
        fy = np.array(fit_y, dtype=float)

        # Sort by fx
        idx_sort = np.argsort(fx)
        fx = fx[idx_sort]
        fy = fy[idx_sort]

        try:
            (a_fit, b_fit), pcov = curve_fit(_power_law, fx, fy, p0=(1.0, -0.5))
            sigma_a, sigma_b = np.sqrt(np.diag(pcov))
        except Exception:
            a_fit = np.nan
            b_fit = np.nan
            sigma_a = np.nan
            sigma_b = np.nan

        # (6) Compute smooth fit curve
        x_fit = np.logspace(np.log10(fx.min()), np.log10(fx.max()), 200)
        y_fit = _power_law(x_fit, a_fit, b_fit)

        # (7) Overlay SQL & Heisenberg if requested
        if show_theory:
            y_sql = y_fit[0] * (x_fit / x_fit[0]) ** (-0.5)
            plt.plot(
                x_fit, y_sql,
                color="black", linestyle="-", linewidth=2, alpha=0.7,
                label="SQL ∝ x⁻⁰․⁵"
            )
            y_heis = y_fit[0] * (x_fit / x_fit[0]) ** (-1.0)
            plt.plot(
                x_fit, y_heis,
                color="blue", linestyle="-", linewidth=2, alpha=0.7,
                label="Heisenberg ∝ x⁻¹"
            )

        # (8) Plot the best‐fit line and annotate uncertainties
        def _round_sig(val, err):
            if np.isnan(err) or err == 0:
                return round(val, 2), round(err, 2)
            sig = -int(np.floor(np.log10(err)))
            return round(val, sig), round(err, sig)

        a_r, a_err_r = _round_sig(a_fit, sigma_a)
        b_r, b_err_r = _round_sig(b_fit, sigma_b)

        plt.plot(
            x_fit,
            y_fit,
            'r--',
            linewidth=2,
            label=f"Fit: y = ({a_r} ± {a_err_r})·x^({b_r} ± {b_err_r})",
            zorder=3,
            clip_on=False
        )

    # (9) Finalize labels and title
    plt.xlabel(f"Sum of {inner_label} (log)", fontsize=16)
    plt.ylabel("Error (log)", fontsize=16)
    plt.title(f"Error vs ∑{inner_label} ( {outer_label} = {outer_value} )", fontsize=18)

    plt.grid(True, which="major", linestyle="-", linewidth=0.5, color="black", alpha=0.7)
    plt.grid(True, which="minor", linestyle="--", linewidth=0.5, color="gray", alpha=0.7)
    plt.legend(fontsize=12, loc="best")
    plt.tight_layout()
    plt.show()



import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def plot_betas_vs_alpha_alternative(alphas, betas, beta_errs):
    """
    Plot “β vs α” in the style of the example’s first panel:
      • A dashed line for the theoretical β_theory(α) = –((2α+1)/(2(α+1))).
      • Error‐bar scatter of (α, fitted b ± σ_b).
      • A linear regression fit of b vs. β_theory(α), plotted as a solid line.

    Inputs:
      alphas    : 1D array of α values (shape (N,))
      betas     : 1D array of fitted exponent b for each α (shape (N,))
      beta_errs : 1D array of 1σ uncertainties in b (shape (N,))
    """
    # 1) Filter out any NaNs so we only fit actual data points
    mask = ~np.isnan(betas)
    a_data = np.array(alphas[mask], dtype=float)
    b_data = np.array(betas[mask], dtype=float)
    e_data = np.array(beta_errs[mask], dtype=float)

    if a_data.size == 0:
        print("No valid data points to plot.")
        return

    # 2) Compute theoretical curve: β_theory(α) = –((2α+1)/(2(α+1)))
    alphas_fine = np.linspace(a_data.min(), a_data.max(), 200)
    beta_theory_fine = -((2 * alphas_fine + 1) / (2 * (alphas_fine + 1)))
    beta_theory_data = -((2 * a_data + 1) / (2 * (a_data + 1)))

    # 3) Fit a linear model: b_data ≈ m·β_theory_data + c
    lr = LinearRegression()
    lr.fit(beta_theory_data.reshape(-1, 1), b_data.reshape(-1, 1))
    m = lr.coef_[0, 0]
    c = lr.intercept_[0]
    fit_line_fine = m * beta_theory_fine + c

    # --- Begin plotting ---
    plt.figure(figsize=(8, 5))

    # 1) Dashed theoretical curve
    plt.plot(
        alphas_fine,
        beta_theory_fine,
        '--',
        color='C0',
        label=r'Theoretical $\beta(\alpha)$'
    )

    # 2) Error‐bar scatter of fitted b(α)
    plt.errorbar(
        a_data,
        b_data,
        yerr=e_data,
        fmt='s',
        capsize=4,
        color='C1',
        label=r'Fitted $b(\alpha)\pm\sigma_b$'
    )

    # 3) Solid regression fit: b_fit(α) = m·β_theory(α) + c
    plt.plot(
        alphas_fine,
        fit_line_fine,
        '-',
        linewidth=2,
        color='C3',
        label=f'Linear fit: slope={m:.2f}, intercept={c:.2f}'
    )

    # Plot formatting
    plt.xlabel(r'$\alpha$', fontsize=14)
    plt.ylabel(r'Error‐scaling exponent $b$', fontsize=14)
    plt.title(r'Error‐Scaling Exponent $b$ vs. $\alpha$', fontsize=16)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=12, loc='best')
    plt.tight_layout()
    plt.show()



def plot_dbetadalpha(
    alphas: np.ndarray,
    betas_time: np.ndarray,
    betas_perturb: np.ndarray,
    label_time: str = "Empirical $dβ_T/dα$",
    label_perturb: str = "Empirical $dβ_R/dα$",
    show_uncertainty_band: bool = True,
    band_half_width: float = 1/16
):
    """
    Plot dβ/dα for both “time‐scaling” and “perturb‐scaling” fits,
    using linear‐fit slopes m_t and m_p from b vs. β_theory(α), so that:
      dβ_empirical/dα = m · (dβ_theory/dα).

    Inputs:
      alphas         : 1D array of α values (shape (N,))
      betas_time     : 1D array of fitted exponent b_T for time‐scaling
      betas_perturb  : 1D array of fitted exponent b_R for perturb‐scaling
    """
    # (1) Filter out any NaNs and sort
    mask = ~np.isnan(betas_time) & ~np.isnan(betas_perturb)
    a_data = alphas[mask].astype(float)
    bt_data = betas_time[mask].astype(float)
    bp_data = betas_perturb[mask].astype(float)

    if a_data.size < 2:
        print("Not enough data to compute derivatives.")
        return

    idx = np.argsort(a_data)
    a_sorted = a_data[idx]
    bt_sorted = bt_data[idx]
    bp_sorted = bp_data[idx]

    # (2) Theoretical derivative of β_theory(α)
    alphas_fine = np.linspace(a_sorted.min(), a_sorted.max(), 200)
    d_theory = -1.0 / (2.0 * (alphas_fine + 1.0)**2)

    # (3) Fit linear model b_T ≈ m_t * β_theory(α) + c_t
    beta_theory_data = -((2 * a_sorted + 1) / (2 * (a_sorted + 1)))
    lr_t = LinearRegression().fit(
        beta_theory_data.reshape(-1, 1),
        bt_sorted.reshape(-1, 1)
    )
    m_t = float(lr_t.coef_[0, 0])

    # (4) Fit linear model b_R ≈ m_p * β_theory(α) + c_p
    lr_p = LinearRegression().fit(
        beta_theory_data.reshape(-1, 1),
        bp_sorted.reshape(-1, 1)
    )
    m_p = float(lr_p.coef_[0, 0])

    # (5) Empirical derivatives: dβ_empirical/dα = m · dβ_theory/dα
    d_fit_t = m_t * d_theory
    d_fit_p = m_p * d_theory

    # (6) Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    # (6a) Theoretical derivative (dashed, C0)
    ax.plot(
        alphas_fine,
        d_theory,
        '--',
        color='C0',
        label=r"Theoretical $d\beta_T/d\alpha$"
    )

    # (6b) Empirical time‐scaling derivative (solid, C3)
    ax.plot(
        alphas_fine,
        d_fit_t,
        '-',
        linewidth=2,
        color='C3',
        label=f"{label_time} (slope={m_t:.2f})"
    )

    # (6c) Empirical perturb‐scaling derivative (solid, C4)
    ax.plot(
        alphas_fine,
        d_fit_p,
        '-',
        linewidth=2,
        color='C4',
        label=f"{label_perturb} (slope={m_p:.2f})"
    )

    # (6d) ±band around time‐scaling derivative if requested
    if show_uncertainty_band:
        ax.fill_between(
            alphas_fine,
            d_fit_t - band_half_width,
            d_fit_t + band_half_width,
            color='C3',
            alpha=0.3,
            label=r'$\pm\,\frac{1}{16}$ around Empirical $d\beta_T/d\alpha$'
        )

    ax.set_xlabel(r'Scheduling exponent $\alpha$', fontsize=14)
    ax.set_ylabel(r'Sensitivity $dβ/dα$', fontsize=14)
    ax.set_title(r'Derivative of Scaling Exponents vs. $\alpha$', fontsize=16)
    ax.legend(loc='best', frameon=True)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()

