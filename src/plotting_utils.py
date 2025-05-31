# src/plotting_utils.py

import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import scoreatpercentile

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
    group_by: str = "perturb"         # <<< added: can be "perturb" or "alpha"
) -> dict:
    """
    Walk through each “combo” subdirectory under base_run_dir (each combo = one time‐stamp configuration).
    For each Hamiltonian in that folder:
      - Load its merged meta‐parameters from config.json
      - Load hamiltonians.json to discover all “codified_name” entries
      - Load the saved embedding .pth file for each codified_name
      - Reconstruct the density matrix from the embedding’s lower‐triangle output
      - Generate the true Hamiltonian via generate_hamiltonian(...)
      - Compute a simple “mean absolute difference” as relative_error

    group_by: str, either "perturb" (default) or "alpha".  Whichever you choose becomes the 3rd element
              in each tuple (rel_error, true_family, key).

    Returns:
      errors_by_time: { time_stamps_tuple → [ (relative_error, true_family, key) ] }
    """
    errors_by_time = {}

    # Each subfolder under base_run_dir should be named like:
    #   alpha_<α>_perturb_<P>_measurements_<M>_shots_<S>_steps_<N>/
    for combo_folder in os.listdir(base_run_dir):
        combo_path = os.path.join(base_run_dir, combo_folder)
        if not os.path.isdir(combo_path):
            continue

        # Expect exactly these two JSON files inside each combo folder:
        #   config.json
        #   hamiltonians.json
        config_path       = os.path.join(combo_path, "config.json")
        hamiltonians_path = os.path.join(combo_path, "hamiltonians.json")

        if not (os.path.exists(config_path) and os.path.exists(hamiltonians_path)):
            print(f"Skipping {combo_path}: missing config.json or hamiltonians.json")
            continue

        # Load all training‐meta parameters from config.json
        with open(config_path, "r") as f:
            meta_params = json.load(f)

        # Extract fields from config.json:
        time_stamps_list = meta_params.get("times", [])
        if len(time_stamps_list) == 0:
            print(f"Skipping {combo_path}: config.json has no 'times' field.")
            continue

        time_stamps = tuple(time_stamps_list)   # use tuple as dict‐key
        alpha       = meta_params.get("alpha", None)    # <<< read both alpha
        perturb     = meta_params.get("perturb", None)  # <<< and perturb
        num_qubits  = meta_params.get("num_qubits", 5)

        # ────────────────────────────────────────────────────────────────────────
        # Figure out how to recreate the activation function from config.json
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
        # ────────────────────────────────────────────────────────────────────────

        # Build input_size/output_size from num_qubits:
        d = 2 ** num_qubits
        tri_len = d * (d + 1) // 2
        input_size  = tri_len
        output_size = tri_len

        # Initialize the list if needed
        if time_stamps not in errors_by_time:
            errors_by_time[time_stamps] = []

        # Read hamiltonians.json (list of { family, index, params, name })
        with open(hamiltonians_path, "r") as f:
            hamiltonians_list = json.load(f)

        # Loop over each Hamiltonian info dict
        for ham_info in hamiltonians_list:
            true_family    = ham_info["family"]
            ham_parameters = ham_info["params"]
            codified_name  = ham_info["name"]

            # Embedding weightfile is embedding_<codename>.pth
            model_fname = f"embedding_{codified_name}.pth"
            model_path  = os.path.join(combo_path, model_fname)
            if not os.path.isfile(model_path):
                # no embedding for this codename → skip
                continue

            # Reconstruct the same Predictor/embedding architecture used at training:
            predictor = Predictor(
                input_size    = input_size,
                output_size   = output_size,
                hidden_layers = meta_params.get("hidden_layers", [200, 200, 200]),
                activation_fn = activation_fn,
                ignore_input  = True
            )
            predictor.load_state_dict(torch.load(model_path))
            predictor.eval()

            # Generate the “true” Hamiltonian matrix:
            original_ham = generate_hamiltonian(true_family, num_qubits, **ham_parameters)

            with torch.no_grad():
                # predictor(batch_size=1) returns a 1×tri_len vector of lower‐triangle entries
                out_flat = predictor(batch_size=1).squeeze(0)  # shape: [tri_len]
                rec_matrix = reconstruct_density_matrix_from_lower(out_flat).to(torch.complex64) / 4.0

                # Compute mean absolute error as “relative_error”
                rel_error = torch.mean((rec_matrix - original_ham).abs()).item()

                # Decide which key to use (alpha vs perturb):
                if group_by == "alpha":
                    key = alpha
                else:
                    key = perturb

                # Append (rel_error, true_family, key) to this time_stamps bucket
                errors_by_time[time_stamps].append((rel_error, true_family, key))

    return errors_by_time


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


def plot_relative_errors_and_b_vs_perturbation(
    errors_by_time: dict,
    include_families: list = None,
    exclude_x_scale: set = None,
    label_prefix: str = "P"    # <<< new: pass "P" or "α"
):
    """
    1) First panel: for each key (perturbation or alpha), fit a power‐law (error ∼ a⋅x^b)
       against sum_of_time_stamps and plot that fit curve.
    2) Second panel: scatter “b vs key” (extracted from the fits),
       then fit a small log2 model for how b changes with key.

    label_prefix: “P” (then squares read “P=…”) or “α” (then legend “α=…”).
    """
    def power_law(x, a, b): return a * x**b
    def log2_model(x, a, b, c): return a * x**-(b/2) + c

    # First: error vs time‐sum fits, one curve per key
    plt.figure(figsize=(10, 7))
    plt.xscale('log')
    plt.yscale('log')

    unique_keys = sorted({k for _, errs in errors_by_time.items() for _, _, k in errs})
    cmap = plt.cm.viridis(np.linspace(0, 1, len(unique_keys)))
    color_map = {k: cmap[i] for i, k in enumerate(unique_keys)}

    fit_data = {}
    b_values = []

    for time_stamps, errors in sorted(errors_by_time.items(), key=lambda x: round(sum(x[0]), 8)):
        ssum = round(sum(time_stamps), 8)
        bucket = {}
        for rel_err, fam, k in errors:
            if (include_families is not None) and (fam not in include_families):
                continue
            bucket.setdefault(k, []).append(rel_err)

        for k, rez in bucket.items():
            pref = [e for e in rez if e < 1.0]
            if len(pref) < 2:
                continue

            q25 = scoreatpercentile(pref, 0)
            q75 = scoreatpercentile(pref, 50)
            filt = [e for e in pref if (q25 <= e <= q75)]

            fit_data.setdefault(k, {"x": [], "y": []})
            fit_data[k]["x"].extend([ssum] * len(filt))
            fit_data[k]["y"].extend(filt)

    for k, data in fit_data.items():
        fx = np.array(data["x"])
        fy = np.array(data["y"])
        if len(fx) < 2 or len(fy) < 2:
            continue

        popt, pcov = curve_fit(power_law, fx, fy, p0=(1, -0.5))
        a, b = popt
        b_values.append((k, b))

        x_fit = np.linspace(min(fx), max(fx), 100)
        y_fit = power_law(x_fit, a, b)
        label_str = f"{label_prefix}={k:.2f}" if label_prefix == "α" else f"{label_prefix}={k}"
        plt.plot(
            x_fit, y_fit, '--',
            color=color_map[k],
            label=f"{label_str} fit (b={b:.2f})"
        )

    plt.xlabel("Sum of Time Stamps (log)", fontsize=16)
    plt.ylabel("Error (log)", fontsize=16)
    if label_prefix == "α":
        plt.title("Effect of α on Error Scaling", fontsize=16)
    else:
        plt.title("Effect of Perturbations on Error Scaling", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Second: plot b‐values vs. key (perturb or alpha), plus a log2 fit
    if len(b_values) == 0:
        return

    keys, b_vals = zip(*sorted(b_values))
    keys    = np.array(keys)
    b_vals  = np.array(b_vals)

    popt2, pcov2 = curve_fit(log2_model, keys, b_vals, p0=(1, 0.1, 0.1))
    perr2       = np.sqrt(np.diag(pcov2))

    plt.figure(figsize=(10, 7))
    plt.scatter(
        keys, b_vals,
        c='black', s=80, alpha=0.8,
        edgecolor='k', linewidth=0.5,
        label="Fitted β"
    )
    idx = np.argsort(keys)
    plt.plot(
        keys[idx], b_vals[idx],
        '-', color='gray', alpha=0.6,
        label="Data trend"
    )

    # Theoretical β = –0.75
    plt.axhline(y=-0.75, color='green', linestyle=':', linewidth=3,
                label=r"Theoretical β = –0.75")
    half_width = 1/8
    plt.axhspan(-0.75-half_width, -0.75+half_width, color='green', alpha=0.2,
                label=r"O(m_t⁻¹) band around –0.75")

    # Overlay the log2‐fit line
    x_f = np.linspace(keys.min(), keys.max(), 100)
    plt.plot(
        x_f, log2_model(x_f, *popt2),
        'r--', lw=2,
        label=f"Log‐fit: y = a·x^(-b/2) + c"
    )

    plt.xscale('log')
    xlabel  = "α (log)" if label_prefix == "α" else "Perturbation (log)"
    ylabel  = "Learning Rate β"
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    if label_prefix == "α":
        plt.title("Learning Rate β vs. α", fontsize=16)
    else:
        plt.title("Learning Rate β vs. Perturbation", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()
