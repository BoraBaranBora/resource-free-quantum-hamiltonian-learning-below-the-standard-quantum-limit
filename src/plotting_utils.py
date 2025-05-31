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


def calculate_relative_errors(base_run_dir: str) -> dict:
    """
    Walk through each “combo” subdirectory under base_run_dir (each combo = one time‐stamp configuration).
    For each Hamiltonian in that folder:
      - Load its merged meta‐parameters from config.json
      - Load hamiltonians.json to discover all “codified_name” entries
      - Load the saved predictor .pth file for each codified_name
      - Reconstruct the density matrix from the predictor’s lower‐triangle output
      - Generate the true Hamiltonian via generate_hamiltonian(...)
      - Compute a simple “mean absolute difference” as relative_error
    Returns:
      errors_by_time: { time_stamps_tuple → [ (relative_error, true_family, perturbation_level) ] }
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
        perturb     = meta_params.get("perturb", None)
        num_qubits  = meta_params.get("num_qubits", 5)

        # ────────────────────────────────────────────────────────────────────────
        # Figure out how to recreate the activation function from config.json
        #
        # Possible stored forms include:
        #   "Tanh"
        #   "torch.nn.Tanh"
        #   "<class 'torch.nn.modules.activation.Tanh'>"
        #   "<class 'Tanh'>"
        #
        raw_act = meta_params.get("activation", "Tanh")

        # 1) If the string contains single quotes, grab whatever is between the first pair of "'"
        if "'" in raw_act:
            # e.g. raw_act = "<class 'torch.nn.modules.activation.Tanh'>"
            #      raw_act.split("'")[1]  → "torch.nn.modules.activation.Tanh"
            inner = raw_act.split("'")[1]
            # Now inner might be "torch.nn.modules.activation.Tanh" or just "Tanh"
            activation_name = inner.split(".")[-1]

        # 2) Otherwise, if the string has a dot, assume something like "torch.nn.Tanh"
        elif "." in raw_act:
            activation_name = raw_act.split(".")[-1]

        # 3) Otherwise, assume it is already only the bare class name, e.g. "Tanh"
        else:
            activation_name = raw_act

        # Finally, look up in torch.nn
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

            # Reconstruct the same Predictor architecture used at training:
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
                # Predictor(batch_size=1) returns a 1×tri_len vector of lower‐triangle entries
                out_flat = predictor(batch_size=1).squeeze(0)  # shape: [tri_len]
                rec_matrix = reconstruct_density_matrix_from_lower(out_flat).to(torch.complex64) / 4.0

                # Compute mean absolute error as “relative_error”
                rel_error = torch.mean((rec_matrix - original_ham).abs()).item()

                # Append (rel_error, true_family, perturb) to this time_stamps bucket
                errors_by_time[time_stamps].append((rel_error, true_family, perturb))

    return errors_by_time


# ──────────────────────────────────────────────────────────────────────────────
#  Plotting routines
# ──────────────────────────────────────────────────────────────────────────────

def plot_relative_errors_by_perturbation(
    errors_by_time: dict,
    include_families: list = None,
    exclude_x_scale: set = None
):
    """
    Plot “relative_error vs. sum_of_time_stamps”, coloring/fitting by perturbation level.
    Each perturbation gets a unique color. Fits a power-law per perturbation
    (using the middle 50% of prefiltered data). All raw points are shown.
    """
    def power_law(x, a, b):
        return a * x**b

    plt.figure(figsize=(10, 7))
    plt.xscale('log')
    plt.yscale('log')

    # Gather all unique perturbation levels
    unique_perts = sorted({p for _, errs in errors_by_time.items() for _, _, p in errs})
    cmap = plt.cm.viridis(np.linspace(0, 1, len(unique_perts)))
    pert_colors = {p: cmap[i] for i, p in enumerate(unique_perts)}

    pert_fit_data = {}

    for time_stamps, errors in sorted(errors_by_time.items(), key=lambda x: round(sum(x[0]), 8)):
        ssum = round(sum(time_stamps), 8)
        pertdict = {}
        for rel_err, fam, p in errors:
            if (include_families is not None) and (fam not in include_families):
                continue
            pertdict.setdefault(p, []).append(rel_err)

        for p, rez in pertdict.items():
            if exclude_x_scale and (ssum in exclude_x_scale):
                continue

            # (1) Scatter all raw points
            plt.scatter(
                [ssum] * len(rez),
                rez,
                color=pert_colors[p],
                edgecolor='black',
                alpha=0.5,
                s=80,
                label=f"P={p}" if f"P={p}" not in plt.gca().get_legend_handles_labels()[1] else None
            )

            # (2) Prefilter for fitting: keep only rel_err < 1.0
            pref = [e for e in rez if e < 1.0]
            if len(pref) < 2:
                continue

            q25 = scoreatpercentile(pref, 0)
            q75 = scoreatpercentile(pref, 50)
            filt = [e for e in pref if (q25 <= e <= q75)]

            if p not in pert_fit_data:
                pert_fit_data[p] = {"x": [], "y": []}
            pert_fit_data[p]["x"].extend([ssum] * len(filt))
            pert_fit_data[p]["y"].extend(filt)

    # Now fit & plot one curve per perturbation level
    for p, data in pert_fit_data.items():
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
        plt.plot(
            x_fit, y_fit, '--',
            color=pert_colors[p],
            label=f"P={p} fit: y=({a_r}±{a_err_r})x^({b_r}±{b_err_r})"
        )

    plt.xlabel("Sum of Time Stamps (log)", fontsize=16)
    plt.ylabel("Error (log)", fontsize=16)
    plt.title("Error vs Sum of Time Stamps (by perturbation)", fontsize=18)
    plt.grid(True, which='major', linestyle='-', linewidth=0.5, color='black', alpha=0.7)
    plt.grid(True, which='minor', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_relative_errors_and_b_vs_perturbation(
    errors_by_time: dict,
    include_families: list = None,
    exclude_x_scale: set = None
):
    """
    1) First panel: for each perturbation level, fit a power-law (error ∼ a⋅x^b)
       against sum_of_time_stamps and plot that fit curve.
    2) Second panel: scatter “b vs perturbation” (extracted from the fits),
       then fit a small log2 model for how b changes with perturbation.
    """
    def power_law(x, a, b): return a * x**b
    def log2_model(x, a, b, c): return a * x**-(b/2) + c

    # First: error vs time‐sum fits, one curve per perturbation
    plt.figure(figsize=(10, 7))
    plt.xscale('log')
    plt.yscale('log')

    unique_perts = sorted({p for _, errs in errors_by_time.items() for _, _, p in errs})
    cmap = plt.cm.viridis(np.linspace(0, 1, len(unique_perts)))
    pert_colors = {p: cmap[i] for i, p in enumerate(unique_perts)}

    pert_fit_data = {}
    b_values = []

    for time_stamps, errors in sorted(errors_by_time.items(), key=lambda x: round(sum(x[0]), 8)):
        ssum = round(sum(time_stamps), 8)
        pertdict = {}
        for rel_err, fam, p in errors:
            if (include_families is not None) and (fam not in include_families):
                continue
            pertdict.setdefault(p, []).append(rel_err)

        for p, rez in pertdict.items():
            pref = [e for e in rez if e < 1.0]
            if len(pref) < 2:
                continue
            q25 = scoreatpercentile(pref, 0)
            q75 = scoreatpercentile(pref, 50)
            filt = [e for e in pref if (q25 <= e <= q75)]

            pert_fit_data.setdefault(p, {"x": [], "y": []})
            pert_fit_data[p]["x"].extend([ssum] * len(filt))
            pert_fit_data[p]["y"].extend(filt)

    for p, data in pert_fit_data.items():
        fx = np.array(data["x"])
        fy = np.array(data["y"])
        if len(fx) < 2 or len(fy) < 2:
            continue

        popt, pcov = curve_fit(power_law, fx, fy, p0=(1, -0.5))
        a, b = popt
        b_values.append((p, b))

        x_fit = np.linspace(min(fx), max(fx), 100)
        y_fit = power_law(x_fit, a, b)
        plt.plot(
            x_fit, y_fit, '--',
            color=pert_colors[p],
            label=f"P={p} fit (b={b:.2f})"
        )

    plt.xlabel("Sum of Time Stamps (log)", fontsize=16)
    plt.ylabel("Error (log)", fontsize=16)
    plt.title("Effect of State Spreadings on Error", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Second: plot b‐values vs. perturbation, plus a log2 fit
    if len(b_values) == 0:
        return

    pert_levels, b_vals = zip(*sorted(b_values))
    pert_levels = np.array(pert_levels)
    b_vals      = np.array(b_vals)

    popt2, pcov2 = curve_fit(log2_model, pert_levels, b_vals, p0=(1, 0.1, 0.1))
    perr2 = np.sqrt(np.diag(pcov2))

    plt.figure(figsize=(10, 7))
    plt.scatter(
        pert_levels, b_vals,
        c='black', s=80, alpha=0.8,
        edgecolor='k', linewidth=0.5,
        label="Fitted β"
    )
    idx = np.argsort(pert_levels)
    plt.plot(
        pert_levels[idx], b_vals[idx],
        '-', color='gray', alpha=0.6, label="Data trend"
    )

    # Theoretical β = –0.75
    plt.axhline(y=-0.75, color='green', linestyle=':', linewidth=3,
                label=r"Theoretical β = –0.75")
    half_width = 1/8
    plt.axhspan(-0.75-half_width, -0.75+half_width, color='green', alpha=0.2,
                label=r"O(m_t⁻¹) band around –0.75")

    # Overlay the log2‐fit line
    x_f = np.linspace(pert_levels.min(), pert_levels.max(), 100)
    plt.plot(
        x_f, log2_model(x_f, *popt2),
        'r--', lw=2,
        label=f"Log‐fit: y = a⋅x^{-(b/2)} + c"
    )

    plt.xscale('log')
    plt.xlabel("Number of State Spreadings (log)", fontsize=14)
    plt.ylabel("Learning Rate β", fontsize=14)
    plt.title("Learning Rate vs. Number of State Spreadings", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()
