#!/usr/bin/env python3
"""
composite_plotting.py

This script runs three pipelines:

  • (Figure 1) “Time‐scaling” grouped by perturbation (scaling_param="times", group_by="perturb"),
    using folders under:
      …/composite_plotting.py
      └─ restructured_run_directory_figure1/
          ├─ run_20241218_092120/
          ├─ run_20241218_101730/
          └─ …  
    Produces:
        1) Error vs. ∑(time‐stamps), colored/fitted by perturbation
        2) β vs. perturbation (error‐bar)

  • (Figure 2) “Time‐scaling” grouped by α (scaling_param="times", group_by="alpha"),
    using folders under:
      …/composite_plotting.py
      ├─ restructured_run_directory_figure2/
      │   ├─ run_20250424_175607/
      │   ├─ run_20250426_233802/
      │   └─ …  
    Produces:
        1) β vs α (error‐bar)  
        2) alternative β vs α (theory + linear fit)  
        3) Error vs ∑(time‐stamps) for one chosen α  

  • (Figure 3) “Perturb‐scaling” grouped by α (scaling_param="perturb", group_by="alpha"),
    using folders under:
      …/composite_plotting.py
      └─ restructured_run_directory_figure3/
          ├─ run_20241222_183631/
          ├─ run_20250425_175216/
          └─ …  
    Produces:
        1) β vs α (error‐bar)  
        2) alternative β vs α (theory + linear fit)  
        3) Error vs ∑(perturbation) for one chosen α  

Finally, if both pipelines (2 & 3) succeed, we plot dβ/dα vs α comparing time‐scaling and perturb‐scaling.

Usage (run from within the same folder that holds these three subfolders and src/):
    python composite_plotting.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ─── Make sure “src/” (plotting_utils.py) is on the path ───
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.abspath(os.path.join(THIS_DIR, "..", "src"))
sys.path.insert(0, SRC_DIR)

from plotting_utils import (
    calculate_relative_errors,
    plot_relative_errors_by_perturbation,
    plot_relative_errors_for_outer,
    compute_betas_from_errors,
    plot_beta_trends,
    plot_betas_vs_alpha_alternative,
    plot_dbetadalpha
)
from sklearn.linear_model import LinearRegression


def run_figure1_pipeline():
    """
    Figure 1 (Time‐scaling grouped by perturbation):
      – scaling_param="times", group_by="perturb"
      – folders in restructured_run_directory_figure1/
    """
    print("\n=== Running Figure 1 Pipeline ===")
    base1 = os.path.join(THIS_DIR, "restructured_run_directory_figure1")
    run_dirs1 = sorted(os.listdir(base1))
    run_dirs1 = [os.path.join(base1, d) for d in run_dirs1 if d.startswith("run_")]

    # Merge all runs into combined_errors_by_time
    combined_errors_by_time = {}
    for run_dir in run_dirs1:
        if not os.path.isdir(run_dir):
            print(f"  [WARN] Skipping {run_dir}: not found")
            continue

        errs = calculate_relative_errors(
            run_dir,
            scaling_param='times',
            group_by='perturb'
        )
        for time_tuple, triplets in errs.items():
            combined_errors_by_time.setdefault(time_tuple, []).extend(triplets)

    if not combined_errors_by_time:
        print("  No data found for Figure 1. Skipping.\n")
        return

    # 1) Plot Error vs ∑(time‐stamps), colored/fitted by perturbation
    plot_relative_errors_by_perturbation(
        combined_errors_by_time,
        include_families=None,
        exclude_x_scale=None,
        label_prefix="P"
    )

    # 2) Compute β vs perturbation and plot (error‐bar)
    keys, betas, beta_errs = compute_betas_from_errors(
        combined_errors_by_time,
        scaling_param='times',
        include_families=None,
        exclude_x_scale=None
    )
    plot_beta_trends(
        keys,
        betas,
        beta_errs,
        label_prefix="P"
    )

    print("=== Finished Figure 1 Pipeline ===\n")



def run_time_scaling_pipeline():
    """
    Figure 2 (Time‐scaling grouped by α):
      – scaling_param="times", group_by="alpha"
      – folders in restructured_run_directory_figure2/
    """
    print("\n=== Running Time‐Scaling (Figure 2) Pipeline ===")
    base_time = os.path.join(THIS_DIR, "restructured_run_directory_figure2")
    run_dirs_time = sorted(os.listdir(base_time))
    run_dirs_time = [os.path.join(base_time, d) for d in run_dirs_time if d.startswith("run_")]

    combined_errors_by_time = {}
    for run_dir in run_dirs_time:
        if not os.path.isdir(run_dir):
            print(f"  [WARN] Skipping {run_dir}: not found")
            continue

        errs = calculate_relative_errors(
            run_dir,
            scaling_param='times',
            group_by='alpha'
        )
        for time_tuple, triplets in errs.items():
            combined_errors_by_time.setdefault(time_tuple, []).extend(triplets)

    if not combined_errors_by_time:
        print("  No time‐scaling data found; skipping Figure 2.\n")
        return None

    # Compute β vs α
    alphas_time, betas_time, beta_errs_time = compute_betas_from_errors(
        combined_errors_by_time,
        scaling_param='times',
        include_families=None,
        exclude_x_scale=None
    )

    # Plot alternative β vs α
    plot_betas_vs_alpha_alternative(
        alphas=alphas_time,
        betas=betas_time,
        beta_errs=beta_errs_time
    )

    # Find available α values, pick one (e.g. last)
    unique_alphas_time = sorted({
        triplet[2]
        for triplets in combined_errors_by_time.values()
        for triplet in triplets
    })
    print("  Available α (time‐scaling):", unique_alphas_time)
    alpha_value_time = unique_alphas_time[-1]

    # Plot “Error vs ∑(time‐stamps)” for chosen α
    plot_relative_errors_for_outer(
        errors_by_scaling=combined_errors_by_time,
        scaling_param='times',
        group_by='alpha',
        outer_value=alpha_value_time,
        include_families=None,
        exclude_x_scale=None,
        show_theory=True
    )

    print("=== Finished Figure 2 Pipeline ===\n")
    return alphas_time, betas_time


def run_perturb_scaling_pipeline():
    """
    Figure 3 (Perturb‐scaling grouped by α):
      – scaling_param="perturb", group_by="alpha"
      – folders in restructured_run_directory_figure3/
    """
    print("\n=== Running Perturb‐Scaling (Figure 3) Pipeline ===")
    base_pert = os.path.join(THIS_DIR, "restructured_run_directory_figure3")
    run_dirs_pert = sorted(os.listdir(base_pert))
    run_dirs_pert = [os.path.join(base_pert, d) for d in run_dirs_pert if d.startswith("run_")]

    combined_errors_by_perturb = {}
    for run_dir in run_dirs_pert:
        if not os.path.isdir(run_dir):
            print(f"  [WARN] Skipping {run_dir}: not found")
            continue

        errs = calculate_relative_errors(
            run_dir,
            scaling_param='perturb',
            group_by='alpha'
        )
        for perturb_tuple, triplets in errs.items():
            combined_errors_by_perturb.setdefault(perturb_tuple, []).extend(triplets)

    if not combined_errors_by_perturb:
        print("  No perturb‐scaling data found; skipping Figure 3.\n")
        return None

    # Compute β vs α
    alphas_pert, betas_pert, beta_errs_pert = compute_betas_from_errors(
        combined_errors_by_perturb,
        scaling_param='perturb',
        include_families=None,
        exclude_x_scale=None
    )

    # Plot alternative β vs α
    plot_betas_vs_alpha_alternative(
        alphas=alphas_pert,
        betas=betas_pert,
        beta_errs=beta_errs_pert
    )

    # Find available α values, pick one (e.g. last)
    unique_alphas_pert = sorted({
        triplet[2]
        for triplets in combined_errors_by_perturb.values()
        for triplet in triplets
    })
    print("  Available α (perturb‐scaling):", unique_alphas_pert)
    alpha_value_pert = unique_alphas_pert[-1]

    # Plot “Error vs ∑(perturbation)” for chosen α
    plot_relative_errors_for_outer(
        errors_by_scaling=combined_errors_by_perturb,
        scaling_param='perturb',
        group_by='alpha',
        outer_value=alpha_value_pert,
        include_families=None,
        exclude_x_scale=None,
        show_theory=True
    )

    print("=== Finished Figure 3 Pipeline ===\n")
    return alphas_pert, betas_pert


if __name__ == "__main__":
    # Run Figure 1 pipeline
    run_figure1_pipeline()

    # Run Figure 2 pipeline and capture results
    time_res = run_time_scaling_pipeline()

    # Run Figure 3 pipeline and capture results
    pert_res = run_perturb_scaling_pipeline()

    # If both time & perturb pipelines returned valid (α, b), plot derivative comparison
    if (time_res is not None) and (pert_res is not None):
        alphas_time, betas_time = time_res
        alphas_pert, betas_pert = pert_res

        # Determine α overlap
        common_alphas = np.array(
            sorted(set(alphas_time).intersection(set(alphas_pert))),
            dtype=float
        )
        if common_alphas.size < 2:
            print("Insufficient α overlap for derivative plot.")
        else:
            idx_time = [ np.where(alphas_time == a)[0][0] for a in common_alphas ]
            idx_pert = [ np.where(alphas_pert == a)[0][0] for a in common_alphas ]
            bt_common = betas_time[idx_time]
            bp_common = betas_pert[idx_pert]

            plot_dbetadalpha(
                alphas=common_alphas,
                betas_time=bt_common,
                betas_perturb=bp_common
            )

    print("All plotting complete.")
