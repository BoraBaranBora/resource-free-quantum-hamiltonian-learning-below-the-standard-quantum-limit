#!/usr/bin/env python3
"""
plotting_pipelines.py

Contains:
  • run_figure1_pipeline(base1)
  • run_time_scaling_pipeline(base_time)
  • run_time_outer_pipeline(base_time)
  • run_perturb_scaling_pipeline(base_pert)
  • run_perturb_outer_pipeline(base_pert)
  • run_derivative_pipeline(time_res, pert_res)
"""

import os
import numpy as np

from extraction_and_evalution import (
    collect_recovery_errors_from_data,
    compute_betas_from_errors
)

from plotting_utils import (
    plot_errors_by_perturbation,
    plot_errors_for_outer,
    plot_beta_trends,
    plot_betas_vs_alpha_alternative,
    plot_dbetadalpha
)


def run_figure1_pipeline(base1: str):
    """
    Figure 1 (Time‐scaling grouped by perturbation):
      – scaling_param='times', group_by='perturb'
      – expects folders under base1/
    """
    print("\n=== Running Figure 1 Pipeline ===")
    run_dirs1 = sorted(os.listdir(base1))
    run_dirs1 = [os.path.join(base1, d) for d in run_dirs1 if d.startswith("run_")]

    combined_errors = {}
    for run_dir in run_dirs1:
        if not os.path.isdir(run_dir):
            print(f"  [WARN] Skipping {run_dir}: not found")
            continue

        errs = collect_recovery_errors_from_data(
            run_dir,
            scaling_param='times',
            group_by='perturb'
        )
        for time_tuple, triplets in errs.items():
            combined_errors.setdefault(time_tuple, []).extend(triplets)

    if not combined_errors:
        print("  No data found for Figure 1. Skipping.\n")
        return

    # 1) Plot Error vs ∑(time‐stamps), colored/fitted by perturbation
    plot_errors_by_perturbation(
        combined_errors,
        include_families=None,
        exclude_x_scale=None,
        label_prefix="P"
    )

    # 2) Compute β vs perturbation and plot (error‐bar)
    keys, betas, beta_errs = compute_betas_from_errors(
        combined_errors,
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


def run_time_scaling_pipeline(base_time: str):
    """
    Figure 2 (Time‐scaling grouped by α):
      – scaling_param='times', group_by='alpha'
      – expects folders under base_time/
    Returns:
      (alphas_time, betas_time) or None if no data.
    """
    print("\n=== Running Figure 2 (Time‐Scaling) Pipeline ===")
    dirs = sorted(os.listdir(base_time))
    run_dirs_time = [os.path.join(base_time, d) for d in dirs if d.startswith("run_")]

    combined_errors = {}
    for run_dir in run_dirs_time:
        if not os.path.isdir(run_dir):
            print(f"  [WARN] Skipping {run_dir}: not found")
            continue

        errs = collect_recovery_errors_from_data(
            run_dir,
            scaling_param='times',
            group_by='alpha'
        )
        for time_tuple, triplets in errs.items():
            combined_errors.setdefault(time_tuple, []).extend(triplets)

    if not combined_errors:
        print("  No time‐scaling data found; skipping.\n")
        return None

    alphas_time, betas_time, beta_errs_time = compute_betas_from_errors(
        combined_errors,
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

    print("=== Finished Figure 2 (Time‐Scaling) Pipeline ===\n")
    return alphas_time, betas_time


def run_time_outer_pipeline(base_time: str):
    """
    Separately: pick one α and plot “Error vs ∑(time‐stamps)” for that α.
      – Uses the same combined_errors_by_time logic as run_time_scaling_pipeline.
      – Expects folders under base_time/.
    """
    print("\n=== Running Time‐Outer Pipeline (Error vs ∑(time) for one α) ===")
    dirs = sorted(os.listdir(base_time))
    run_dirs_time = [os.path.join(base_time, d) for d in dirs if d.startswith("run_")]

    combined_errors = {}
    for run_dir in run_dirs_time:
        if not os.path.isdir(run_dir):
            print(f"  [WARN] Skipping {run_dir}: not found")
            continue

        errs = collect_recovery_errors_from_data(
            run_dir,
            scaling_param='times',
            group_by='alpha'
        )
        for time_tuple, triplets in errs.items():
            combined_errors.setdefault(time_tuple, []).extend(triplets)

    if not combined_errors:
        print("  No data found for Time‐Outer. Skipping.\n")
        return

    unique_alphas = sorted({
        triplet[2]
        for triplets in combined_errors.values()
        for triplet in triplets
    })
    print("  Available α (time‐outer):", unique_alphas)
    alpha_value = unique_alphas[-1]

    plot_errors_for_outer(
        errors_by_scaling=combined_errors,
        scaling_param='times',
        group_by='alpha',
        outer_value=alpha_value,
        include_families=None,
        exclude_x_scale=None,
        show_theory=True
    )
    print("=== Finished Time‐Outer Pipeline ===\n")


def run_perturb_scaling_pipeline(base_pert: str):
    """
    Figure 3 (Perturb‐scaling grouped by α):
      – scaling_param='perturb', group_by='alpha'
      – expects folders under base_pert/
    Returns:
      (alphas_pert, betas_pert) or None if no data.
    """
    print("\n=== Running Figure 3 (Perturb‐Scaling) Pipeline ===")
    dirs = sorted(os.listdir(base_pert))
    run_dirs_pert = [os.path.join(base_pert, d) for d in dirs if d.startswith("run_")]

    combined_errors = {}
    for run_dir in run_dirs_pert:
        if not os.path.isdir(run_dir):
            print(f"  [WARN] Skipping {run_dir}: not found")
            continue

        errs = collect_recovery_errors_from_data(
            run_dir,
            scaling_param='perturb',
            group_by='alpha'
        )
        for perturb_tuple, triplets in errs.items():
            combined_errors.setdefault(perturb_tuple, []).extend(triplets)

    if not combined_errors:
        print("  No perturb‐scaling data found; skipping.\n")
        return None

    alphas_pert, betas_pert, beta_errs_pert = compute_betas_from_errors(
        combined_errors,
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

    print("=== Finished Figure 3 (Perturb‐Scaling) Pipeline ===\n")
    return alphas_pert, betas_pert


def run_perturb_outer_pipeline(base_pert: str):
    """
    Separately: pick one α and plot “Error vs ∑(perturbation)” for that α.
      – Uses the same combined_errors_by_perturb logic as run_perturb_scaling_pipeline.
      – Expects folders under base_pert/.
    """
    print("\n=== Running Perturb‐Outer Pipeline (Error vs ∑(perturb) for one α) ===")
    dirs = sorted(os.listdir(base_pert))
    run_dirs_pert = [os.path.join(base_pert, d) for d in dirs if d.startswith("run_")]

    combined_errors = {}
    for run_dir in run_dirs_pert:
        if not os.path.isdir(run_dir):
            print(f"  [WARN] Skipping {run_dir}: not found")
            continue

        errs = collect_recovery_errors_from_data(
            run_dir,
            scaling_param='perturb',
            group_by='alpha'
        )
        for perturb_tuple, triplets in errs.items():
            combined_errors.setdefault(perturb_tuple, []).extend(triplets)

    if not combined_errors:
        print("  No data found for Perturb‐Outer. Skipping.\n")
        return

    unique_alphas = sorted({
        triplet[2]
        for triplets in combined_errors.values()
        for triplet in triplets
    })
    print("  Available α (perturb‐outer):", unique_alphas)
    alpha_value = unique_alphas[-1]

    plot_errors_for_outer(
        errors_by_scaling=combined_errors,
        scaling_param='perturb',
        group_by='alpha',
        outer_value=alpha_value,
        include_families=None,
        exclude_x_scale=None,
        show_theory=True
    )
    print("=== Finished Perturb‐Outer Pipeline ===\n")


def run_derivative_pipeline(
    time_res: tuple,
    pert_res: tuple
):
    """
    If both time_res and pert_res are not None, extract (alphas_time, betas_time)
    and (alphas_pert, betas_pert), find their common α values, and call plot_dbetadalpha.
    """
    if (time_res is None) or (pert_res is None):
        print("Skipping derivative comparison: missing data.")
        return

    alphas_time, betas_time = time_res
    alphas_pert, betas_pert = pert_res

    common_alphas = np.array(
        sorted(set(alphas_time).intersection(set(alphas_pert))),
        dtype=float
    )
    if common_alphas.size < 2:
        print("Insufficient α overlap for derivative plot.")
        return

    idx_time = [np.where(alphas_time == a)[0][0] for a in common_alphas]
    idx_pert = [np.where(alphas_pert == a)[0][0] for a in common_alphas]
    bt_common = betas_time[idx_time]
    bp_common = betas_pert[idx_pert]

    plot_dbetadalpha(
        alphas=common_alphas,
        betas_time=bt_common,
        betas_perturb=bp_common
    )


