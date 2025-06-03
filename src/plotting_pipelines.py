#!/usr/bin/env python3
"""
plotting_pipelines.py

Each “sweep” pipeline now takes an explicit `cache_dir` (instead of inferring
it from __file__).  That way, whatever folder your composite script lives in
can choose exactly which “cached_errors/” to use.
"""

import os
import pickle
import numpy as np

from extraction_and_evalution import (
    collect_recovery_errors_from_data,
    compute_betas_from_errors
)

from plotting_utils import (
    plot_errors_by_spreadings,
    plot_errors_for_outer,
    plot_beta_trends,
    plot_betas_vs_alpha_alternative,
    plot_dbetadalpha
)



def run_sweep1_pipeline(base1: str, cache_dir: str):
    """
    Sweep 1 (time‐scaling by spreading).  Expects
      base1
    to point to first_parameter_sweep_data/.  It will look for
      {cache_dir}/sweep1_errors.pkl
    first, and only if missing will it recompute.
    """
    print("\n=== Running Sweep 1 Pipeline ===")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "sweep1_errors.pkl")

    if os.path.exists(cache_path):
        print(f"  → Loading cached errors from {cache_path}")
        with open(cache_path, "rb") as f:
            combined_errors = pickle.load(f)
    else:
        print("  → No cached file; collecting errors from raw embeddings…")
        run_dirs1 = sorted(d for d in os.listdir(base1) if d.startswith("run_"))
        combined_errors = {}

        for d in run_dirs1:
            run_dir = os.path.join(base1, d)
            if not os.path.isdir(run_dir):
                print(f"  [WARN] Skipping {run_dir}: not found")
                continue

            errs = collect_recovery_errors_from_data(
                run_dir,
                scaling_param="times",
                group_by="spreading"
            )
            for time_tuple, triplets in errs.items():
                combined_errors.setdefault(time_tuple, []).extend(triplets)

        if not combined_errors:
            print("  No data found for Sweep 1. Skipping.\n")
            return

        with open(cache_path, "wb") as f:
            pickle.dump(combined_errors, f)
        print(f"  → Wrote out {len(combined_errors)} keys to {cache_path}")

    # 1) Error vs ∑(time), colored/fitted by spreading
    plot_errors_by_spreadings(
        combined_errors,
        include_families=None,
        exclude_x_scale=None,
        label_prefix="P"
    )

    # 2) Compute β vs spreading and plot
    keys, betas, beta_errs = compute_betas_from_errors(
        combined_errors,
        scaling_param="times",
        include_families=None,
        exclude_x_scale=None
    )
    plot_beta_trends(
        keys,
        betas,
        beta_errs,
        label_prefix="P"
    )

    print("=== Finished Sweep 1 Pipeline ===\n")


def run_sweep2_pipeline(base_time: str, cache_dir: str):
    """
    Sweep 2 (time‐scaling by alpha).  Expects
      base_time
    to point to second_parameter_sweep_data/.  It will look for
      {cache_dir}/sweep2_errors.pkl first.
    """
    print("\n=== Running Sweep 2 Pipeline ===")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "sweep2_errors.pkl")

    if os.path.exists(cache_path):
        print(f"  → Loading cached errors from {cache_path}")
        with open(cache_path, "rb") as f:
            combined_errors = pickle.load(f)
    else:
        print("  → No cached file; collecting errors from raw embeddings…")
        run_dirs_time = sorted(d for d in os.listdir(base_time) if d.startswith("run_"))
        combined_errors = {}

        for d in run_dirs_time:
            run_dir = os.path.join(base_time, d)
            if not os.path.isdir(run_dir):
                print(f"  [WARN] Skipping {run_dir}: not found")
                continue

            errs = collect_recovery_errors_from_data(
                run_dir,
                scaling_param="times",
                group_by="alpha"
            )
            for time_tuple, triplets in errs.items():
                combined_errors.setdefault(time_tuple, []).extend(triplets)

        if not combined_errors:
            print("  No Sweep 2 data found; skipping.\n")
            return None

        with open(cache_path, "wb") as f:
            pickle.dump(combined_errors, f)
        print(f"  → Wrote out {len(combined_errors)} keys to {cache_path}")

    # Fit β(α)
    alphas_time, betas_time, beta_errs_time = compute_betas_from_errors(
        combined_errors,
        scaling_param="times",
        include_families=None,
        exclude_x_scale=None
    )

    # Plot alternative β vs α
    plot_betas_vs_alpha_alternative(
        alphas=alphas_time,
        betas=betas_time,
        beta_errs=beta_errs_time
    )

    print("=== Finished Sweep 2 Pipeline ===\n")
    return alphas_time, betas_time


def run_sweep2_outer(base_time: str, cache_dir: str):
    """
    Sweep 2 Outer: pick one α (largest) and plot Error vs ∑(time).
    Expects cached errors under {cache_dir}/sweep2_errors.pkl.
    """
    print("\n=== Running Sweep 2 Outer (Error vs ∑(time) for one α) ===")
    cache_path = os.path.join(cache_dir, "sweep2_errors.pkl")

    if os.path.exists(cache_path):
        print(f"  → Loading cached errors from {cache_path}")
        with open(cache_path, "rb") as f:
            combined_errors = pickle.load(f)
    else:
        print("  → No cached file; cannot run Sweep 2 Outer.")
        return

    unique_alphas = sorted({
        triplet[2]
        for triplets in combined_errors.values()
        for triplet in triplets
    })
    print("  Available α (Sweep 2 Outer):", unique_alphas)
    alpha_value = unique_alphas[-1]

    plot_errors_for_outer(
        errors_by_scaling=combined_errors,
        scaling_param="times",
        group_by="alpha",
        outer_value=alpha_value,
        include_families=None,
        exclude_x_scale=None,
        show_theory=True
    )
    print("=== Finished Sweep 2 Outer ===\n")


def run_sweep3_pipeline(base_pert: str, cache_dir: str):
    """
    Sweep 3 (spreading‐scaling by alpha).  Expects
      base_pert
    to point to third_parameter_sweep_data/.  It will look for
      {cache_dir}/sweep3_errors.pkl first.
    """
    print("\n=== Running Sweep 3 Pipeline ===")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "sweep3_errors.pkl")

    if os.path.exists(cache_path):
        print(f"  → Loading cached errors from {cache_path}")
        with open(cache_path, "rb") as f:
            combined_errors = pickle.load(f)
    else:
        print("  → No cached file; collecting errors from raw embeddings…")
        run_dirs_pert = sorted(d for d in os.listdir(base_pert) if d.startswith("run_"))
        combined_errors = {}

        for d in run_dirs_pert:
            run_dir = os.path.join(base_pert, d)
            if not os.path.isdir(run_dir):
                print(f"  [WARN] Skipping {run_dir}: not found")
                continue

            errs = collect_recovery_errors_from_data(
                run_dir,
                scaling_param="spreading",
                group_by="alpha"
            )
            for spreading_tuple, triplets in errs.items():
                combined_errors.setdefault(spreading_tuple, []).extend(triplets)

        if not combined_errors:
            print("  No Sweep 3 data found; skipping.\n")
            return None

        with open(cache_path, "wb") as f:
            pickle.dump(combined_errors, f)
        print(f"  → Wrote out {len(combined_errors)} keys to {cache_path}")

    alphas_pert, betas_pert, beta_errs_pert = compute_betas_from_errors(
        combined_errors,
        scaling_param="spreading",
        include_families=None,
        exclude_x_scale=None
    )

    # Plot alternative β vs α
    plot_betas_vs_alpha_alternative(
        alphas=alphas_pert,
        betas=betas_pert,
        beta_errs=beta_errs_pert
    )

    print("=== Finished Sweep 3 Pipeline ===\n")
    return alphas_pert, betas_pert


def run_sweep3_outer(base_pert: str, cache_dir: str):
    """
    Sweep 3 Outer: pick one α (largest) and plot Error vs ∑(spreading).
    Expects cached errors under {cache_dir}/sweep3_errors.pkl.
    """
    print("\n=== Running Sweep 3 Outer (Error vs ∑(spreading) for one α) ===")
    cache_path = os.path.join(cache_dir, "sweep3_errors.pkl")

    if os.path.exists(cache_path):
        print(f"  → Loading cached errors from {cache_path}")
        with open(cache_path, "rb") as f:
            combined_errors = pickle.load(f)
    else:
        print("  → No cached file; cannot run Sweep 3 Outer.")
        return

    unique_alphas = sorted({
        triplet[2]
        for triplets in combined_errors.values()
        for triplet in triplets
    })
    print("  Available α (Sweep 3 Outer):", unique_alphas)
    alpha_value = unique_alphas[-1]

    plot_errors_for_outer(
        errors_by_scaling=combined_errors,
        scaling_param="spreading",
        group_by="alpha",
        outer_value=alpha_value,
        include_families=None,
        exclude_x_scale=None,
        show_theory=True
    )
    print("=== Finished Sweep 3 Outer ===\n")


def run_derivative_pipeline(time_res: tuple, pert_res: tuple):
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
        betas_spreading=bp_common
    )

spreading