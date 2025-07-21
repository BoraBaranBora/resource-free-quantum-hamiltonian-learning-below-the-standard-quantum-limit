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
    compute_betas_from_errors,
)

from plotting_utils import (
    plot_errors_by_spreadings,
    plot_errors_for_outer,
    plot_beta_trends_per_family,
    plot_betas_vs_alpha_per_family,
    plot_each_family_separately
)



def run_sweep2_pipeline(base1: str, cache_dir: str):
    """
    Sweep 2 (time‐scaling by spreading).  Expects
      base1
    to point to second_parameter_sweep_data/.  It will look for
      {cache_dir}/sweep1_errors.pkl
    first, and only if missing will it recompute.
    """
    print("\n=== Running Sweep 2 Pipeline ===")
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
        include_families=["XYZ"],
        exclude_x_scale=[],
        label_prefix="P"
    )

    # 2) Compute β vs spreading and plot
    results = compute_betas_from_errors(
        combined_errors,
        scaling_param="times",
        include_families=["XYZ","XYZ2","XYZ3","XXYGL"],
        exclude_x_scale= None,
        exclude_above_one =True
    )
    plot_beta_trends_per_family(
        results,
        label_prefix="P"
    )    

    # 3) Print β and its 1σ uncertainty for each family & key
    print("\nSpreadings & β ± δβ by family:")
    for orig_fam, (keys, betas, errs) in results.items():
        fam = "XXZ" if orig_fam == "XXYGL" else orig_fam
        print(f"\nFamily: {fam}")
        for k, b, err in zip(keys, betas, errs):
            # ensure we have plain floats
            k_val   = float(k)
            b_val   = float(b)
            err_val = float(err)
            print(f"  {k_val:.3f} → {b_val:.3f} ± {err_val:.3f}")

    print("=== Finished Sweep 2 Pipeline ===\n")


def run_sweep1_pipeline(base_time: str, cache_dir: str):
    """
    Sweep 1 (time‐scaling by alpha).  Expects base_time
    to point to second_parameter_sweep_data/.  It will look for
      {cache_dir}/sweep1_errors.pkl first.
    """
    print("\n=== Running Sweep 2 Pipeline ===")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "sweep1_errors.pkl")

    # (1) Load or collect raw errors
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

    # (2) Compute per‐family β(α)
    results = compute_betas_from_errors(
        combined_errors,
        scaling_param="times",
        include_families=["XYZ","XYZ2","XYZ3","XXYGL"],
        exclude_x_scale=[],
        exclude_above_one=True
    )

    # (3) Plot using the new per‐family routine
    plot_betas_vs_alpha_per_family(
        results,
        exclude_alphas=[]
        #label_prefix="α"   # or "P" if you prefer that notation
    )
    
    # (4) Print β ± σ for each family, with your chosen label_prefix
    print("\nα & β(α) ± δβ by family:")
    for orig_fam, (keys, betas, errs) in results.items():
        fam = "XXZ" if orig_fam == "XXYGL" else orig_fam
        print(f"\nFamily: {fam}")
        for k, b, err in zip(keys, betas, errs):
            k_val   = float(k)
            b_val   = float(b)
            err_val = float(err)
            # prefix the key
            print(f"  α={k_val:.3f} → {b_val:.3f} ± {err_val:.3f}")


    print("\n=== Finished Sweep 1 (Figure 1) Pipeline ===\n")
    return results


def run_sweep1_outer(base_time: str, cache_dir: str):
    """
    Sweep 1 (Figure 1): pick one α (largest) and plot Error vs ∑(time).
    Expects cached errors under {cache_dir}/sweep2_errors.pkl.
    """
    print("\n=== Running Sweep 1 (Error vs ∑(time) for one α) ===")
    cache_path = os.path.join(cache_dir, "sweep1_errors.pkl")

    if os.path.exists(cache_path):
        print(f"  → Loading cached errors from {cache_path}")
        with open(cache_path, "rb") as f:
            combined_errors = pickle.load(f)
    else:
        print("  → No cached file; cannot run Sweep 1 Outer.")
        return

    unique_alphas = sorted({
        triplet[2]
        for triplets in combined_errors.values()
        for triplet in triplets
    })
    alpha_value = unique_alphas[-1]

    #plot_errors_for_outer(
    #    errors_by_scaling=combined_errors,
    #    scaling_param="times",
    #    group_by="alpha",
    #    outer_value=alpha_value,
    #    include_families=None,
    #    exclude_x_scale=[],
    #    show_theory=True
    #)
    
    plot_each_family_separately(
        errors_by_scaling=combined_errors,
        scaling_param='times',
        group_by='alpha',
        outer_value=alpha_value,
        families=["XYZ","XYZ2","XYZ3","XXYGL"],
        exclude_x_scale={},  
        show_theory=True
    )

    print("=== Finished Sweep 1 (Figure 2) ===\n")

