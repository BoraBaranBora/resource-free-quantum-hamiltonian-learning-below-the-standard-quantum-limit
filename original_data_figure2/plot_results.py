#!/usr/bin/env python3
"""
plot_results_figure2.py

1) Error vs. summed time‐stamps (grouped by α)
2) β vs. α
3) b ± σ_b vs. α (error‐bar plot)
"""

import os
import sys
import numpy as np

# 1) Ensure we can import from src/plotting_utils.py
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.abspath(os.path.join(THIS_DIR, "..", "src"))
sys.path.insert(0, SRC_DIR)

from plotting_utils import (
    calculate_relative_errors,
    plot_relative_errors_for_outer,
    compute_betas_from_errors,
    plot_beta_trends,
    plot_betas_vs_alpha_alternative
)

if __name__ == "__main__":
    # (A) Gather run directories
    base = os.path.join(THIS_DIR, "restructured_run_directory")
    run_dirs = [
        os.path.join(base, "run_20250424_175607"),
        os.path.join(base, "run_20250426_233802"),
        os.path.join(base, "run_20250427_012136"),
        os.path.join(base, "run_20250427_025805"),
        os.path.join(base, "run_20250427_043005"),
        os.path.join(base, "run_20250428_153348"),
        os.path.join(base, "run_20250428_165350"),
        os.path.join(base, "run_20250428_181105"),
        os.path.join(base, "run_20250428_192626"),
    ]

    # (B) Compute combined_errors_by_time grouped by alpha
    combined_errors_by_time = {}
    for run_dir in run_dirs:
        if not os.path.isdir(run_dir):
            print(f"Skipping {run_dir}: not found")
            continue

        errors = calculate_relative_errors(
            run_dir,
            scaling_param='times',
            group_by='alpha'
        )
        for tkey, lst in errors.items():
            combined_errors_by_time.setdefault(tkey, []).extend(lst)

    if not combined_errors_by_time:
        print("No data found. Exiting.")
        sys.exit(0)

    # (D) Compute β vs α
    alphas, betas, beta_errs = compute_betas_from_errors(
        combined_errors_by_time,
        scaling_param='times',
        include_families=None,
        exclude_x_scale=None
    )
    
    plot_betas_vs_alpha_alternative(alphas, betas, beta_errs)

    
    # (B.1) Check which alpha values actually exist
    unique_alphas = sorted({
        triplet[2]
        for triplets in combined_errors_by_time.values()
        for triplet in triplets
    })
    print("Available α values:", unique_alphas)
    # Now pick one:
    alpha_value = unique_alphas[8]   # e.g. the smallest α, or pick whichever index you want

    # (C) Plot “Error vs ∑(time‐stamps)” for that alpha_value
    plot_relative_errors_for_outer(
        errors_by_scaling=combined_errors_by_time,
        scaling_param='times',
        group_by='alpha',
        outer_value=alpha_value,
        include_families=None,
        exclude_x_scale=None,
        show_theory=True
    )
    
    print("Done plotting Figure 2 (including error‐bar plot of $b\\pm\sigma_b$).")
