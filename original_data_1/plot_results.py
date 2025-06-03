#!/usr/bin/env python3
"""
plot_results_figure1.py

Draws Figure 1:
  1) Error vs. summed time‐stamps (grouped by perturbation)
  2) β vs. perturbation
"""

import os
import sys

# Ensure we can import plotting_utils from src/
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.abspath(os.path.join(THIS_DIR, "..", "src"))
sys.path.insert(0, SRC_DIR)

from plotting_utils import (
    calculate_relative_errors,
    plot_relative_errors_by_perturbation,
    plot_relative_errors_and_b_vs_perturbation,
    compute_betas_from_errors,
    plot_beta_trends,
)

if __name__ == "__main__":
    # 1) List all “run” folders you want to combine for Figure 1
    #    (each run_dir should follow the new “combo‐subfolder” structure).
    base = os.path.join(THIS_DIR, "restructured_run_directory")
    run_dirs = [
        os.path.join(base, "run_20241218_092120"),
        os.path.join(base, "run_20241218_101730"),
        os.path.join(base, "run_20241218_114241"),
        os.path.join(base, "run_20241218_133933"),
        os.path.join(base, "run_20241218_162936"),
        os.path.join(base, "run_20241218_212100"),
        os.path.join(base, "run_20241219_074707"),
        os.path.join(base, "run_20241219_234721"),
    ]

    # 2) Gather all run_dirs into a single errors_by_time dict (group_by="perturb")
    combined_errors_by_time = {}
    for run_dir in run_dirs:
        if not os.path.isdir(run_dir):
            print(f"Skipping {run_dir}: not found")
            continue

        errors = calculate_relative_errors(run_dir, scaling_param='times', group_by='perturb')
        for tkey, err_list in errors.items():
            combined_errors_by_time.setdefault(tkey, []).extend(err_list)

    # 3a) Plot “Error vs. summed time‐stamps, colored/fitted by perturbation”
    plot_relative_errors_by_perturbation(
        combined_errors_by_time,
        include_families=None,
        exclude_x_scale=None,
        label_prefix="P"
    )
    
    # 2) Compute betas and their uncertainties:
    keys, betas, beta_errs = compute_betas_from_errors(
        combined_errors_by_time,
        include_families=None,
        exclude_x_scale=None
    )
    
    plot_beta_trends(
        keys,
        betas,
        beta_errs,          # pass None if you don’t want error bars
        label_prefix="P"    # or "α" if your keys are alpha‐values
    )

    print("Done plotting Figure 1 (grouped by perturbation).")
