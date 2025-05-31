#!/usr/bin/env python3
"""
plot_results.py

Draws only:
  1) Error vs. summed time-stamps, colored by perturbation
  2) β vs. perturbation (learning-rate exponent)
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
    plot_relative_errors_and_b_vs_perturbation
)

if __name__ == "__main__":
    # 1) List all the “run” directories you want to combine
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

    combined_errors_by_time = {}
    for run_dir in run_dirs:
        if not os.path.isdir(run_dir):
            print(f"Skipping {run_dir}: not found")
            continue

        errors = calculate_relative_errors(run_dir)
        for tkey, err_list in errors.items():
            combined_errors_by_time.setdefault(tkey, []).extend(err_list)

    # 2) Plot error vs. summed time-stamps (colored by perturb)
    plot_relative_errors_by_perturbation(
        combined_errors_by_time,
        include_families=None,
        exclude_x_scale=None
    )

    # 3) Plot β vs. perturbation
    plot_relative_errors_and_b_vs_perturbation(
        combined_errors_by_time,
        include_families=None,
        exclude_x_scale=None
    )

    print("Done plotting.")
