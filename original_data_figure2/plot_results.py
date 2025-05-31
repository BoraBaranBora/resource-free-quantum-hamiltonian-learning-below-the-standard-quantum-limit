#!/usr/bin/env python3
"""
plot_results_figure2.py

Draws Figure 2:
  1) Error vs. summed time‐stamps (grouped by α)
  2) β vs. α
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
    # 1) List all “run” folders you want to combine for Figure 2
    #    (each run_dir should follow the new “combo‐subfolder” structure).
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

    # 2) Gather all run_dirs into a single errors_by_time dict (group_by="alpha")
    combined_errors_by_time = {}
    for run_dir in run_dirs:
        if not os.path.isdir(run_dir):
            print(f"Skipping {run_dir}: not found")
            continue

        errors = calculate_relative_errors(run_dir, group_by="alpha")
        for tkey, err_list in errors.items():
            combined_errors_by_time.setdefault(tkey, []).extend(err_list)

    # 3a) Plot “Error vs. summed time‐stamps, colored/fitted by α”
    plot_relative_errors_by_perturbation(
        combined_errors_by_time,
        include_families=None,
        exclude_x_scale=None,
        label_prefix="α"
    )

    # 3b) Plot “β vs. α”
    plot_relative_errors_and_b_vs_perturbation(
        combined_errors_by_time,
        include_families=None,
        exclude_x_scale=None,
        label_prefix="α"
    )

    print("Done plotting Figure 2 (grouped by α).")
