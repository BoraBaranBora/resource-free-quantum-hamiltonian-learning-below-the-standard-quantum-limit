#!/usr/bin/env python3
"""
composite_replotting.py

This script orchestrates the plotting pipelines for all the figures shown in the publication, using
the original data cached errors within the folder in this directory.

  When complete, you will see on‐screen figures corresponding to Figures 1, 2, 3,
  and the derivative‐comparison.  Examine the legends and titles to verify which sweep each
  plot represents.

────────────────────────────────────────────────────────────────────────────────
"""

#!/usr/bin/env python3
import os, sys
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.abspath(os.path.join(THIS_DIR, "..", "src"))
sys.path.insert(0, SRC_DIR)

from plotting_pipelines import (
    run_sweep1_pipeline,
    run_sweep2_pipeline,
    run_sweep2_outer,
    run_sweep3_pipeline,
    run_sweep3_outer,
    run_derivative_pipeline
)

if __name__ == "__main__":
    # point to your three “sweep_data” folders
    sweep1_path = os.path.join(THIS_DIR, "first_parameter_sweep_data")
    sweep2_path = os.path.join(THIS_DIR, "second_parameter_sweep_data")
    sweep3_path = os.path.join(THIS_DIR, "third_parameter_sweep_data")

    # point to your “cached_errors” folder
    cache_dir   = os.path.join(THIS_DIR, "cached_errors")

    # 1) Sweep 1
    run_sweep1_pipeline(sweep1_path, cache_dir)

    # 2) Sweep 2 (returns (alphas_time, betas_time))
    time_res = run_sweep2_pipeline(sweep2_path, cache_dir)

    # 2a) Sweep 2 Outer
    run_sweep2_outer(sweep2_path, cache_dir)

    # 3) Sweep 3 (returns (alphas_pert, betas_pert))
    pert_res = run_sweep3_pipeline(sweep3_path, cache_dir)

    # 3a) Sweep 3 Outer
    run_sweep3_outer(sweep3_path, cache_dir)

    # 4) Derivative
    run_derivative_pipeline(time_res, pert_res)

    print("All plotting complete.")
