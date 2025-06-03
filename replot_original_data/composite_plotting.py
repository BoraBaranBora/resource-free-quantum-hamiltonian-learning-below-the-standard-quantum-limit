#!/usr/bin/env python3
"""
composite_plotting.py

Calls five pipelines defined in plotting_pipelines.py:

  – run_figure1_pipeline(first_parameter_sweep_path)
  – run_time_scaling_pipeline(second_parameter_sweep_path)
  – run_time_outer_pipeline(second_parameter_sweep_path)
  – run_perturb_scaling_pipeline(third_parameter_sweep_path)
  – run_perturb_outer_pipeline(third_parameter_sweep_path)
  – run_derivative_pipeline(time_res, pert_res)

Usage (run from within the same folder that contains these three subfolders and src/):
    python composite_plotting.py
"""

import sys
import os
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.abspath(os.path.join(THIS_DIR, "..", "src"))
sys.path.insert(0, SRC_DIR)

# Paths to the three restructured directories
first_parameter_sweep_path = os.path.join(THIS_DIR, "first_parameter_sweep_data")
second_parameter_sweep_path = os.path.join(THIS_DIR, "second_parameter_sweep_data")
third_parameter_sweep_path = os.path.join(THIS_DIR, "third_parameter_sweep_data")

from plotting_pipelines import (
    run_figure1_pipeline,
    run_time_scaling_pipeline,
    run_time_outer_pipeline,
    run_perturb_scaling_pipeline,
    run_perturb_outer_pipeline,
    run_derivative_pipeline
)

if __name__ == "__main__":
    # 1) Figure 1 pipeline
    run_figure1_pipeline(first_parameter_sweep_path)

    # 2) Figure 2 pipeline (time‐scaling, grouped by α)
    time_res = run_time_scaling_pipeline(second_parameter_sweep_path)

    # 2a) Figure 2 “outer” (pick one α, plot Error vs ∑(time))
    run_time_outer_pipeline(second_parameter_sweep_path)

    # 3) Figure 3 pipeline (perturb‐scaling, grouped by α)
    pert_res = run_perturb_scaling_pipeline(third_parameter_sweep_path)

    # 3a) Figure 3 “outer” (pick one α, plot Error vs ∑(perturb))
    run_perturb_outer_pipeline(third_parameter_sweep_path)

    # 4) If both pipelines (2 & 3) returned valid (alphas, betas), run derivative comparison
    run_derivative_pipeline(time_res, pert_res)

    print("All plotting complete.")
