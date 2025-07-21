#!/usr/bin/env python3
"""
composite_replotting.py

This script orchestrates the plotting pipelines for all the figures shown in the publication, using
the original data cached errors within the folder in this directory.

  When complete, you will see on‐screen figures corresponding to Figures 1, 2,
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

)

if __name__ == "__main__":
    # point to your three “sweep_data” folders
    sweep1_path = os.path.join(THIS_DIR, "first_parameter_sweep_data")
    sweep2_path = os.path.join(THIS_DIR, "second_parameter_sweep_data")

    # point to your “cached_errors” folder
    cache_dir   = os.path.join(THIS_DIR, "cached_errors")

    # Extending the quadratic Regime
    # Figure 1 a) & b) (purely from Sweep 1 Data)
    run_sweep1_pipeline(sweep1_path, cache_dir)

    # From From Temporal Quadratic Sensitivity to Emergent Coherence in State Space
    # Sweeping alpha
    # Sweep 2 (returns (alphas_time, betas_time))
    # Figure 2a)
    time_res = run_sweep2_pipeline(sweep2_path, cache_dir)

    # From From Temporal Quadratic Sensitivity to Emergent Coherence in State Space
    # Illustrative Example Figure 2
    # 2b) (Single Run from Sweep 2)
    run_sweep2_outer(sweep2_path, cache_dir)

    print("All plotting complete.")
