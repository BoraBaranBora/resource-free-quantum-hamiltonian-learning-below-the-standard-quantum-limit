#!/usr/bin/env python3
"""
composite_plotting.py

This script orchestrates all the plotting pipelines for all the figures shown in the manuscript, using
re-computed data folders.  It assumes you have already re-generated the data for each
“figure” by running learn_hamiltonian.py (via the reproduction pipelines) so that the following
three top‐level directories exist at the same level as this script:

    first_parameter_sweep_data/
      ├── run_SWEEP1_perturb_1/
      │     ├── alpha_1.000_perturb_1_measurements_50_shots_1_steps_1/
      │     ├── alpha_1.000_perturb_1_measurements_50_shots_1_steps_2/
      │     └── … up to steps=8
      ├── run_SWEEP1_perturb_10/
      │     └── … (same structure)
      └── … (one run_SWEEP1_perturb_<p> per perturb value)

    second_parameter_sweep_data/
      ├── run_SWEEP2_alpha_0.3/
      │     ├── alpha_0.300_perturb_50_measurements_25_shots_1_steps_1/
      │     ├── alpha_0.300_perturb_50_measurements_25_shots_1_steps_2/
      │     └── … up to steps=8
      ├── run_SWEEP2_alpha_0.4/
      │     └── … (same structure)
      └── … (one run_SWEEP2_alpha_<α> per α value)

    third_parameter_sweep_data/
      ├── run_SWEEP3_alpha_0.3/
      │     ├── alpha_0.300_perturb_1_measurements_25_shots_1_steps_8/
      │     ├── alpha_0.300_perturb_10_measurements_25_shots_1_steps_8/
      │     └── … (all perturb values at steps=8)
      ├── run_SWEEP3_alpha_0.4/
      │     └── … (same structure)
      └── … (one run_SWEEP3_alpha_<α> per α value)

Each “combo” subfolder (alpha_<…>_perturb_<…>_measurements_<…>_shots_<…>_steps_<…>/) contains:
  • config.json
  • hamiltonians.json
  • embedding_<codename>.pth
  • embedding_<codename>_loss.json
  • … etc.

These directories supply all the raw embeddings and metadata needed to compute
recovery errors, fit power‐law exponents β, and plot error‐scaling curves.

────────────────────────────────────────────────────────────────────────────────
What this script does:
────────────────────────────────────────────────────────────────────────────────

1) **Figure 1 pipeline** (run_figure1_pipeline):
     - “Time‐scaling” grouped by perturbation (α = 1.0, sweep perturb ∈ {1,10,25,50,100,250,500}).
     - Expects to find subfolders in “first_parameter_sweep_data/” of the form run_SWEEP1_perturb_<p>/.
     - For each such run, it collects recovery errors as a function of summed time stamps,
       colored and fitted by perturbation.
     - Produces:
         a) Scatter & fit: Error vs ∑(time‐stamps), colored by perturbation.
         b) Error‐bar plot: β vs perturbation.

2) **Figure 2 pipeline** (run_time_scaling_pipeline + run_time_outer_pipeline):
     - “Time‐scaling” grouped by α (perturb = 50, sweep α ∈ {0.3,0.4,…,1.0}).
     - Expects to find subfolders in “second_parameter_sweep_data/” of the form run_SWEEP2_alpha_<α>/.
     - **run_time_scaling_pipeline**:
         • Aggregates all data for each α → fits β(α) using error vs ∑(time).
         • Plots:
              1) Error‐bar: β vs α.
              2) “Alternative” plot: theoretical β(α) plus linear‐regression fit of empirical β to theory.
     - **run_time_outer_pipeline**:
         • Picks one representative α (by default, the largest α found).
         • Plots “Error vs ∑(time‐stamps)” curve for that α, including SQL (∝ ∑time^−½) and Heisenberg (∝ ∑time^−1) reference lines.

3) **Figure 3 pipeline** (run_perturb_scaling_pipeline + run_perturb_outer_pipeline):
     - “Perturb‐scaling” grouped by α (steps=8, sweep perturb ∈ {1,10,25,50,100,250,500}, per α).
     - Expects to find subfolders in “third_parameter_sweep_data/” of the form run_SWEEP3_alpha_<α>/.
     - **run_perturb_scaling_pipeline**:
         • Aggregates all data for each α → fits β vs α using error vs ∑(perturb).
         • Plots:
              1) Error‐bar: β vs α.
              2) “Alternative” plot: theory vs empirical β(α).
     - **run_perturb_outer_pipeline**:
         • Picks one representative α (largest α by default).
         • Plots “Error vs ∑(perturbation)” for that α, including SQL and Heisenberg references.

4) **Derivative‐comparison pipeline** (run_derivative_pipeline):
     - Only runs if both Figure 2 and Figure 3 pipelines returned valid (alphas, betas).
     - Finds the intersection of α values common to both time‐scaling and perturb‐scaling.
     - Computes the theoretical derivative dβ_theory/dα and multiplies by fitted slopes (from linear
       regression of empirical β vs β_theory) to create “empirical derivatives” dβ/dα.
     - Plots dβ/dα vs α for both “time‐scaling” and “perturb‐scaling,” with an uncertainty band.

────────────────────────────────────────────────────────────────────────────────
Usage:
────────────────────────────────────────────────────────────────────────────────

  1. First, generate data by running `run_selected.py` (in the same directory).  That script
     invokes learn_hamiltonian.py in src/ to create all “combo” subfolders in:
       • first_parameter_sweep_data/
       • second_parameter_sweep_data/
       • third_parameter_sweep_data/

  2. Once those directories exist (populated by embeddings and config/hamiltonians JSONs),
     execute:

       $ python composite_plotting.py

     from within the same folder that holds “first_parameter_sweep_data/,”
     “second_parameter_sweep_data/,” and “third_parameter_sweep_data/” (i.e. this script’s folder).

  3. The script will call each pipeline in turn and display (or save) the corresponding plots:
       • Error‐vs‐time (Figure 1)
       • β vs perturbation (Figure 1)
       • Alternative β vs α (Figure 2)
       • Error vs ∑time for chosen α (Figure 2 outer)
       • Alternative β vs α (Figure 3)
       • Error vs ∑perturb for chosen α (Figure 3 outer)
       • dβ/dα vs α comparison (if applicable)

  4. When complete, you will see on‐screen figures corresponding to Figures 1, 2, 3,
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
