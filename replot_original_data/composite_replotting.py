#!/usr/bin/env python3
"""
composite_replotting.py

This script orchestrates the plotting pipelines for all the figures shown in the manuscript, using
the original data within the folder in this directory, so that the following
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

import sys
import os
import numpy as np

# Ensure “src/” (where plotting_pipelines.py resides) is on the import path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.abspath(os.path.join(THIS_DIR, "..", "src"))
sys.path.insert(0, SRC_DIR)

# Paths to the three restructured data folders (relative to this script)
first_parameter_sweep_path  = os.path.join(THIS_DIR, "first_parameter_sweep_data")
second_parameter_sweep_path = os.path.join(THIS_DIR, "second_parameter_sweep_data")
third_parameter_sweep_path  = os.path.join(THIS_DIR, "third_parameter_sweep_data")

# Import the six pipeline functions from plotting_pipelines.py (located in src/)
from plotting_pipelines import (
    run_figure1_pipeline,
    run_time_scaling_pipeline,
    run_time_outer_pipeline,
    run_perturb_scaling_pipeline,
    run_perturb_outer_pipeline,
    run_derivative_pipeline
)

if __name__ == "__main__":
    # ────────────────────────────────────────────────────────────────────────────
    # 1) Figure 1 pipeline: time‐scaling grouped by perturbation
    #    • Expects “first_parameter_sweep_data/run_SWEEP1_perturb_<p>/” directories
    #    • Plots:
    #         - Error vs ∑(time‐stamps), colored by perturbation
    #         - β vs perturbation (error‐bar)
    # ────────────────────────────────────────────────────────────────────────────
    run_figure1_pipeline(first_parameter_sweep_path)

    # ────────────────────────────────────────────────────────────────────────────
    # 2) Figure 2 pipeline: time‐scaling grouped by α
    #    • Expects “second_parameter_sweep_data/run_SWEEP2_alpha_<α>/” directories
    #    • run_time_scaling_pipeline returns (alphas_time, betas_time)
    #      – Plots:
    #         - β vs α (error‐bar)
    #         - alternative β vs α (theory + linear fit)
    # ────────────────────────────────────────────────────────────────────────────
    time_res = run_time_scaling_pipeline(second_parameter_sweep_path)

    # ────────────────────────────────────────────────────────────────────────────
    # 2a) Time‐outer pipeline: pick one α and plot Error vs ∑(time‐stamps)
    #    • Uses the same aggregated data as run_time_scaling_pipeline
    #    • By default, chooses the largest α found
    #    • Plots error‐vs‐sum(time) for that α, with SQL & Heisenberg references
    # ────────────────────────────────────────────────────────────────────────────
    run_time_outer_pipeline(second_parameter_sweep_path)

    # ────────────────────────────────────────────────────────────────────────────
    # 3) Figure 3 pipeline: perturb‐scaling grouped by α
    #    • Expects “third_parameter_sweep_data/run_SWEEP3_alpha_<α>/” directories
    #    • run_perturb_scaling_pipeline returns (alphas_pert, betas_pert)
    #      – Plots:
    #         - β vs α (error‐bar)
    #         - alternative β vs α (theory + linear fit)
    # ────────────────────────────────────────────────────────────────────────────
    pert_res = run_perturb_scaling_pipeline(third_parameter_sweep_path)

    # ────────────────────────────────────────────────────────────────────────────
    # 3a) Perturb‐outer pipeline: pick one α and plot Error vs ∑(perturbation)
    #    • Uses the same aggregated data as run_perturb_scaling_pipeline
    #    • By default, chooses the largest α found
    #    • Plots error‐vs‐sum(perturb) for that α, with SQL & Heisenberg references
    # ────────────────────────────────────────────────────────────────────────────
    run_perturb_outer_pipeline(third_parameter_sweep_path)

    # ────────────────────────────────────────────────────────────────────────────
    # 4) Derivative pipeline: if both Figure 2 and Figure 3 returned valid results,
    #    find the common α values, compute empirical dβ/dα, and overlay theoretical dβ/dα.
    #    • Requires time_res and pert_res both non‐None
    #    • Plots dβ/dα vs α for both time‐scaling and perturb‐scaling
    # ────────────────────────────────────────────────────────────────────────────
    run_derivative_pipeline(time_res, pert_res)

    print("All plotting complete.")
