#!/usr/bin/env python3
"""
run_selected_sweeps.py

This script controls which data‐generation sweeps to execute, by toggling
three boolean flags at the top.  Each sweep invokes learn_hamiltonian.py
with a particular set of command‐line arguments.  As a result, for each
‘sweep’ we create a directory tree that looks like:

    <base_folder>/
      run_<SWEEP_IDENTIFIER>/
        alpha_<…>_perturb_<…>_measurements_<…>_shots_<…>_steps_<…>/
          config.json
          hamiltonians.json
          embedding_<codename>.pth
          embedding_<codename>_loss.json
          …

In other words, each call to learn_hamiltonian.py (with `--output-dir run_…`)
automatically builds the expected experimental parameter combination subfolders (alpha_…_perturb_…_…) 
and writes out everything needed for later plotting.  Once all sweeps complete,
you can run composite_plotting.py to produce Figures 1, 2, 3, and the derivative
comparison.

────────────────────────────────────────────────────────────────────────────────
Toggle which sweeps to run:
────────────────────────────────────────────────────────────────────────────────

run_sweep1:  “SWEEP 1” generates a data‐set in which
             • α = 1.0 (fixed)
             • perturb ∈ {1, 10, 25, 50, 100, 250, 500}  
             • measurements = 50  
             • shots = 1  
             • steps = 1..8 (internally swept)  

             For each value of “perturb”, learn_hamiltonian.py produces one 
             folder named run_SWEEP1_perturb_<pert>.  Inside that, it
             generates combo‐directories “alpha_1.000_perturb_<pert>_…/” for
             each time‐stamp step, writing config.json, hamiltonians.json,
             embeddings, etc.  In effect, SWEEP 1 explores “error vs total 
             experiment time” as we increase the spread‐state ensemble.

run_sweep2:  “SWEEP 2” generates a data‐set in which
             • α ∈ {0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}
             • perturb = 50 (fixed)
             • measurements = 25  
             • shots = 1  
             • steps = 1..8 (internally swept)  

             For each α, learn_hamiltonian.py produces run_SWEEP2_alpha_<α>/.  
             Inside, you get combo‐folders “alpha_<α>_perturb_50_…/”.  This
             sweep explores “error vs total experiment time” as α varies.

run_sweep3:  “SWEEP 3” generates a data‐set in which
             • α ∈ {0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}
             • perturb ∈ {1,10,25,50,100,250,500}  (internally swept)  
             • measurements = 25  
             • shots = 1  
             • steps = 8  (fixed)  

             For each α, learn_hamiltonian.py produces run_SWEEP3_alpha_<α>/.  
             Inside that, it creates combo‐folders “alpha_<α>_perturb_<p>_<…>/”
             for every perturbation value.  This sweep explores “error vs ensemble
             size” at fixed total experiment time.

────────────────────────────────────────────────────────────────────────────────
What happens in the background:
────────────────────────────────────────────────────────────────────────────────

1.  `reproduce_data_SWEEP1(base_folder)` calls learn_hamiltonian.py for each
    perturb in PERTURBS, passing:

      --alphas 1.0
      --perturbs <pert>
      --measurements 50
      --shots 1
      --steps 1,2,…,8
      --output-dir <base_folder>/run_SWEEP1_perturb_<pert>/

    Internally, learn_hamiltonian.py:
      • Reads its own defaults + config, builds a neural net predictor,
      • Loops over each time stamp (1 through 8), generates Hamiltonians,
      • Trains embeddings, saves model weights and metadata under
        alpha_1.000_perturb_<pert>_measurements_50_shots_1_steps_<N>/,
      • Exports config.json, hamiltonians.json, embedding_<codename>.pth, etc.

    At the end, you have “first_parameter_sweep_data/run_SWEEP1_perturb_<pert>/”
    containing exactly the directory structure that composite_plotting.py expects
    for Figure 1.

2.  `reproduce_data_SWEEP2(base_folder)` calls learn_hamiltonian.py for each α in ALPHAS:

      --alphas <α>
      --perturbs 50
      --measurements 25
      --shots 1
      --steps 1,2,…,8
      --output-dir <base_folder>/run_SWEEP2_alpha_<α>/

    Similarly, learn_hamiltonian.py writes out combo‐subfolders
    alpha_<α>_perturb_50_…/ for each time stamp.  This supplies data for “time‐scaling
    grouped by α” in Figure 2.

3.  `reproduce_data_SWEEP3(base_folder)` calls learn_hamiltonian.py for each α:

      --alphas <α>
      --perturbs 1,10,25,50,100,250,500
      --measurements 25
      --shots 1
      --steps 8
      --output-dir <base_folder>/run_SWEEP3_alpha_<α>/

    Now, within run_SWEEP3_alpha_<α>/, learn_hamiltonian.py internally loops
    over perturb = 1, 10, 25, …, 500 (because of the comma‐separated list).
    It fixes step count = 8.  You end up with combos
    alpha_<α>_perturb_<pert>_measurements_25_shots_1_steps_8/ for each perturb.
    This supplies data for “perturb‐scaling grouped by α” in Figure 3.

────────────────────────────────────────────────────────────────────────────────
Expected results:
────────────────────────────────────────────────────────────────────────────────

After running `run_selected.py` with, say, `run_sweep1 = True`:
  • Directory “first_parameter_sweep_data/” will be created.
  • Inside it: run_SWEEP1_perturb_1/, run_SWEEP1_perturb_10/, …, run_SWEEP1_perturb_500/.
  • Each run_SWEEP1_perturb_<p>/ contains subfolders:
      alpha_1.000_perturb_<p>_measurements_50_shots_1_steps_<N>/
    for N=1..8, each with config.json, hamiltonians.json, and embedding_*.pth.

Once all three flags are set and `python run_selected.py` completes, you should see:

    first_parameter_sweep_data/
      run_SWEEP1_perturb_1/
        alpha_1.000_perturb_1_measurements_50_shots_1_steps_1/
        alpha_1.000_perturb_1_measurements_50_shots_1_steps_2/
        …
      run_SWEEP1_perturb_10/
        alpha_1.000_perturb_10_measurements_50_shots_1_steps_1/
        …
      …
      run_SWEEP1_perturb_500/
        alpha_1.000_perturb_500_measurements_50_shots_1_steps_1/
        …

    second_parameter_sweep_data/
      run_SWEEP2_alpha_0.3/
        alpha_0.300_perturb_50_measurements_25_shots_1_steps_1/
        alpha_0.300_perturb_50_measurements_25_shots_1_steps_2/
        …
      run_SWEEP2_alpha_0.4/
      …
      run_SWEEP2_alpha_1.0/

    third_parameter_sweep_data/
      run_SWEEP3_alpha_0.3/
        alpha_0.300_perturb_1_…_steps_8/
        alpha_0.300_perturb_10_…_steps_8/
        …
        alpha_0.300_perturb_500_…_steps_8/
      run_SWEEP3_alpha_0.4/
      …
      run_SWEEP3_alpha_1.0/

Once that directory tree is present, you can run:

    python composite_plotting.py

and it will produce:

  • Figure 1 (“Error vs ∑time, colored/fitted by perturbation” and “β vs perturbation”)  
  • Figure 2 (“β vs α,” “alternative β vs α,” and “Error vs ∑time” for one chosen α)  
  • Figure 3 (“β vs α,” “alternative β vs α,” and “Error vs ∑perturb” for one chosen α)  
  • Derivative‐comparison plot dβ/dα vs α  

All outputs (plots, saved images, etc.) appear in whatever location
composite_plotting.py is configured to show them (usually displayed to screen
or saved by your own modifications).
"""

import sys
import os
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.abspath(os.path.join(THIS_DIR, "..", "src"))
sys.path.insert(0, SRC_DIR)

from reproduction_pipelines import (
    reproduce_data_SWEEP1,
    reproduce_data_SWEEP2,
    reproduce_data_SWEEP3
)

# ───────────────────────────────────────────────────────────────────────────────
#   Toggle which sweeps to re‐generate.  Set to True if you want that dataset
#   to be re‐built from scratch by calling learn_hamiltonian.py.  If False,
#   we assume data already exists in the corresponding folder.
# ───────────────────────────────────────────────────────────────────────────────
run_sweep1 = True    # SWEEP 1: perturb‐sweep (α=1.0), measurements=50
run_sweep2 = False   # SWEEP 2: α‐sweep (perturb=50), measurements=25
run_sweep3 = False   # SWEEP 3: nested α+perturb, measurements=25
# ───────────────────────────────────────────────────────────────────────────────


def main():
    """
    Main entry.  Checks three Boolean flags and calls the appropriate
    reproduction function(s).  Each function writes into a “base_folder”
    (creating it if necessary), then within that folder creates subfolders 
    named run_SWEEP1_…, run_SWEEP2_…, or run_SWEEP3_… respectively.

    After running this script, you should see three top‐level directories:
      • first_parameter_sweep_data/
      • second_parameter_sweep_data/
      • third_parameter_sweep_data/

    Inside each, a collection of “run_…” subfolders, each containing all
    the combo directories learn_hamiltonian.py generated.
    """
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))

    # Define the three top‐level destinations where each sweep’s “run_…” folders live:
    first_folder  = os.path.join(THIS_DIR, "first_parameter_sweep_data")
    second_folder = os.path.join(THIS_DIR, "second_parameter_sweep_data")
    third_folder  = os.path.join(THIS_DIR, "third_parameter_sweep_data")

    if run_sweep1:
        print("→ Generating SWEEP 1 data (perturbation sweep, measurements=50)…")
        reproduce_data_SWEEP1(first_folder)

    if run_sweep2:
        print("→ Generating SWEEP 2 data (α sweep, measurements=25)…")
        reproduce_data_SWEEP2(second_folder)

    if run_sweep3:
        print("→ Generating SWEEP 3 data (nested α+perturb, measurements=25)…")
        reproduce_data_SWEEP3(third_folder)

    print("\nrun_selected.py finished. Check these directories for generated data:")
    print("  ", first_folder)
    print("  ", second_folder)
    print("  ", third_folder)


if __name__ == "__main__":
    main()
