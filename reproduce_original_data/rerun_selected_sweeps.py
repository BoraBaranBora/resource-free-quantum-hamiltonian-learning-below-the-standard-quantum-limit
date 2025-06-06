#!/usr/bin/env python3
"""
rerun_selected_sweeps.py

This script controls which data‐generation sweeps to execute, by toggling
three boolean flags at the top.  Each sweep invokes learn_hamiltonian.py
with a particular set of command‐line arguments.  As a result, for each
‘sweep’ we create a directory tree that looks like:

    <base_folder>/
      run_<SWEEP_IDENTIFIER>/
        alpha_<…>_spreading_<…>_measurements_<…>_shots_<…>_steps_<…>/
          config.json
          hamiltonians.json
          embedding_<codename>.pth
          embedding_<codename>_loss.json
          …

In other words, each call to learn_hamiltonian.py (with `--output-dir run_…`)
automatically builds the expected experimental parameter combination subfolders (alpha_…_spreading_…_…) 
and writes out everything needed for later plotting.  Once all sweeps complete,
you can run composite_plotting.py to produce Figures 1, 2, 3, and the derivative
comparison.

────────────────────────────────────────────────────────────────────────────────
Toggle which sweeps to run:
────────────────────────────────────────────────────────────────────────────────

run_sweep1:  “SWEEP 1” generates a data‐set in which
             • α = 1.0 (fixed)
             • spreading ∈ {1, 10, 25, 50, 100, 250, 500}  
             • measurements = 50  
             • shots = 1  
             • steps = 1..8 (internally swept)  

             For each value of “spreading”, learn_hamiltonian.py produces one 
             folder named run_SWEEP1_spreading_<pert>.  Inside that, it
             generates combo‐directories “alpha_1.000_spreading_<pert>_…/” for
             each time‐stamp step, writing config.json, hamiltonians.json,
             embeddings, etc.  In effect, SWEEP 1 explores “error vs total 
             experiment time” as we increase the spread‐state ensemble.

run_sweep2:  “SWEEP 2” generates a data‐set in which
             • α ∈ {0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}
             • spreading = 50 (fixed)
             • measurements = 25  
             • shots = 1  
             • steps = 1..8 (internally swept)  

             For each α, learn_hamiltonian.py produces run_SWEEP2_alpha_<α>/.  
             Inside, you get combo‐folders “alpha_<α>_spreading_50_…/”.  This
             sweep explores “error vs total experiment time” as α varies.

run_sweep3:  “SWEEP 3” generates a data‐set in which
             • α ∈ {0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}
             • spreading ∈ {1,10,25,50,100,250,500}  (internally swept)  
             • measurements = 25  
             • shots = 1  
             • steps = 8  (fixed)  

             For each α, learn_hamiltonian.py produces run_SWEEP3_alpha_<α>/.  
             Inside that, it creates combo‐folders “alpha_<α>_spreading_<p>_<…>/”
             for every spreading value.  This sweep explores “error vs ensemble
             size” at fixed total experiment time.

────────────────────────────────────────────────────────────────────────────────
What happens in the background:
────────────────────────────────────────────────────────────────────────────────

1.  `reproduce_data_SWEEP1(base_folder)` calls learn_hamiltonian.py for each
    spreading in spreadingS, passing:

      --alphas 1.0
      --spreadings <pert>
      --measurements 50
      --shots 1
      --steps 1,2,…,8
      --output-dir <base_folder>/run_SWEEP1_spreading_<pert>/

    Internally, learn_hamiltonian.py:
      • Reads its own defaults + config, builds a neural net predictor,
      • Loops over each time stamp (1 through 8), generates Hamiltonians,
      • Trains embeddings, saves model weights and metadata under
        alpha_1.000_spreading_<pert>_measurements_50_shots_1_steps_<N>/,
      • Exports config.json, hamiltonians.json, embedding_<codename>.pth, etc.

    At the end, you have “first_parameter_sweep_data/run_SWEEP1_spreading_<pert>/”
    containing exactly the directory structure that composite_plotting.py expects
    for Figure 1.

2.  `reproduce_data_SWEEP2(base_folder)` calls learn_hamiltonian.py for each α in ALPHAS:

      --alphas <α>
      --spreadings 50
      --measurements 25
      --shots 1
      --steps 1,2,…,8
      --output-dir <base_folder>/run_SWEEP2_alpha_<α>/

    Similarly, learn_hamiltonian.py writes out combo‐subfolders
    alpha_<α>_spreading_50_…/ for each time stamp.  This supplies data for “time‐scaling
    grouped by α” in Figure 2.

3.  `reproduce_data_SWEEP3(base_folder)` calls learn_hamiltonian.py for each α:

      --alphas <α>
      --spreadings 1,10,25,50,100,250,500
      --measurements 25
      --shots 1
      --steps 8
      --output-dir <base_folder>/run_SWEEP3_alpha_<α>/

    Now, within run_SWEEP3_alpha_<α>/, learn_hamiltonian.py internally loops
    over spreading = 1, 10, 25, …, 500 (because of the comma‐separated list).
    It fixes step count = 8.  You end up with combos
    alpha_<α>_spreading_<pert>_measurements_25_shots_1_steps_8/ for each spreading.
    This supplies data for “spreading‐scaling grouped by α” in Figure 3.

────────────────────────────────────────────────────────────────────────────────
Expected results:
────────────────────────────────────────────────────────────────────────────────

After running `run_selected.py` with, say, `run_sweep1 = True`:
  • Directory “first_parameter_sweep_data/” will be created.
  • Inside it: run_SWEEP1_spreading_1/, run_SWEEP1_spreading_10/, …, run_SWEEP1_spreading_500/.
  • Each run_SWEEP1_spreading_<p>/ contains subfolders:
      alpha_1.000_spreading_<p>_measurements_50_shots_1_steps_<N>/
    for N=1..8, each with config.json, hamiltonians.json, and embedding_*.pth.

Once all three flags are set and `python run_selected.py` completes, you should see:

    first_parameter_sweep_data/
      run_SWEEP1_spreading_1/
        alpha_1.000_spreading_1_measurements_50_shots_1_steps_1/
        alpha_1.000_spreading_1_measurements_50_shots_1_steps_2/
        …
      run_SWEEP1_spreading_10/
        alpha_1.000_spreading_10_measurements_50_shots_1_steps_1/
        …
      …
      run_SWEEP1_spreading_500/
        alpha_1.000_spreading_500_measurements_50_shots_1_steps_1/
        …

    second_parameter_sweep_data/
      run_SWEEP2_alpha_0.3/
        alpha_0.300_spreading_50_measurements_25_shots_1_steps_1/
        alpha_0.300_spreading_50_measurements_25_shots_1_steps_2/
        …
      run_SWEEP2_alpha_0.4/
      …
      run_SWEEP2_alpha_1.0/

    third_parameter_sweep_data/
      run_SWEEP3_alpha_0.3/
        alpha_0.300_spreading_1_…_steps_8/
        alpha_0.300_spreading_10_…_steps_8/
        …
        alpha_0.300_spreading_500_…_steps_8/
      run_SWEEP3_alpha_0.4/
      …
      run_SWEEP3_alpha_1.0/

Once that directory tree is present, you can run:

    python composite_plotting.py

and it will produce:

  • Figure 1 (“Error vs ∑time, colored/fitted by spreading” and “β vs spreading”)  
  • Figure 2 (“β vs α,” “alternative β vs α,” and “Error vs ∑time” for one chosen α)  
  • Figure 3 (“β vs α,” “alternative β vs α,” and “Error vs spreading” for one chosen α)  
  • Derivative‐comparison plot dβ/dα vs α  

All outputs (plots, saved images, etc.) appear in whatever location
composite_plotting.py is configured to show them (usually displayed to screen
or saved by your own modifications).
"""

#!/usr/bin/env python3
"""
rerun_selected_sweeps.py

This script controls which data‐generation sweeps to execute, by toggling
three boolean flags at the top.  Each sweep invokes learn_hamiltonian.py
with a particular set of command‐line arguments.  As a result, for each
‘sweep’ we create a directory tree that looks like:

    <base_folder>/
      run_<SWEEP_IDENTIFIER>/
        alpha_<…>_spreading_<…>_measurements_<…>_shots_<…>_steps_<…>/
          config.json
          hamiltonians.json
          embedding_<codename>.pth
          embedding_<codename>_loss.json
          …

Each reproduction function now takes an extra `families` string so you can
select which Hamiltonian families to sweep without changing reproduction_pipelines.py.
"""

import sys
import os

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
run_sweep1 = False    # SWEEP 1: spreading‐sweep (α=1.0), measurements=50
run_sweep2 = True     # SWEEP 2: α‐sweep (spreading=50), measurements=25
run_sweep3 = True     # SWEEP 3: nested α+spreading, measurements=25

# ───────────────────────────────────────────────────────────────────────────────
#   Specify which families to sweep (comma‐separated list).  For example:
#     "Heisenberg", "XYZ2", "Heisenberg3,XYZ2", etc.
# ───────────────────────────────────────────────────────────────────────────────
chosen_families = "XYZ2, XYZ3" #"Heisenberg"
# ───────────────────────────────────────────────────────────────────────────────


def main():
    """
    Main entry.  Checks three Boolean flags and calls the appropriate
    reproduction function(s), passing along the `chosen_families` string.
    Each function writes into a “base_folder” (creating it if necessary),
    then within that folder creates subfolders named run_SWEEP1_…,
    run_SWEEP2_…, or run_SWEEP3_… respectively.
    """
    # Define the three top‐level destinations where each sweep’s “run_…” folders live:
    first_folder  = os.path.join(THIS_DIR, "first_parameter_sweep_data")
    second_folder = os.path.join(THIS_DIR, "second_parameter_sweep_data")
    third_folder  = os.path.join(THIS_DIR, "third_parameter_sweep_data")

    if run_sweep1:
        print("→ Generating SWEEP 1 data (spreading sweep, measurements=50)…")
        reproduce_data_SWEEP1(first_folder, chosen_families)

    if run_sweep2:
        print("→ Generating SWEEP 2 data (α sweep, measurements=25)…")
        reproduce_data_SWEEP2(second_folder, chosen_families)

    if run_sweep3:
        print("→ Generating SWEEP 3 data (nested α+spreading, measurements=25)…")
        reproduce_data_SWEEP3(third_folder, chosen_families)

    print("\nrun_selected.py finished. Check these directories for generated data:")
    print("  ", first_folder)
    print("  ", second_folder)
    print("  ", third_folder)


if __name__ == "__main__":
    main()
