# Data Reproduction Pipelines

This subdirectory contains everything needed to re‐generate the embedding files and metadata (“combo” folders) from scratch—so that you can recreate the exact datasets used for Figures 1–3. Once these datasets exist, you can run the plotting scripts (in `replot_original_data/`) to produce all manuscript figures.

---

## Directory Structure

Place the following items at this level:

reproduce_original_data/
├── rerun_selected_sweeps.py
├── first_parameter_sweep_data/     ← created by SWEEP 1 when rerun_selected_sweeps.py
├── second_parameter_sweep_data/    ← created by SWEEP 2 when rerun_selected_sweeps.py
├── third_parameter_sweep_data/     ← created by SWEEP 3 when rerun_selected_sweeps.py
├── precompute_errors.py
├── cached_errors/                  ← created by running precompute_errors.py (optional)
│   ├── sweep1_errors.pkl
│   ├── sweep2_errors.pkl
│   └── sweep3_errors.pkl
└── composite_replotting.py

- **`first_parameter_sweep_data/`**, **`second_parameter_sweep_data/`**, **`third_parameter_sweep_data/`**  
  Each of these folders should contain subfolders named `run_SWEEP#_<…>/`, and inside each “run_…” folder are the “combo” directories (e.g. `alpha_<…>_spreadings_<…>_measurements_<…>_shots_<…>_steps_<…>/`) with:
  - `config.json`
  - `hamiltonians.json`
  - `embedding_<codename>.pth`
  - `embedding_<codename>_loss.json`
  - …etc.

- **`composite_replotting.py`**  
  Orchestrates all plotting pipelines (Figures 1–3). It expects to load precomputed error–data pickles from `cached_errors/`. If those pickles don’t yet exist, it will fall back to recomputing errors—so it’s recommended to run `precompute_errors.py` first.

- **`precompute_errors.py`**  
  Traverses each “run_…” and “combo” folder, calls `collect_recovery_errors_from_data(...)` to extract and aggregate all recovery‐error tuples, then writes three pickle files:
  - `cached_errors/sweep1_errors.pkl`  
    (time‐scaling grouped by number of initial state spreadings, from `first_parameter_sweep_data/`)
  - `cached_errors/sweep2_errors.pkl`  
    (time‐scaling grouped by α, from `second_parameter_sweep_data/`)
  - `cached_errors/sweep3_errors.pkl`  
    (spreadings‐scaling grouped by α, from `third_parameter_sweep_data/`)

- **`cached_errors/` (auto‐generated)**  
  Contains the three pickles listed above. Once these exist, `composite_replotting.py` can load them directly (avoiding hours of re‐processing all embeddings).

---

## Scripts Overview

### 1. `rerun_selected_sweeps.py`

- **Purpose**  
  Re‐generate all “combo” embedding datasets by invoking `learn_hamiltonian.py` with the correct parameters for each sweep.

- **Toggle Which Sweeps to Run**  
  At the top of `rerun_selected_sweeps.py`, set the booleans:
  ```python
  run_sweep1 = True    # SWEEP 1: α=1.0 (fixed), spreadings ∈ {1,10,25,50,100,250,500}, measurements=50, shots=1, steps=1..8
  run_sweep2 = False   # SWEEP 2: α ∈ {0.3,0.4,…,1.0}, spreadings=50 (fixed), measurements=25, shots=1, steps=1..8
  run_sweep3 = False   # SWEEP 3: α ∈ {0.3,0.4,…,1.0}, spreadings ∈ {1,10,25,50,100,250,500}, measurements=25, shots=1, steps=8 (fixed)

### SWEEP 1

Creates `first_parameter_sweep_data/run_SWEEP1_spreadings_<p>/` for each `p ∈ {1,10,25,50,100,250,500}`.

Inside each `run_SWEEP1_spreadings_<p>/`, you’ll find combo directories named  
`alpha_1.000_spreadings_<p>_measurements_50_shots_1_steps_<N>/` for `N=1..8`.

Each combo folder contains `config.json`, `hamiltonians.json`, and the trained embeddings (`.pth`) and loss logs.

---

### SWEEP 2

Creates `second_parameter_sweep_data/run_SWEEP2_alpha_<α>/` for each `α ∈ {0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}`.

Inside each `run_SWEEP2_alpha_<α>/`, you’ll find  
`alpha_<α>_spreadings_50_measurements_25_shots_1_steps_<N>/` for `N=1..8`.

---

### SWEEP 3

Creates `third_parameter_sweep_data/run_SWEEP3_alpha_<α>/` for each `α ∈ {0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}`.

Inside each `run_SWEEP3_alpha_<α>/`, you’ll find combo folders  
`alpha_<α>_spreadings_<p>_measurements_25_shots_1_steps_8/` for each `p ∈ {1,10,25,50,100,250,500}`.

---

### What Happens Under the Hood

For each enabled sweep, the script calls one of:

reproduce_data_SWEEP1(first_folder)
reproduce_data_SWEEP2(second_folder)
reproduce_data_SWEEP3(third_folder)


where each function invokes `learn_hamiltonian.py` (from `src/`) with the appropriate `--alphas`, `--spreadingss`, `--measurements`, `--shots`, `--steps`, and `--output-dir` arguments.

`learn_hamiltonian.py` builds the neural network predictor, loops over the requested time or spreadings values, trains embeddings, and writes out:

- `config.json`
- `hamiltonians.json`
- `embedding_<codename>.pth`
- `embedding_<codename>_loss.json`
- …etc.

---

### 2. `precompute_errors.py`

- **Purpose**  
  Precompute and save aggregated recovery errors from the raw “combo” embeddings.  
  Without this step, `composite_replotting.py` would need to load and evaluate every `.pth` file on‐the‐fly, which can be very slow.

- **How It Works**  
  1. For each `run_*` subfolder in `first_parameter_sweep_data/`, calls  
     ```python
     collect_recovery_errors_from_data(run_folder_path, scaling_param="times", group_by="spreadings")
     ```  
     and merges all results into a single dictionary.  
  2. Repeats for `second_parameter_sweep_data/` with `scaling_param="times"`, `group_by="alpha"`.  
  3. Repeats for `third_parameter_sweep_data/` with `scaling_param="spreadings"`, `group_by="alpha"`.  
  4. Writes out three pickles into `cached_errors/`:  
     - `sweep1_errors.pkl`  
     - `sweep2_errors.pkl`  
     - `sweep3_errors.pkl`



### 3. `composite_replotting.py`

- **Purpose**  
  Generate the final composite figures after completing all three parameter sweeps. These plots summarize error and β‐scaling relationships across time and number of initial state spreadings dimensions.

- **How It Works**    
  1. Run:
     ```bash
     python composite_replotting.py
     ```  
  2. The script will load cached sweep data (produced by `precompute_errors.py`) and produce the following outputs:  
   - **Figure 1:** Error vs ∑time (colored/fitted by number of state initial state spreadings) and β vs number of state initial state spreadings.  
   - **Figure 2:** Error vs ∑time for one α and Error vs spreadings for one α.  
   - **Figure 3a:** β vs α, alternative β vs α (theory + linear‐fit), β vs α, alternative β vs α (theory + linear‐fit).  
   - **Figure 3a:** Derivative comparison: dβ/dα vs α (time‐scaling vs spreadings‐scaling).  
 
  3. Plots will appear on‐screen (or be saved to disk, depending on your plotting configuration).
