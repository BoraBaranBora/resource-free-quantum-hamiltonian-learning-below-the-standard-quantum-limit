# Data Reproduction Pipelines

This subdirectory contains everything needed to re‐generate the embedding files and metadata (“combo” folders) from scratch—so that you can recreate the exact datasets used for Figures 1–3. Once these datasets exist, you can run the plotting scripts (in `replot_original_data/`) to produce all manuscript figures.

---

## Directory Structure

Place the following items at this level:



reproduce_original_data/
├── rerun_selected_sweeps.py
├── first_parameter_sweep_data/     ← created by SWEEP 1 when rerun_selected_sweeps.py
├── second_parameter_sweep_data/    ← created by SWEEP 2 when rerun_selected_sweeps.py
├── precompute_errors.py
├── cached_errors/                  ← created by running precompute_errors.py (optional)
│   ├── sweep1_errors.pkl
│   ├── sweep2_errors.pkl
└── composite_replotting.py


- **`first_parameter_sweep_data/`**, **`second_parameter_sweep_data/`**  
  Each of these folders contains subfolders named `run_SWEEP#_<…>/`, and inside each `run_…` folder are the “combo” directories (e.g. `alpha_<…>_spreadings_<…>_measurements_<…>_shots_<…>_steps_<…>/`) with:
  - `config.json`
  - `hamiltonians.json`
  - `embedding_<codename>.pth`
  - `embedding_<codename>_loss.json`
  - …etc.

  These correspond to simulations over:
- \( R \) spread initial states: \( \{ \ket{\psi_r} \}_{r=1}^{R} \)
- \( m_t \) evolution times \( t_j \)
- \( K \) Pauli product measurement bases \( p_k \)
- \( S \) shots per measurement setting (typically \( S = 1 \))


- **`composite_replotting.py`**  
  Orchestrates all plotting pipelines (Figures 1–3). It expects to load precomputed error–data pickles from `cached_errors/`. If those pickles don’t yet exist, it will fall back to recomputing errors—so it’s recommended to run `precompute_errors.py` first.

- **`precompute_errors.py`**  
  Traverses each “run_…” and “combo” folder, calls `collect_recovery_errors_from_data(...)` to extract and aggregate all recovery‐error tuples, then writes two pickle files:
  - `cached_errors/sweep1_errors.pkl`  
    (time‐scaling grouped by α, from `first_parameter_sweep_data/`, used for Figures 1 and 2)
  - `cached_errors/sweep2_errors.pkl`  
    (time‐scaling grouped by number of spreadings, from `second_parameter_sweep_data/`, used for Figure 3)

- **`cached_errors/` (auto‐generated)**  
  Contains the two pickles listed above. Once these exist, `composite_replotting.py` can load them directly (avoiding hours of re‐processing all embeddings).

---

## Scripts Overview

### 1. `rerun_selected_sweeps.py`

- **Purpose**  
  Re‐generate all “combo” embedding datasets by invoking `learn_hamiltonian.py` with the correct parameters for each sweep.

- **Toggle Which Sweeps to Run**  
  At the top of `rerun_selected_sweeps.py`, set the booleans:
  ```python
  run_sweep1 = True    # SWEEP 1: α ∈ {0.1, ..., 1.0}, spreadings = 32, measurements = 25, shots = 1, steps = 1..8
  run_sweep2 = False   # SWEEP 2: α = 1.0 (fixed), spreadings ∈ {1, 2, ..., 128}, measurements = 25, shots = 1, steps = 1..8


### SWEEP 1

Creates `first_parameter_sweep_data/run_SWEEP1_alpha_<α>/` for each  
`α ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}`.

Inside each `run_SWEEP1_alpha_<α>/`, you’ll find combo directories named:  

alpha_<α>spreadings_32_measurements_25_shots_1_steps<m_t>/

for `\(m_t\) = 1..8`.

---

### SWEEP 2

Creates `second_parameter_sweep_data/run_SWEEP2_spreadings_<r>/` for each  
`R ∈ {1, 2, 4, 8, 16, 32, 64, 128}`.

Inside each `run_SWEEP2_spreadings_<r>/`, you’ll find combo directories named:  

alpha_1.000_spreadings_<p>measurements_25_shots_1_steps<N>/

for `\(m_t\) = 1..8`.

---


### What Happens Under the Hood

For each enabled sweep, the script calls one of:

```python
reproduce_data_SWEEP1(first_folder)   # SWEEP 1: α sweep
reproduce_data_SWEEP2(second_folder)  # SWEEP 2: spreadings sweep

Each function invokes learn_hamiltonian.py (from src/) with the appropriate command-line arguments:

--alphas
--spreadings
--measurements
--shots
--steps
--output-dir

learn_hamiltonian.py builds the neural network predictor, loops over the requested time or spreadings values, trains embeddings, and writes out:
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
  1. For each `run_*` subfolder in `first_parameter_sweep_data/`, calls:  
     ```python
     collect_recovery_errors_from_data(run_folder_path, scaling_param="times", group_by="alpha")
     ```  
     and merges all results into a single dictionary (used for **Figures 1 and 2**).
  
  2. Repeats for `second_parameter_sweep_data/` with:  
     ```python
     scaling_param="times", group_by="spreadings"
     ```  
     (used for **Figure 3**).
  
  3. Writes out two pickles into `cached_errors/`:  
     - `sweep1_errors.pkl`  ← α-sweep (first sweep)  
     - `sweep2_errors.pkl`  ← spreadings-sweep (second sweep)



### 3. `composite_replotting.py`

- **Purpose**  
  Generate the final composite figures after completing the two parameter sweeps. These plots summarize error and β‐scaling relationships across time, scheduling α, and the number of initial state spreadings.

- **How It Works**    
  1. Run:
     ```bash
     python composite_replotting.py
     ```  

  2. The script will load cached sweep data (produced by `precompute_errors.py`) and produce the following outputs:

     - **Figure 1:** Error vs ∑time for α = 1.0 across four Hamiltonians  
     - **Figure 2:** Error-scaling exponent β vs scheduling parameter α  
     - **Figure 3(a, b):**  
       - (a) Error vs ∑time for increasing number of state spreadings (α = 1.0)  
       - (b) β vs number of spreadings

  3. Plots will appear on-screen (or be saved to disk, depending on your plotting configuration).
