# Replot Original Data

This directory contains everything needed to regenerate and plot the results for Sweeps 1, 2, and 3 using precomputed error caches. Below is a detailed description of the folder structure, how the raw embeddings were generated, how the caches are organized, and how to invoke the plotting pipeline.

---

## Directory Layout


first_parameter_sweep_data/
├── run_SWEEP1_perturb_1/
│ ├── alpha_1.000_perturb_1_measurements_50_shots_1_steps_1/
│ ├── alpha_1.000_perturb_1_measurements_50_shots_1_steps_2/
│ └── … up to steps=8
├── run_SWEEP1_perturb_10/
│ └── … (same structure)
└── … (one run_SWEEP1_perturb_<p> per perturb value)

second_parameter_sweep_data/
├── run_SWEEP2_alpha_0.3/
│ ├── alpha_0.300_perturb_50_measurements_25_shots_1_steps_1/
│ ├── alpha_0.300_perturb_50_measurements_25_shots_1_steps_2/
│ └── … up to steps=8
├── run_SWEEP2_alpha_0.4/
│ └── … (same structure)
└── … (one run_SWEEP2_alpha_<α> per α value)

third_parameter_sweep_data/
├── run_SWEEP3_alpha_0.3/
│ ├── alpha_0.300_perturb_1_measurements_25_shots_1_steps_8/
│ ├── alpha_0.300_perturb_10_measurements_25_shots_1_steps_8/
│ └── … (all perturb values at steps=8)
├── run_SWEEP3_alpha_0.4/
│ └── … (same structure)
└── … (one run_SWEEP3_alpha_<α> per α value)

cached_errors/
├── sweep1_errors.pkl
├── sweep2_errors.pkl
└── sweep3_errors.pkl

composite_replotting.py
precompute_errors.py
README.md


### 1) Raw Data Folders

- **`first_parameter_sweep_data/`**  
  Contains subdirectories for Sweep 1. Each `run_SWEEP1_perturb_<p>/` folder holds one “perturbation” experiment. Inside each, there are eight “combo” subfolders (one per `steps` value). Each combo folder includes:  
  - `config.json`  
  - `hamiltonians.json`  
  - `embedding_<codename>.pth` (the learned embedding parameters)  
  - `embedding_<codename>_loss.json`  
  - …any additional metadata files.  

  **Purpose**: These raw embeddings are used to compute recovery errors as a function of *time stamps*. We then aggregate them by perturbation value, fit a power-law exponent β, and produce two plots:  
  1. Error vs ∑(time-stamps), colored by perturbation.  
  2. β vs perturbation (error-bar plot).  

- **`second_parameter_sweep_data/`**  
  Contains subdirectories for Sweep 2. Each `run_SWEEP2_alpha_<α>/` folder holds one “α” experiment (perturb = 50). Inside each are eight combos (one per `steps`). Each combo has the same file types (`config.json`, `hamiltonians.json`, `*.pth`, etc.).  

  **Purpose**: These raw embeddings feed our “time-scaling by α” analysis. We:  
  1. Aggregate recovery errors for each α → fit β(α), plot β vs α (error-bar).  
  2. Pick the largest α → plot Error vs ∑(time-stamps), including SQL (∝ ∑time⁻½) and Heisenberg (∝ ∑time⁻¹) reference lines.  

- **`third_parameter_sweep_data/`**  
  Contains subdirectories for Sweep 3. Each `run_SWEEP3_alpha_<α>/` folder holds one “α” experiment (steps = 8). Inside each are multiple combos (one per perturbation value).  

  **Purpose**: These raw embeddings feed our “perturb-scaling by α” analysis. We:  
  1. Aggregate recovery errors for each α → fit β(α), plot β vs α (error-bar).  
  2. Pick the largest α → plot Error vs ∑(perturb), including SQL and Heisenberg references.  

---

## 2) Why Precompute “Cached Errors”?

Each combo folder can contain dozens (or hundreds) of Hamiltonians. Computing `errors_by_scaling` means:

1. Loading `config.json` and `hamiltonians.json` to reconstruct metadata.  
2. Loading each `embedding_<codename>.pth` into a PyTorch `Predictor`.  
3. Forward-passing a random input to get the reconstructed density matrix.  
4. Computing a mean-absolute error against the true Hamiltonian.  
5. Organizing those error values into a dictionary keyed by either “times” or “perturb”.

Performing this for *every* combo in *every* `run_SWEEP*` folder can take tens of minutes (or more). By running `precompute_errors.py` once, we:

- Traverse all three sweep folders (Sweep 1/2/3).  
- Build a single `combined_errors` dictionary for each sweep:  
  - **Sweep 1**: `{ time_stamps_tuple → [ (error, true_family, perturb) ] }`  
  - **Sweep 2**: `{ time_stamps_tuple → [ (error, true_family, alpha) ] }`  
  - **Sweep 3**: `{ perturb_tuple     → [ (error, true_family, alpha) ] }`  
- Serialize that dictionary as:  
  - `cached_errors/sweep1_errors.pkl`  
  - `cached_errors/sweep2_errors.pkl`  
  - `cached_errors/sweep3_errors.pkl`  

Later, when `composite_replotting.py` calls into `plotting_pipelines.py`, each “sweep” function simply loads its corresponding `*.pkl` and skips all the raw embedding reads. This makes plotting nearly instantaneous.

---

## How to Use

1. **Generate raw embeddings**  
   If you haven’t already, run whichever training or reproduction pipeline populates  
   `first_parameter_sweep_data/`, `second_parameter_sweep_data/`, and `third_parameter_sweep_data/`. For example:
   ```bash
   cd <ProjectRoot>/replot_original_data
   python run_selected.py
