# On the Fisher Information's Temporal Regimes in Quantum Hamiltonian Learning and their Exploitation

## Abstract

In this work, we introduce a novel quantum Hamiltonian Learning (HL) strategy based on Haar‐random single‐qubit state preparation. By examining three temporal regimes of the Fisher information—and their relation to the Heisenberg and Standard Quantum Limit (SQL)—we show how one can surpass the SQL with our strategy. We then validate our theoretical findings with numerical simulations which learn a disordered, anisotropic Heisenberg model with local transverse fields, without prior knowledge about structure or coefficients. Most notably, our work requires only state preparation control, with one‐local Haar‐random unitary operations with respect to a fixed (separable) reference state. This means that, counter to recent works, we can show that one may surpass the SQL with a strategy that is evolution‐control free and many‐body control free, making it particularly well suited to experimental applications. Code for our method is available online and open‐source.

---

This repository provides the code and data for our Hamiltonian learning strategy introduced in our manuscript, which leverages an “initial‐state spreading” protocol and explicitly explores the temporal regimes of Fisher information. It includes:

1. **Original (“manuscript”) data** (`replot_original_data/`)  
   Scripts under `replot_original_data/` that contain the _original_ data folders and produce all plots exactly as in the manuscript:  
   - **Figure 1:** Error vs ∑time (colored/fitted by perturbation) and β vs perturbation.  
   - **Figure 2:** β vs α, alternative β vs α (theory + linear‐fit), Error vs ∑time for one α.  
   - **Figure 3:** β vs α, alternative β vs α (theory + linear‐fit), Error vs ∑perturb for one α.  
   - **Derivative comparison:** dβ/dα vs α (time‐scaling vs perturb‐scaling).  

   You can re‐plot Figures 1–3 (and the derivative comparison) without re‐training from the datasets in the same directory by using the `composite_replotting.py` script:  
   - `first_parameter_sweep_data/` → Figure 1 data (α = 1.0; varying perturbation; measurements = 50; steps = 1…8)  
   - `second_parameter_sweep_data/` → Figure 2 data (perturb = 50; varying α; measurements = 25; steps = 1…8)  
   - `third_parameter_sweep_data/` → Figure 3 data (steps = 8; varying α and perturb; measurements = 25)  
   - `composite_replotting.py` → orchestrates all re‐plotting pipelines for Figures 1–3 and the derivative comparison.

2. **Data‐(re-)generation pipelines** (`reproduce_original_data/`)  
   Scripts to re‐generate the embedding files (`.pth` weights + loss logs) for all three “figures” (Figure 1, 2, 3) via `learn_hamiltonian.py`.  
   - The main controller is `rerun_selected_sweeps.py`, where you toggle three flags to choose which sweeps (SWEEP 1, 2, 3) to run.  
   - Each sweep produces a set of folders under:
     ```
     first_parameter_sweep_data/     ← SWEEP 1 (α = 1.0, varying perturbation; measurements = 50; steps = 1…8)
     second_parameter_sweep_data/    ← SWEEP 2 (perturb = 50; varying α; measurements = 25; steps = 1…8)
     third_parameter_sweep_data/     ← SWEEP 3 (steps = 8; varying α and perturb; measurements = 25)
     ```
   - Each `run_…` subfolder then contains “combo” directories of the form  
     `alpha_<…>_perturb_<…>_measurements_<…>_shots_<…>_steps_<…>/`  
     with `config.json`, `hamiltonians.json`, `embedding_<codename>.pth`, etc.

   You can also plot Figures 1–3 (and the derivative comparison) from the reproduced data using the `composite_replotting.py` script, which orchestrates all plotting pipelines for Figures 1–3 from the datasets:
   - `first_parameter_sweep_data/` → Figure 1 data (α = 1.0; varying perturbation; measurements = 50; steps = 1…8)  
   - `second_parameter_sweep_data/` → Figure 2 data (perturb = 50; varying α; measurements = 25; steps = 1…8)  
   - `third_parameter_sweep_data/` → Figure 3 data (steps = 8; varying α and perturb; measurements = 25)  

3. **Source code** (`src/`)  
   - `learn_hamiltonian.py` – main training script  
   - `predictor.py`, `loss.py`, `hamiltonian_generator.py`, `datagen.py`, `utils.py` (helper modules)  
   - `extraction_and_evaluation.py` – loads embeddings, reconstructs density matrices, computes recovery errors, fits β across Fisher regimes  
   - `plotting_utils.py`, `plotting_pipelines.py` – collect & plot errors, compute β, draw Fisher‐regime figures  
   - `reproduction_pipelines.py` – functions to re‐generate all sweeps  

---

## Requirements

- **Python 3.10+**  
- PyTorch (≥ 1.13 recommended)  
- NumPy, SciPy, scikit‐learn, tqdm, Matplotlib  

Install all dependencies via:

```bash
pip install -r requirements.txt
