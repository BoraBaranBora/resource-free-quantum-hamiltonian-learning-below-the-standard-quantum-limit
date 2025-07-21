# Resource-Free Quantum Hamiltonian Learning Below the Standard Quantum Limit

## Abstract

Accurate and resource-efficient estimation of quantum Hamiltonians is crucial for developing practical quantum technologies, yet current methods typically demand entanglement resources or dynamical control. Here, we demonstrate a method that surpasses the standard quantum limit without requiring entanglement resources, coherent measurements, or dynamical control. Our method relies on trajectory-based Hamiltonian learning, in which we apply local, randomized pre-processing to probe states and apply the maximum-likelihood estimation to optimally scheduled Pauli measurements. Analytically, we establish the emergence of a transient Heisenberg-limited regime for short-time probes for our procedure. Furthermore, we outline how to estimate all Hamiltonian parameters in parallel using ensembles of probe states, removing the need for parameter isolation and structural priors. Finally, we supplement our findings with a numerical study, learning multiple disordered, anisotropic Heisenberg models for a 1D chain of spin-1/2 particles, featuring local transverse fields with both nearest- and next-nearest-neighbour interactions, as well as a gapless XXZ Hamiltonian. Importantly, our numerics show that our method needs only one shot per Pauli measurement, making it well-suited for experimental scenarios. The code for our method is available online and open-source.

---

This repository provides the code and data for the Hamiltonian learning strategy introduced in our publication. It includes:

1. **Original (“publication”) data** (`replot_original_data/`)  
   Scripts under `replot_original_data/` that contain the _original_ data folders and produce all plots exactly as in the publication:  
   - **Figure 1:** Error vs ∑time for α = 1.0 (four Hamiltonian families).  
   - **Figure 2:** Error-scaling exponent β vs scheduling exponent α (with spreadings = 32).  
   - **Figure 3(a, b):** (a) Error vs ∑time for α = 1.0 with increasing number of state spreadings; (b) Fitted β vs number of spreadings.

   You can re‐plot Figures 1–3 (and the derivative comparison) without re‐training from the datasets in the same directory by using the `composite_replotting.py` script:  
   - `first_parameter_sweep_data/` → Figures 1 and 2 data (spreadings = 32; varying α; measurements = 25; steps = 1…8)  
   - `second_parameter_sweep_data/` → Figure 3 data (α = 1.0; varying number of state spreadings; measurements = 25; steps = 1…8)  

   - `composite_replotting.py` → orchestrates all re‐plotting pipelines for Figures 1–3 and the derivative comparison.

2. **Data‐(re-)generation pipelines** (`reproduce_original_data/`)  
   Scripts to re‐generate the embedding files (`.pth` weights + loss logs) for all figure data via `learn_hamiltonian.py`.  
   - The main controller is `rerun_selected_sweeps.py`, where you toggle flags to choose which sweeps (SWEEP 1 or 2) to run.  
   - Each sweep produces a set of folders under:
     ```
     first_parameter_sweep_data/     ← SWEEP 1 (α = 1.0, varying number of state spreadings; measurements = 25; steps = 1…8)
     second_parameter_sweep_data/    ← SWEEP 2 (spreadings = 32; varying α; measurements = 25; steps = 1…8)
     ```
   - Each `run_…` subfolder then contains directories of the form  
     `alpha_<…>_spreadings_<…>_measurements_<…>_shots_<…>_steps_<…>/`  
     with `config.json`, `hamiltonians.json`, `embedding_<codename>.pth`, etc.

   You can also re-plot Figures 1–2 (and the derivative comparison) from the reproduced data using the `composite_replotting.py` script.

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
