# Original Data Replotting

This subdirectory contains everything needed to go from the raw “combo” embedding folders to the final figures in the manuscript—without having to commit multi‐GB embedding files. By splitting the workflow into two steps (precomputation of errors → plotting), you can commit only a small `cached_errors/` folder (a few MB) instead of all raw embeddings.

---

## Directory Structure

Place the following items at this level:

replot_original_data/
├─ composite_replotting.py
└─ cached_errors/ ← created by running precompute_errors.py
    ├─ sweep1_errors.pkl
    ├─ sweep2_errors.pkl
    └─ sweep3_errors.pkl

- **`composite_replotting.py`**  
  Orchestrates all plotting pipelines (Figures 1–3). It expects to load precomputed error–data pickles from `cached_errors/

- **`cached_errors/` (auto‐generated)**  
  Contains the three pickles listed above. Once these exist, `composite_replotting.py` can load them directly (avoiding hours of re‐processing all embeddings).

---

## Scripts Overview

### `composite_replotting.py`

- **Purpose**  
  Generate the final composite figures after completing all three parameter sweeps. These plots summarize error and β‐scaling relationships across time and perturbation dimensions.

- **How It Works**  
  1. Copy or symlink `composite_replotting.py` from `replot_original_data/` into this directory (or adjust paths accordingly).  
  2. Run:
     ```bash
     python composite_replotting.py
     ```  
  3. The script will load cached sweep data (produced by `precompute_errors.py`) and produce the following outputs:  
     - **Figure 1**: “Error vs ∑time” (colored/fitted by perturbation) and “β vs perturbation”  
     - **Figure 2**: “β vs α” (standard & theory+fit) and “Error vs ∑time” for one representative α  
     - **Figure 3**: “β vs α” (standard & theory+fit) and “Error vs ∑perturb” for one representative α  
     - **Derivative‐comparison**: dβ/dα vs α (time‐ vs perturb‐scaling)  
  4. Plots will appear on‐screen (or be saved to disk, depending on your plotting configuration).


