# Original Data Replotting

This subdirectory contains everything needed to go from the raw “combo” embedding folders to the final figures in the manuscript-without having to download multi‐GB embedding files. By splitting the workflow into two steps (precomputation of errors → plotting), you can commit only a small `cached_errors/` folder (a few MB) instead of all raw embeddings.

---

## Directory Structure

Place the following items at this level:


replot_original_data/
├── cached_errors/                  
│   ├── sweep1_errors.pkl
│   ├── sweep2_errors.pkl
└── composite_replotting.py


- **`composite_replotting.py`**  
  Orchestrates all plotting pipelines (Figures 1–3). It expects to load precomputed error–data pickles from `cached_errors/`.

- **`cached_errors/` (auto‐generated)**  
  Contains the two pickles listed above. Once these exist, `composite_replotting.py` can load them directly (avoiding hours of re‐processing all embeddings).

---

## Scripts Overview

### `composite_replotting.py`

- **Purpose**  
  Generate the final composite figures after completing the two parameter sweeps. These plots summarize error and β‐scaling relationships across time, schedule shape, and ensemble size.

- **How It Works**   
  1. Run:
     ```bash
     python composite_replotting.py
     ```  
  2. The script will load cached sweep data (produced by `precompute_errors.py`) and produce the following outputs:  
   - **Figure 1:** Error vs ∑time for α = 1.0 (across four Hamiltonian families).  
   - **Figure 2:** Error-scaling exponent β vs α (with 32 spreadings).  
   - **Figure 3(a, b):** (a) Error vs ∑time for increasing number of spreadings (α = 1.0);  
     (b) β vs number of spreadings (for full Hamiltonian recovery).

  3. Plots will appear on‐screen (or be saved to disk, depending on your plotting configuration).
