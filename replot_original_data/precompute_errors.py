#!/usr/bin/env python3
"""
precompute_errors.py

Why we do this:
---------------
Running the full “collect_recovery_errors_from_data(...)” over hundreds or
thousands of embedding files is time‐consuming (it needs to load each model,
run a forward pass, compute errors, etc.). By precomputing and storing the
aggregated `errors_by_scaling` dict as a single pickle, downstream plotting
routines can skip all that heavy work and load the cache instantly. This also
lets us commit a small “cached_errors/” folder to Git (a few MB), rather than
tracking the raw embeddings (which total multiple GB).

Directory layout:
---------------
Place this script in the same folder as your three “*_parameter_sweep_data/”
directories and your `composite_replotting.py`. For example:

    <ProjectRoot>/replot_original_data/
    ├─ first_parameter_sweep_data/
    ├─ second_parameter_sweep_data/
    ├─ composite_replotting.py
    └─ precompute_errors.py      ← save this here

When you run:

    python precompute_errors.py

it will traverse each “run_*/combo/” subfolder, compute `errors_by_scaling`
via `collect_recovery_errors_from_data(...)`, and then write:

    <ProjectRoot>/replot_original_data/cached_errors/
    ├─ sweep1_errors.pkl
    ├─ sweep2_errors.pkl

Afterward, your plotting pipelines (inside `plotting_pipelines.py`) simply
load these pickles instead of re-reading all the raw embeddings.

Usage:
---------------
    cd <ProjectRoot>/replot_original_data
    python precompute_errors.py

Screenshot:
---------------
(base) PS C:\…\replot_original_data> python precompute_errors.py
  → Precomputing errors for: …/first_parameter_sweep_data (scaling_param=times, group_by=spreading)
  → Pickled ... keys to: …/cached_errors/sweep1_errors.pkl
  → Precomputing errors for: …/second_parameter_sweep_data (scaling_param=times, group_by=alpha)
  → Pickled ... keys to: …/cached_errors/sweep2_errors.pkl


After this, running `composite_replotting.py` will be much faster:
plots will load from cached_errors/*.pkl instead of re-computing.

"""

import sys
import os
import pickle

# Ensure we can import collect_recovery_errors_from_data from src/
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.abspath(os.path.join(THIS_DIR, "..", "src"))
sys.path.insert(0, SRC_DIR)

from extraction_and_evalution import collect_recovery_errors_from_data

def ensure_dir(path: str):
    """Create `path` if it doesn’t already exist."""
    os.makedirs(path, exist_ok=True)

def precompute_for_base(
    base_path: str,
    scaling_param: str,
    group_by: str,
    out_fname: str
):
    """
    For each run_* subfolder under base_path:
      • Call collect_recovery_errors_from_data(...),
      • Merge all per-run dicts into a single `combined_errors` dict,
      • Write that dict as a pickle to cached_errors/out_fname.

Parameters:
  - base_path (str): Directory like "first_parameter_sweep_data/"
  - scaling_param (str): Either "times" or "spreading"
  - group_by (str): Either "spreading" or "alpha" (must differ from scaling_param)
  - out_fname (str): e.g. "sweep1_errors.pkl"
"""
    print(f"\n→ Precomputing errors for:\n    {base_path}\n"
          f"  (scaling_param={scaling_param}, group_by={group_by})")

    if not os.path.isdir(base_path):
        print(f"  [ERROR] Base path '{base_path}' not found. Skipping.")
        return

    combined_errors = {}
    run_dirs = sorted(d for d in os.listdir(base_path) if d.startswith("run_"))
    if not run_dirs:
        print(f"  [WARN] No 'run_*' folders found under '{base_path}'.")
    for run_folder in run_dirs:
        run_folder_path = os.path.join(base_path, run_folder)
        if not os.path.isdir(run_folder_path):
            print(f"  [WARN] Skipping '{run_folder_path}': Not a directory.")
            continue

        try:
            # This call may be expensive: it loads each embedding and computes errors
            errs = collect_recovery_errors_from_data(
                run_folder_path,
                scaling_param=scaling_param,
                group_by=group_by
            )
        except Exception as e:
            print(f"  [ERROR] Failed in '{run_folder_path}': {e}")
            continue

        # Merge per-run results into one dict
        for scale_tuple, triplets in errs.items():
            combined_errors.setdefault(scale_tuple, []).extend(triplets)

    if not combined_errors:
        print("  [WARN] No data collected; not writing pickle.")
        return

    # Write out the pickle under cached_errors/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir  = os.path.join(script_dir, "cached_errors")
    ensure_dir(cache_dir)

    out_path = os.path.join(cache_dir, out_fname)
    with open(out_path, "wb") as f:
        pickle.dump(combined_errors, f)
    print(f"  → Pickled {len(combined_errors)} keys to:\n      {out_path}")


if __name__ == "__main__":
    # Locate the three data folders relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 2) Sweep 2: time‐scaling grouped by alpha
    base1 = os.path.join(script_dir, "first_parameter_sweep_data")
    precompute_for_base(
        base_path=base1,
        scaling_param="times",
        group_by="alpha",
        out_fname="sweep1_errors.pkl"
    )
    
        # 1) Sweep 1: time‐scaling grouped by spreadingation
    base2 = os.path.join(script_dir, "second_parameter_sweep_data")
    precompute_for_base(
        base_path=base2,
        scaling_param="times",
        group_by="spreading",
        out_fname="sweep2_errors.pkl"
    )

    print("\nAll cached pickles (if data existed) have been written.")
