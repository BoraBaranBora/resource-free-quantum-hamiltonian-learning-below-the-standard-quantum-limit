#!/usr/bin/env python3
"""
run_selected.py

Toggle which data‐generation sweeps to run by commenting/uncommenting.
"""
import os
import subprocess
from tqdm import tqdm

# ────────────────────────────────────────────────────────
#   **Select which datasets to re-generate**:
# ────────────────────────────────────────────────────────
run_sweep1 = True      # Sweep perturbations (α=1.0, N=8)  → measurements=50
run_sweep2 = False     # Sweep α (perturb=50, N=8)          → measurements=25
run_sweep3 = False     # Nested: for each perturb, sweep α  → measurements=25
# ────────────────────────────────────────────────────────

# Path to your demo script
DEMOPATH = os.path.dirname(os.path.abspath(__file__))
DEMO    = os.path.join(DEMOPATH, "learn_hamiltonian.py")

# Shared fixed settings
MEASUREMENTS_SWEEP1 = 50   # for SWEEP 1 sweep
MEASUREMENTS_SWEEP2 = 25   # for SWEEP 2 (and SWEEP 3)
SHOTS             = 1
STEPS             = 8

# Sweep lists
PERTURBS = [1, 10, 25, 50, 100, 250, 500] # (Number of Spread Initial States)
ALPHAS   = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def run(cmd):
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def reproduce_data_SWEEP1(output_root):
    """Sweep over parameter combinations for SWEEP 1 (measurements=50)."""
    step_list = ",".join(str(i) for i in range(1, STEPS+1))
    for pert in tqdm(PERTURBS, desc="SWEEP 1 Data (measurements=50)"):
        outdir = os.path.join(output_root, f"SWEEP1_perturb_{pert}")
        os.makedirs(outdir, exist_ok=True)
        run([
            "python", DEMO,
            "--alphas",       "1.0",                     # fixed alpha
            "--perturbs",     str(pert),                 # sweep this spread‐state size
            "--measurements", str(MEASUREMENTS_SWEEP1),    # 50
            "--shots",        str(SHOTS),
            "--steps",        step_list,                 # sweep steps internally
            "--output-dir",   outdir
        ])

def reproduce_data_SWEEP2(output_root):
    """Sweep over parameter combinations for SWEEP 2 (measurements=25)."""
    step_list = ",".join(str(i) for i in range(1, STEPS+1))
    for alpha in tqdm(ALPHAS, desc="SWEEP 2 Data (measurements=25)"):
        outdir = os.path.join(output_root, f"SWEEP2_alpha_{alpha}")
        os.makedirs(outdir, exist_ok=True)
        run([
            "python", DEMO,
            "--alphas",       str(alpha),                # sweep this α
            "--perturbs",     "50",                      # fixed spread‐state size
            "--measurements", str(MEASUREMENTS_SWEEP2),    # 25
            "--shots",        str(SHOTS),
            "--steps",        step_list,                 # sweep steps internally
            "--output-dir",   outdir
        ])

def reproduce_data_SWEEP3(output_root):
    """Sweep over parameter combinations for SWEEP 3 (measurements=25)."""
    perturb_list = ",".join(str(p) for p in PERTURBS)
    for alpha in tqdm(ALPHAS, desc="SWEEP 3 Data (measurements=25)"):
        outdir = os.path.join(output_root, f"SWEEP3_alpha_{alpha}")
        os.makedirs(outdir, exist_ok=True)
        run([
            "python", DEMO,
            "--alphas",       str(alpha),                # sweep this α
            "--perturbs",     perturb_list,              # sweep perturb internally
            "--measurements", str(MEASUREMENTS_SWEEP2),    # 25
            "--shots",        str(SHOTS),
            "--steps",        str(STEPS),                 # fixed number of steps
            "--output-dir",   outdir
        ])

def main():
    root = "outputs"
    if run_sweep1:
        reproduce_data_SWEEP1(os.path.join(root, "sweep1"))
    if run_sweep2:
        reproduce_data_SWEEP2(os.path.join(root, "sweep2"))
    if run_sweep3:
        reproduce_data_SWEEP3(os.path.join(root, "sweep3"))

    print("Selected sweeps complete. Check the “outputs/” folder.")

if __name__ == "__main__":
    main()
