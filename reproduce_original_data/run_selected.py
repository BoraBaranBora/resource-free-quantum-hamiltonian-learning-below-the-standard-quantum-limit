#!/usr/bin/env python3
"""
run_selected.py

Toggle which data‐generation sweeps to run by commenting/uncommenting.
"""
import os
import subprocess
from tqdm import tqdm

# ────────────────────────────────────────────────────────
#   **Select which plots to generate**:
# ────────────────────────────────────────────────────────
run_plot1 = True      # Sweep perturbations (α=1.0, N=8)
run_plot2 = False     # Sweep α (perturb=50, N=8)
run_plot3 = False     # Nested: for each perturb, sweep α (N=8)
# ────────────────────────────────────────────────────────

# Path to your demo script
DEMOPATH = os.path.dirname(os.path.abspath(__file__))
DEMO    = os.path.join(DEMOPATH, "learn_hamiltonian.py")

# Shared fixed settings
MEASUREMENTS = 25
SHOTS        = 1
STEPS        = 8

# Sweep lists
PERTURBS = [1, 10, 25, 50, 100, 250, 500]
ALPHAS   = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def run(cmd):
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def reproduce_data_figure1(output_root):
    """Sweep over parameter combinations for Figure 1."""
    step_list = ",".join(str(i) for i in range(1, STEPS+1))
    for pert in tqdm(PERTURBS, desc="Figure 1 Data: Recovery Error Scaling against total Experiment Time with increasing Spread State Ensemble"):
        outdir = os.path.join(output_root, f"figure1_perturb_{pert}")
        os.makedirs(outdir, exist_ok=True)
        run([
            "python", DEMO,
            "--alphas",       "1.0",                # fixed alpha
            "--perturbs",     str(pert),            # sweep this spread state ensemble size
            "--measurements", str(MEASUREMENTS),
            "--shots",        str(SHOTS),
            "--steps",        step_list,            # sweep steps internally
            "--output-dir",   outdir
        ])

def reproduce_data_figure2(output_root):
    """Sweep over parameter combinations for Figure 2."""
    step_list = ",".join(str(i) for i in range(1, STEPS+1))
    for alpha in tqdm(ALPHAS, desc="Figure 2 Data: Recovery Error Scaling against total Experiment Time with increasing Alpha Value"):
        outdir = os.path.join(output_root, f"figure2_alpha_{alpha}")
        os.makedirs(outdir, exist_ok=True)
        run([
            "python", DEMO,
            "--alphas",       str(alpha),           # sweep this
            "--perturbs",     "50",                 # fixed spread state ensemble size
            "--measurements", str(MEASUREMENTS),
            "--shots",        str(SHOTS),
            "--steps",        step_list,            # sweep steps internally
            "--output-dir",   outdir
        ])
        
def reproduce_data_figure3(output_root):
    """Sweep over parameter combinations for Figure 3."""
    perturb_list = ",".join(str(p) for p in PERTURBS)
    for alpha in tqdm(ALPHAS, desc="Figure 3 Data: Recovery Error Scaling against Spread Ensemble Size with increasing Alpha Value"):
        outdir = os.path.join(output_root, f"figure3_alpha_{alpha}")
        os.makedirs(outdir, exist_ok=True)
        run([
            "python", DEMO,
            "--alphas",       str(alpha),           # sweep this
            "--perturbs",     perturb_list,         # sweep steps internally
            "--measurements", str(MEASUREMENTS),
            "--shots",        str(SHOTS),
            "--steps",        str(STEPS),            # fixed time stamps
            "--output-dir",   outdir
        ])


def main():
    root = "outputs"
    if run_plot1:
        reproduce_data_figure1(os.path.join(root, "plot1"))
    if run_plot2:
        reproduce_data_figure2(os.path.join(root, "plot2"))
    if run_plot3:
        reproduce_data_figure3(os.path.join(root, "plot3"))

    print("Selected sweeps complete. Check the “outputs/” folder.")

if __name__ == "__main__":
    main()
