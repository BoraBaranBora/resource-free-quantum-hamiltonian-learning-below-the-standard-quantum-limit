#!/usr/bin/env python3
"""
run_selected.py

Toggle which data‐generation sweeps to run by commenting/uncommenting.
"""

import os
import subprocess

# ────────────────────────────────────────────────────────
#   **Select which plots to generate**:
# Comment out the ones you DON’T want.
# Uncomment the ones you DO want.
# ────────────────────────────────────────────────────────
# run_plot1 = True    # Sweep perturbations (α=1.0, N=8)
run_plot1 = True

run_plot2 = False    # Sweep α (perturb=50, N=8)

# run_plot3 = True  # Nested: for each perturb, sweep α (N=8)
run_plot3 = False
# ────────────────────────────────────────────────────────

# Path to your demo script
# Find the directory containing this script (i.e. the demo/ folder)
DEMOPATH = os.path.dirname(os.path.abspath(__file__))
# The learn_hamiltonian script lives right next to this in demo/
DEMO = os.path.join(DEMOPATH, "learn_hamiltonian.py")

# Shared fixed settings
MEASUREMENTS = 25
SHOTS        = 1
STEPS        = 8

# Sweep lists
PERTURBS = [1, 10, 25]#, 50, 100, 250, 500]
ALPHAS   = [0.2, 0.5, 0.8, 1.0]

def run(cmd):
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def sweep_plot1(output_root):
    for pert in PERTURBS:
        outdir = os.path.join(output_root, f"plot1_perturb_{pert}")
        os.makedirs(outdir, exist_ok=True)
        run([
            "python", DEMO,
            "--alphas",       "1.0",
            "--perturbs",     str(pert),
            "--measurements", str(MEASUREMENTS),
            "--shots",        str(SHOTS),
            "--steps",        str(STEPS),
            "--output-dir",   outdir
        ])

def sweep_plot2(output_root):
    for alpha in ALPHAS:
        outdir = os.path.join(output_root, f"plot2_alpha_{alpha}")
        os.makedirs(outdir, exist_ok=True)
        run([
            "python", DEMO,
            "--alphas",       str(alpha),
            "--perturbs",     "50",
            "--measurements", str(MEASUREMENTS),
            "--shots",        str(SHOTS),
            "--steps",        str(STEPS),
            "--output-dir",   outdir
        ])

def sweep_plot3(output_root):
    for pert in PERTURBS:
        outdir = os.path.join(output_root, f"plot3_perturb_{pert}")
        os.makedirs(outdir, exist_ok=True)
        alpha_list = ",".join(str(a) for a in ALPHAS)
        run([
            "python", DEMO,
            "--alphas",       alpha_list,
            "--perturbs",     str(pert),
            "--measurements", str(MEASUREMENTS),
            "--shots",        str(SHOTS),
            "--steps",        str(STEPS),
            "--output-dir",   outdir
        ])

def main():
    root = "outputs"
    if run_plot1:
        sweep_plot1(os.path.join(root, "plot1"))
    if run_plot2:
        sweep_plot2(os.path.join(root, "plot2"))
    if run_plot3:
        sweep_plot3(os.path.join(root, "plot3"))

    print("Selected sweeps complete. Check the “outputs/” folder.")

if __name__ == "__main__":
    main()
