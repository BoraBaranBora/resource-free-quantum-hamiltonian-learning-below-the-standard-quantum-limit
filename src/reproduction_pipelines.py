#!/usr/bin/env python3
"""
reproduction_pipelines.py

Contains three functions to re‐generate data sweeps:
  • reproduce_data_SWEEP1(base_folder, families)
  • reproduce_data_SWEEP2(base_folder, families)
  • reproduce_data_SWEEP3(base_folder, families)

Each “base_folder” argument is a path where subfolders named run_SWEEP1_…,
run_SWEEP2_…, or run_SWEEP3_… will be created.  The `families` string is 
passed through to learn_hamiltonian.py.
"""

import os
import subprocess
from tqdm import tqdm
import sys

# Path to your demo script
DEMOPATH = os.path.dirname(os.path.abspath(__file__))
DEMO    = os.path.join(DEMOPATH, "learn_hamiltonian.py")

# Shared fixed settings
MEASUREMENTS_SWEEP1 = 50   # for SWEEP 1
MEASUREMENTS_SWEEP2 = 25   # for SWEEP 2 and SWEEP 3
SHOTS               = 1
STEPS               = 8

# Sweep lists
SPREADINGS = [1, 10, 25, 50, 100, 250, 500]
ALPHAS     = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def run(cmd):
    """Helper to print and execute a subprocess command using the same Python interpreter."""
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True)
    
def run(cmd):
    """Helper to print and execute a subprocess command using the same Python interpreter."""
    print(">>>", " ".join(cmd))
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    # Print the subprocess's stdout
    if result.stdout:
        print(result.stdout, end="")
    # If it failed, print stderr and raise
    if result.returncode != 0:
        print(f"ERROR: subprocess exited with code {result.returncode}", file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        result.check_returncode()


def reproduce_data_SWEEP1(base_folder: str, families: str):
    """
    Sweep over parameter combinations for SWEEP 1 (measurements=50).
    Creates subfolders run_SWEEP1_spreading_<p>/ under base_folder.
    """
    os.makedirs(base_folder, exist_ok=True)
    step_list = ",".join(str(i) for i in range(1, STEPS + 1))

    for spread in tqdm(SPREADINGS, desc="SWEEP 1 Data (measurements=50)"):
        run_dir_name = f"run_SWEEP1_spreading_{spread}"
        outdir = os.path.join(base_folder, run_dir_name)
        os.makedirs(outdir, exist_ok=True)

        run([
            sys.executable, DEMO,
            "--alphas",       "1.0",                      # fixed α
            "--spreadings",   str(spread),                # sweep this spreading
            "--measurements", str(MEASUREMENTS_SWEEP1),   # 50
            "--shots",        str(SHOTS),
            "--steps",        step_list,                  # sweep steps = 1…8
            "--families",     families,                   # now passed in
            "--output-dir",   outdir
        ])


def reproduce_data_SWEEP2(base_folder: str, families: str):
    """
    Sweep over parameter combinations for SWEEP 2 (measurements=25).
    Creates subfolders run_SWEEP2_alpha_<α>/ under base_folder.
    """
    os.makedirs(base_folder, exist_ok=True)
    step_list = ",".join(str(i) for i in range(1, STEPS + 1))

    for alpha in tqdm(ALPHAS, desc="SWEEP 2 Data (measurements=25)"):
        run_dir_name = f"run_SWEEP2_alpha_{alpha}"
        outdir = os.path.join(base_folder, run_dir_name)
        os.makedirs(outdir, exist_ok=True)

        run([
            sys.executable, DEMO,
            "--alphas",       str(alpha),                 # sweep this α
            "--spreadings",   "50",                       # fixed spreading = 50
            "--measurements", str(MEASUREMENTS_SWEEP2),   # 25
            "--shots",        str(SHOTS),
            "--steps",        step_list,                  # sweep steps = 1…8
            "--families",     families,                   # now passed in
            "--output-dir",   outdir
        ])


def reproduce_data_SWEEP3(base_folder: str, families: str):
    """
    Sweep over parameter combinations for SWEEP 3 (measurements=25).
    Creates subfolders run_SWEEP3_alpha_<α>/ under base_folder.
    """
    os.makedirs(base_folder, exist_ok=True)
    spreading_list = ",".join(str(p) for p in SPREADINGS)

    for alpha in tqdm(ALPHAS, desc="SWEEP 3 Data (measurements=25)"):
        run_dir_name = f"run_SWEEP3_alpha_{alpha}"
        outdir = os.path.join(base_folder, run_dir_name)
        os.makedirs(outdir, exist_ok=True)

        run([
            sys.executable, DEMO,
            "--alphas",       str(alpha),                 # sweep this α
            "--spreadings",   spreading_list,             # sweep all SPREADINGS internally
            "--measurements", str(MEASUREMENTS_SWEEP2),   # 25
            "--shots",        str(SHOTS),
            "--steps",        str(STEPS),                  # exactly 8 steps
            "--families",     families,                   # now passed in
            "--output-dir",   outdir
        ])
