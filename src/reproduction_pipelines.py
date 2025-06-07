#!/usr/bin/env python3
"""
reproduction_pipelines.py

Contains three functions to re‐generate data sweeps:
  • reproduce_data_SWEEP1(base_folder, families)
  • reproduce_data_SWEEP2(base_folder, families)
  • reproduce_data_SWEEP3(base_folder, families)

Each “base_folder” argument is a path where the data for each sweep will be generated
directly into that folder (no additional subfolders).
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
ALPHAS     = [1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3]


def run(cmd):
    """Helper to print and execute a subprocess command using the same Python interpreter."""
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True)
    

def reproduce_data_SWEEP1(base_folder: str, families: str):
    """
    Sweep over parameter combinations for SWEEP 1 (measurements=50).
    Generates outputs directly in base_folder.
    """
    os.makedirs(base_folder, exist_ok=True)
    step_list = ",".join(str(i) for i in range(1, STEPS + 1))

    for spread in tqdm(SPREADINGS, desc="SWEEP 1 Data (measurements=50)"):
        run([
            sys.executable, DEMO,
            "--alphas",       "1.0",                      # fixed α
            "--spreadings",   str(spread),                # sweep this spreading
            "--measurements", str(MEASUREMENTS_SWEEP1),   # 50
            "--shots",        str(SHOTS),
            "--steps",        step_list,                  # sweep steps = 1…8
            "--families",     families,                   # now passed in
            "--output-dir",   base_folder
        ])


def reproduce_data_SWEEP2(base_folder: str, families: str):
    """
    Sweep over parameter combinations for SWEEP 2 (measurements=25).
    Generates outputs directly in base_folder.
    """
    os.makedirs(base_folder, exist_ok=True)
    step_list = ",".join(str(i) for i in range(1, STEPS + 1))

    for alpha in tqdm(ALPHAS, desc="SWEEP 2 Data (measurements=25)"):
        run([
            sys.executable, DEMO,
            "--alphas",       str(alpha),                 # sweep this α
            "--spreadings",   "50",                       # fixed spreading = 50
            "--measurements", str(MEASUREMENTS_SWEEP2),   # 25
            "--shots",        str(SHOTS),
            "--steps",        step_list,                  # sweep steps = 1…8
            "--families",     families,                   # now passed in
            "--output-dir",   base_folder
        ])


def reproduce_data_SWEEP3(base_folder: str, families: str):
    """
    Sweep over parameter combinations for SWEEP 3 (measurements=25).
    Generates outputs directly in base_folder.
    """
    os.makedirs(base_folder, exist_ok=True)
    spreading_list = ",".join(str(p) for p in SPREADINGS)

    for alpha in tqdm(ALPHAS, desc="SWEEP 3 Data (measurements=25)"):
        run([
            sys.executable, DEMO,
            "--alphas",       str(alpha),                 # sweep this α
            "--spreadings",   spreading_list,             # sweep all SPREADINGS internally
            "--measurements", str(MEASUREMENTS_SWEEP2),   # 25
            "--shots",        str(SHOTS),
            "--steps",        str(STEPS),                  # exactly 8 steps
            "--families",     families,                   # now passed in
            "--output-dir",   base_folder
        ])
