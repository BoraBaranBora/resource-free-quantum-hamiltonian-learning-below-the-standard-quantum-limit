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
MEASUREMENTS_SWEEP1 = 100   # for SWEEP 1
MEASUREMENTS_SWEEP2 = 50  # for SWEEP 2 and SWEEP 3
SHOTS               = 1
STEPS               = 8

STEPS_SWEEP1        = 15

# Sweep lists
SPREADINGS_SWEEP1 = [100]#[1, 2, 4, 8, 16, 32, 64, 128]
SPREADINGS_SWEEP3 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]##[1, 10, 25, 50, 100, 250, 500]# #  
ALPHAS     = [1.0]#[1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3]

QUBIT_SIZES = [2,3,4,5,6,7]
SPREADING_SWEEP4 = 100

def run(cmd):
    """Helper to print and execute a subprocess command using the same Python interpreter."""
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True)
    

def run(cmd):
    print(f"Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print("Subprocess stdout:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print("=== Subprocess failed ===")
        print("Command:", e.cmd)
        print("Return code:", e.returncode)
        if e.stdout:
            print("Stdout:\n", e.stdout)
        if e.stderr:
            print("Stderr:\n", e.stderr)
        # Optionally write to log file for easier inspection:
        with open("last_failed_subprocess.log", "w") as f:
            f.write("STDOUT:\n" + (e.stdout or "") + "\n\n")
            f.write("STDERR:\n" + (e.stderr or ""))
        # Re-raise so your outer script still sees the failure:
        raise


def run(cmd):
    print(">>>", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        # On failure, rerun quickly with capture to get full stderr?
        # WARNING: rerunning re-executes the expensive job—usually not acceptable.
        # Instead, rely on logs written by the child script.
        print("Subprocess failed; check log files under the run folder for traceback.")
        raise
    

def reproduce_data_SWEEP1(base_folder: str, families: str):
    """
    Sweep over parameter combinations for SWEEP 1 .
    Generates outputs directly in base_folder.
    """
    os.makedirs(base_folder, exist_ok=True)
    step_list = ",".join(str(i) for i in range(1, STEPS_SWEEP1 + 1))

    for spread in tqdm(SPREADINGS_SWEEP1, desc=f"SWEEP 1 Data (measurements={MEASUREMENTS_SWEEP1})"):
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
    Sweep over parameter combinations for SWEEP 2 .
    Generates outputs directly in base_folder.
    """
    os.makedirs(base_folder, exist_ok=True)
    step_list = ",".join(str(i) for i in range(1, STEPS + 1))

    for alpha in tqdm(ALPHAS, desc=f"SWEEP 2 Data (measurements={MEASUREMENTS_SWEEP2})"):
        run([
            sys.executable, DEMO,
            "--alphas",       str(alpha),                 # sweep this α
            "--spreadings",   "32",                       # fixed spreading = 50
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
    spreading_list = ",".join(str(p) for p in SPREADINGS_SWEEP3)

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


def reproduce_data_SWEEP4(base_folder: str, families: str):
    """
    Sweeps over qubit sizes for sweep 4.
    Each run's data is stored in its own timestamped subfolder under base_folder.
    """
    os.makedirs(base_folder, exist_ok=True)
    step_list = ",".join(str(i) for i in range(1, STEPS_SWEEP1 + 1))

    for nq in tqdm(QUBIT_SIZES, desc="Qubit Size Sweep"):
        run([
            sys.executable, DEMO,
            "--alphas",       "1.0",                      # fixed α
            "--spreadings",   str(SPREADING_SWEEP4),             # fixed spreading value
            "--measurements", str(MEASUREMENTS_SWEEP1),   # fixed measurements
            "--shots",        str(SHOTS),
            "--steps",        step_list,                  # steps = 1…15
            "--families",     families,
            "--num-qubits",   str(nq),                    # sweep this
            "--output-dir",   base_folder
        ])
