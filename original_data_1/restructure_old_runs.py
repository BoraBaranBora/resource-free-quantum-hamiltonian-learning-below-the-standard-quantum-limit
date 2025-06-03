#!/usr/bin/env python3
"""
restructure_old_runs.py

Traverse *all* run_<timestamp>/ folders inside original_data/run_directory/,
and for each one, rebuild a “new‐style” structure:

    restructured_run_directory/
      run_<timestamp>/
        alpha_<α>_perturb_<P>_measurements_<M>_shots_<S>_steps_<N>/
          config.json
          hamiltonians.json
          embedding_<codename>.pth
          embedding_<codename>_loss.json
          …

This way, your old results (which used “times_N/” subfolders) get transformed
to exactly the directory‐and‐file structure that the new `learn_hamiltonian.py`
and `plotting_utils.py` expect. No retraining is necessary.

Usage: from within `original_data/`, run:
    python restructure_old_runs.py
"""

import os
import json
import shutil
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
#  1) We assume this script sits in:
#       …/original_data/
#
#    and that inside that folder there is a subfolder:
#       run_directory/
#
#    containing multiple run_<timestamp>/ folders.  We will build a parallel
#    directory called restructured_run_directory/ where we recreate the “combo” subfolders.
# ──────────────────────────────────────────────────────────────────────────────

THIS_DIR        = os.path.dirname(os.path.abspath(__file__))
OLD_BASE_DIR    = os.path.join(THIS_DIR, "run_directory")
NEW_BASE_PARENT = os.path.join(THIS_DIR, "restructured_run_directory")

if not os.path.isdir(OLD_BASE_DIR):
    raise FileNotFoundError(f"Expected to find 'run_directory/' here:\n\t{OLD_BASE_DIR}\nBut it does not exist.")

# Create the top‐level “restructured_run_directory/” if it doesn’t already exist
os.makedirs(NEW_BASE_PARENT, exist_ok=True)
print(f"Old‐runs location:       {OLD_BASE_DIR}")
print(f"New‐runs will be written under:\n    {NEW_BASE_PARENT}\n")

# ──────────────────────────────────────────────────────────────────────────────
#  Iterate over each run_<timestamp>/ folder in OLD_BASE_DIR
# ──────────────────────────────────────────────────────────────────────────────
for run_folder in sorted(os.listdir(OLD_BASE_DIR)):
    if not run_folder.startswith("run_"):
        continue

    old_run_path = os.path.join(OLD_BASE_DIR, run_folder)
    if not os.path.isdir(old_run_path):
        continue

    # Prepare the new root for this run_<timestamp>:
    new_run_root = os.path.join(NEW_BASE_PARENT, run_folder)
    os.makedirs(new_run_root, exist_ok=True)
    print(f"=== Restructuring: {run_folder} → {new_run_root} ===")

    # Each old_run_path contains subfolders named times_<N>/
    for times_folder in sorted(os.listdir(old_run_path)):
        if not times_folder.startswith("times_"):
            continue

        old_times_path = os.path.join(old_run_path, times_folder)
        if not os.path.isdir(old_times_path):
            continue

        # ────────────────────────────────────────────────────────────────────
        #  A) Load the old JSON files from <old_times_path>:
        #     1) training_meta_parameters.json
        #     2) hamiltonian_meta_parameters.json
        #     3) hamiltonians_list.json
        # ────────────────────────────────────────────────────────────────────
        train_meta_path = os.path.join(old_times_path, "training_meta_parameters.json")
        hamil_meta_path = os.path.join(old_times_path, "hamiltonian_meta_parameters.json")
        hlist_old_path  = os.path.join(old_times_path, "hamiltonians_list.json")

        if not (os.path.isfile(train_meta_path) and
                os.path.isfile(hamil_meta_path) and
                os.path.isfile(hlist_old_path)):
            print(f"  [WARN] Skipping {old_times_path}: missing one of the JSON files.")
            continue

        with open(train_meta_path, "r") as f:
            old_train = json.load(f)
        with open(hamil_meta_path, "r") as f:
            old_hamil = json.load(f)
        with open(hlist_old_path, "r") as f:
            old_hamil_list = json.load(f)

        # ────────────────────────────────────────────────────────────────────
        #  B) Recover all “combo” parameters from old_train & old_hamil:
        #     - α, perturb, measurements, shots, steps, delta_t, times, etc.
        # ────────────────────────────────────────────────────────────────────

        # old_train["times"] is like [0.0, δt*(1**α), δt*(2**α), …, δt*(N**α)]
        old_times = old_train.get("times", [])
        N = len(old_times) - 1
        if N <= 0:
            print(f"  [WARN] times list {old_times} has no valid steps.  Skipping.")
            continue

        # δt = old_times[1], because old_times[1] = δt * (1**α)
        delta_t = float(old_times[1]) if len(old_times) > 1 else 1.0

        # If N ≥ 2, we can recover α from old_times[2] = δt * (2**α) ⇒ α = log(old_times[2]/δt)/log(2).
        if len(old_times) >= 3 and old_times[2] > 0:
            alpha = float(np.log(old_times[2] / delta_t) / np.log(2))
        else:
            # If N == 1, we cannot recover α uniquely; default to α = 1.0
            alpha = 1.0

        # Extract other fields from old_train:
        epochs        = old_train.get("epochs", 1000)
        tolerance     = old_train.get("tolerance", 1e-4)
        perturb       = old_train.get("perturbations", 50)
        shots         = old_train.get("shots", 1)
        measurements  = old_train.get("num_measurements", 25)
        hidden_layers = old_train.get("hidden_layers", [200, 200, 200])
        activation_fn = old_train.get("activation_function", "Tanh")
        nn_seed       = old_train.get("nn_seed", 99901)

        # Extract from old_hamil:
        num_qubits          = old_hamil.get("num_qubits", 5)
        per_family          = old_hamil.get("num_hamiltonians", 10)
        families            = old_hamil.get("families", ["Heisenberg"])
        coupling_t          = old_hamil.get("coupling_type", "anisotropic_normal")
        h_field_t           = old_hamil.get("h_field_type", "random")
        include_transverse  = old_hamil.get("include_transverse_field", True)
        include_higher      = old_hamil.get("include_higher_order", 0)

        # “steps” is simply N = len(old_times) - 1
        steps = N

        # The “times” list remains exactly old_times
        times_list = old_times[:]

        # ────────────────────────────────────────────────────────────────────
        #  C) Build the “combo” folder name exactly as new code does:
        #
        #     alpha_<α>_perturb_<P>_measurements_<M>_shots_<S>_steps_<N>/
        # ────────────────────────────────────────────────────────────────────
        combo_name = (
            f"alpha_{alpha:.3f}_perturb_{perturb}"
            f"_measurements_{measurements}"
            f"_shots_{shots}_steps_{steps}"
        )
        new_combo_dir = os.path.join(new_run_root, combo_name)
        os.makedirs(new_combo_dir, exist_ok=True)

        print(f"  • Restructuring '{times_folder}'  →  '{combo_name}'")

        # ────────────────────────────────────────────────────────────────────
        #  D) Create the new config.json (as the “new” learn_hamiltonian.py expects):
        # ────────────────────────────────────────────────────────────────────
        cfg = {
            "alpha":                alpha,
            "perturb":              perturb,
            "measurements":         measurements,
            "shots":                shots,
            "steps":                steps,

            "num_qubits":           num_qubits,
            "per_family":           per_family,
            "epochs":               epochs,
            "window":               old_train.get("batch_size", 10),
            "tolerance":            tolerance,
            "delta_t":              delta_t,

            "families":             families,
            "coupling_type":        coupling_t,
            "h_field_type":         h_field_t,
            "include_transverse":   include_transverse,
            "include_higher_order": include_higher,

            "hidden_layers":        hidden_layers,
            # strip off any “torch.nn.modules.activation.” prefix if present:
            "activation":           activation_fn.replace("torch.nn.modules.activation.", ""),
            "nn_seed":              nn_seed,
            "device":               "cpu",   # assume CPU (adjust if needed)

            # Preserve the entire “times” array
            "times":                times_list
        }
        config_path = os.path.join(new_combo_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=4)

        # ────────────────────────────────────────────────────────────────────
        #  E) Transform & write the new ham_list as “hamiltonians.json”:
        #     (Rename "parameters" → "params",  "codified_name" → "name")
        # ────────────────────────────────────────────────────────────────────
        transformed_list = []
        for old_entry in old_hamil_list:
            # old_entry keys: "family", "index", "parameters", "codified_name"
            new_entry = {
                "family": old_entry["family"],
                "index":  old_entry["index"],
                "params": old_entry["parameters"],       # rename
                "name":   old_entry["codified_name"]      # rename
            }
            transformed_list.append(new_entry)

        new_hamil_list_path = os.path.join(new_combo_dir, "hamiltonians.json")
        with open(new_hamil_list_path, "w") as f:
            json.dump(transformed_list, f, indent=4)

        # ────────────────────────────────────────────────────────────────────
        #  F) Copy each predictor_*.pth and loss_history_*.json:
        #     - Rename “predictor_” → “embedding_”
        #     - Rename “loss_history_<codename>.json” → “embedding_<codename>_loss.json”
        # ────────────────────────────────────────────────────────────────────
        for fname in os.listdir(old_times_path):
            full_old = os.path.join(old_times_path, fname)

            # 1) predictor_*.pth → embedding_*.pth
            if fname.startswith("predictor_") and fname.endswith(".pth"):
                # Extract the codename part after “predictor_” but before “.pth”
                codename_full = fname[len("predictor_") : -len(".pth")]
                new_pred_name = f"embedding_{codename_full}.pth"
                dst = os.path.join(new_combo_dir, new_pred_name)
                shutil.copyfile(full_old, dst)

            # 2) loss_history_<codename>.json → embedding_<codename>_loss.json
            #    The old loss file always has format: loss_history_<codename>.json
            elif fname.startswith("loss_history_") and fname.endswith(".json"):
                # Extract codename after “loss_history_” but before “.json”
                codename_full = fname[len("loss_history_") : -len(".json")]
                new_loss_name = f"embedding_{codename_full}_loss.json"
                dst = os.path.join(new_combo_dir, new_loss_name)
                shutil.copyfile(full_old, dst)

        print(f"    → Wrote config.json, hamiltonians.json, embedding_*.pth & embedding_*_loss.json\n")

print("\n  Finished restructuring ALL runs.\n")
