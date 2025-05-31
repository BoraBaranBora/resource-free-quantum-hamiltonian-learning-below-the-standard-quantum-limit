#!/usr/bin/env python3
"""
learn_hamiltonian.py

Demo: recover Hamiltonians while sweeping arbitrary hyperparameters.
Any combination of alpha, perturbations, measurements, shots, steps can be provided.
"""
import os, sys

# __file__ might be "…/demo/learn_hamiltonian.py"
# DEMO_DIR = directory containing this script, i.e. .../demo/
DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
# SRC_DIR = sibling "src" folder under the repo root
SRC_PATH = os.path.abspath(os.path.join(DEMO_DIR, "..", "src"))
sys.path.insert(0, SRC_PATH)

import json, gc, argparse
from itertools import product
from datetime import datetime
from tqdm import tqdm, trange

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset

from datagen import DataGen
from hamiltonian_generator import generate_hamiltonian, generate_hamiltonian_parameters
from predictor import Predictor
from loss import Loss
from utils import convert_to_serializable, generate_advanced_codified_name

def get_max_batch_size(num_qubits, gpu_memory_gb=24, memory_overhead_gb=2):
    hilbert_dim = 2 ** num_qubits
    density_mb = (hilbert_dim**2)*16/(1024**2)
    avail_mb   = (gpu_memory_gb - memory_overhead_gb)*1024
    per_batch  = 3*density_mb
    return int((avail_mb//per_batch)*0.95//50*50)

def generate_times(alpha, N, delta_t):
    return [0.0] + [delta_t*(k**alpha) for k in range(1, N+1)]

def save_json(obj, path):
    clean = convert_to_serializable(obj)
    with open(path, 'w') as f:
        json.dump(clean, f, indent=4)

def run_single_run(run_root, params, fixed):
    """
    params: dict with keys alpha, perturb, measurements, shots, steps
    fixed:  dict with the other fixed settings
    """
    # Build a unique subdir name
    name_parts = [f"{k}_{v}" for k,v in params.items()]
    subdir = os.path.join(run_root, "_".join(name_parts))
    os.makedirs(subdir, exist_ok=True)

    # Build a JSON-safe run config
    cfg = {
        **params,
        "num_qubits":    fixed["num_qubits"],
        "per_family":    fixed["per_family"],
        "epochs":        fixed["epochs"],
        "window":        fixed["window"],
        "tolerance":     fixed["tolerance"],
        "delta_t":       fixed["delta_t"],
        "families":      fixed["families"],
        "coupling_type": fixed["coupling_type"],
        "h_field_type":  fixed["h_field_type"],
        "hidden_layers": fixed["hidden_layers"],
        "activation":    fixed["ACTIVATION"].__name__,
        "nn_seed":       fixed["nn_seed"],
        "device":        str(fixed["device"]),
    }
    save_json(cfg, os.path.join(subdir, "config.json"))

    # Compute the (fixed) time stamps
    times_all     = generate_times(params["alpha"], params["steps"], fixed["delta_t"])
    current_times = times_all[: params["steps"]]

    # Build & save Hamiltonian list
    ham_list = []
    for fam in fixed["families"]:
        for idx in range(fixed["per_family"]):
            ham_p = generate_hamiltonian_parameters(
                family=fam,
                num_qubits=fixed["num_qubits"],
                coupling_type=fixed["coupling_type"],
                h_field_type=fixed["h_field_type"],
            )
            name = generate_advanced_codified_name(fam, idx, ham_p)
            ham_list.append({
                "family": fam,
                "index": idx,
                "params": convert_to_serializable(ham_p),
                "name": name
            })
    save_json(ham_list, os.path.join(subdir, "hamiltonians.json"))

    # Loop over each Hamiltonian with its own progress bar
    for info in tqdm(ham_list, desc="Hamiltonians", leave=False):
        H = generate_hamiltonian(
            family=info["family"],
            num_qubits=fixed["num_qubits"],
            **info["params"]
        )

        dg = DataGen(
            num_qubits=fixed["num_qubits"],
            times=current_times,
            num_measurements=params["measurements"],
            shots=params["shots"],
            perturbations=params["perturb"],
            initial_state_indices=[0],
            seed=fixed["nn_seed"],
            hamiltonian=H,
        )
        targets, times_t, basis, init = dg.generate_dataset()
        ds     = TensorDataset(targets, times_t, basis, init)
        loader = DataLoader(ds,
                            batch_size=get_max_batch_size(fixed["num_qubits"]),
                            shuffle=True)

        # Build model+loss+optimizer
        torch.manual_seed(fixed["nn_seed"])
        d = 2 ** fixed["num_qubits"]
        tri_len = d * (d + 1) // 2
        predictor = Predictor(
            input_size=tri_len,
            output_size=tri_len,
            hidden_layers=fixed["hidden_layers"],
            activation_fn=fixed["ACTIVATION"],
            ignore_input=True
        ).to(fixed["device"])
        criterion = Loss(num_qubits=fixed["num_qubits"])
        optimizer = optim.AdamW(predictor.parameters())

        # Training loop with progress bar
        loss_hist = []
        for epoch in trange(fixed["epochs"], desc="  Epochs", leave=False):
            total = 0.0
            predictor.train()
            optimizer.zero_grad()
            for xb, tb, bb, ib in loader:
                xb, tb, bb, ib = (t.to(fixed["device"]) for t in (xb,tb,bb,ib))
                loss = criterion(predictor, tb, ib, xb, bb)
                loss.backward()
                total += loss.item()
            optimizer.step()
            avg = total/len(loader)
            loss_hist.append(avg)
            # Early stopping
            if epoch >= fixed["window"] + 5:
                recent = np.mean(loss_hist[-fixed["window"]:])
                prev   = np.mean(loss_hist[-(fixed["window"]+5):-5])
                if abs(recent - prev) < fixed["tolerance"]:
                    break

        # Save outputs
        #base = os.path.join(subdir, info["name"])
        #torch.save(predictor.state_dict(), base + ".pth")
        #save_json({"loss_history": loss_hist}, base + "_loss.json")
        
        # Save outputs under “embedding_<codename>.pth” and “embedding_<codename>_loss.json”
        codename = info["name"]  # e.g. "hamiltonian_heisenberg_000_a8c99b86"
        model_filename = f"embedding_{codename}.pth"
        loss_filename  = f"embedding_{codename}_loss.json"

        torch.save(predictor.state_dict(), os.path.join(subdir, model_filename))
        save_json({"loss_history": loss_hist}, os.path.join(subdir, loss_filename))


        # Cleanup
        del ds, predictor, xb, tb, bb, ib
        torch.cuda.empty_cache()
        gc.collect()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    # These flags can be comma-lists
    p.add_argument("--alphas",       required=True,
                   help="Comma-separated α values, e.g. 0.8,1.0,1.2")
    p.add_argument("--perturbs",     required=True,
                   help="Comma-separated perturb levels, e.g. 10,50,100")
    p.add_argument("--measurements", required=True,
                   help="Comma-separated #measurements, e.g. 25,50")
    p.add_argument("--shots",        required=True,
                   help="Comma-separated shots, e.g. 1,5")
    p.add_argument("--steps",        required=True,
                   help="Comma-separated N (time-step counts), e.g. 5,8,12")

    # Fixed settings
    p.add_argument("--num-qubits",   type=int, default=5,   help="Number of qubits")
    p.add_argument("--per-family",   type=int, default=10,  help="Hams per family")
    p.add_argument("--epochs",       type=int, default=500,help="Training epochs")
    p.add_argument("--window",       type=int, default=10,  help="Early-stop window")
    p.add_argument("--tolerance",    type=float,default=1e-4,help="Convergence tol.")
    p.add_argument("--delta-t",      type=float,default=0.1, help="Δt for time steps")
    p.add_argument("--output-dir",   type=str, required=True,
                   help="Where to dump all outputs")
    args = p.parse_args()

    # Expand comma-lists into Python lists
    sweep = {
        "alphas":       [float(x) for x in args.alphas.split(",")],
        "perturbs":     [int(x)   for x in args.perturbs.split(",")],
        "measurements": [int(x)   for x in args.measurements.split(",")],
        "shots":        [int(x)   for x in args.shots.split(",")],
        "steps":        [int(x)   for x in args.steps.split(",")],
    }

    # Fixed run settings
    fixed = {
        "num_qubits":    args.num_qubits,
        "per_family":    args.per_family,
        "epochs":        args.epochs,
        "window":        args.window,
        "tolerance":     args.tolerance,
        "delta_t":       args.delta_t,
        "families":      ["Heisenberg"],
        "coupling_type": "anisotropic_normal",
        "h_field_type":  "random",
        "include_transverse": True,
        "include_higher_order": 0,
        "hidden_layers": [200,200,200],
        "ACTIVATION":    nn.Tanh,
        "nn_seed":       99901,
        "device":        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    print(f"Using device: {fixed['device']}")

    # Prepare top-level run folder
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(args.output_dir, f"run_{ts}")
    os.makedirs(run_root, exist_ok=True)

    # Iterate over every combination with a progress bar
    combos = list(product(
        sweep["alphas"], sweep["perturbs"],
        sweep["measurements"], sweep["shots"],
        sweep["steps"]
    ))
    for (alpha, perturb, meas, shot, step) in tqdm(combos, desc="Inner Sweep (Fig1&2: time stamps; Fig3: spread state ensemble size)"):
        params = {
          "alpha":        alpha,
          "perturb":      perturb,
          "measurements": meas,
          "shots":        shot,
          "steps":        step
        }
        run_single_run(run_root, params, fixed)

    print("All sweeps completed.")

if __name__ == "__main__":
    main()
