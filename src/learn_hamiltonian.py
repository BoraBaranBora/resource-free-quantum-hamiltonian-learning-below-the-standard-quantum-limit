#!/usr/bin/env python3
"""
learn_hamiltonian.py

Demo: recover Hamiltonians while sweeping arbitrary hyperparameters.
Any combination of alpha, spreadings, measurements, shots, steps, and families can be provided.
"""
import os
import sys
import json
import gc
import argparse
from itertools import product
from datetime import datetime
from tqdm import tqdm

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset

from datagen import DataGen
from hamiltonian_generator import generate_hamiltonian, generate_hamiltonian_parameters
from predictor import Predictor
from loss import Loss
from utils import convert_to_serializable, generate_advanced_codified_name




def get_max_batch_size_np(nq, gpu_mem_gb, overhead_gb=2, round_to=50, safety=0.95):
    """
    Estimate max batch size for density-matrix sims of 2**nq dimension,
    given GPU memory in GB, reserved overhead, safety factor, and rounding.
    
    Parameters:
        nq (int): number of qubits.
        gpu_mem_gb (float): total GPU memory in GB.
        overhead_gb (float): GB to reserve (default 2).
        round_to (int or None): round down to nearest multiple (default 50). Use <=1 or None to skip.
        safety (float): safety factor before rounding (default 0.95).
    Returns:
        int: max batch size (multiple of round_to), or 0 if none fits.
    """
    avail = (gpu_mem_gb - overhead_gb) * 1024**3
    if avail <= 0:
        return 0
    # log per-sample bytes = ln(48) + nq * ln(4)
    log_ps = np.log(48.0) + nq * np.log(4.0)
    if log_ps > np.log(avail + 1e-9):
        return 0
    bs = int(np.exp(np.log(avail) - log_ps) * safety)
    if round_to and round_to > 1:
        bs = (bs // round_to) * round_to
    return bs

def generate_times(alpha, N, delta_t):
    return [delta_t * (k**alpha) for k in range(1, N+1)]


def save_json(obj, path):
    clean = convert_to_serializable(obj)
    with open(path, 'w') as f:
        json.dump(clean, f, indent=4)


def run_single_run(run_root, params, fixed):
    """
    Creates one combo‐directory under run_root, saves config.json and hamiltonians.json,
    and returns (subdir_path, ham_list, current_times) so that training can occur outside.
    """
    # Build a unique subdir name
    name_parts = [f"{k}_{v}" for k, v in params.items()]
    subdir = os.path.join(run_root, "_".join(name_parts))
    os.makedirs(subdir, exist_ok=True)

    # Compute the (fixed) time stamps
    times_all     = generate_times(params["alpha"], params["steps"], fixed["delta_t"])
    current_times = times_all[: params["steps"]]

    # Build and save JSON‐safe run config
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
        "times":         current_times,
    }
    save_json(cfg, os.path.join(subdir, "config.json"))


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

    return subdir, ham_list, current_times


def main():
    p = argparse.ArgumentParser(description=__doc__)
    # These flags can be comma‐lists
    p.add_argument("--alphas",       required=True,
                   help="Comma‐separated α values, e.g. 0.8,1.0,1.2")
    p.add_argument("--spreadings",   required=True,
                   help="Comma‐separated spreading levels, e.g. 10,50,100")
    p.add_argument("--measurements", required=True,
                   help="Comma‐separated #measurements, e.g. 25,50")
    p.add_argument("--shots",        required=True,
                   help="Comma‐separated shots, e.g. 1,5")
    p.add_argument("--steps",        required=True,
                   help="Comma‐separated N (time‐step counts), e.g. 5,8,12")
    p.add_argument("--families",     required=True,
                   help="Comma‐separated Hamiltonian families, e.g. Heisenberg,XXZ,TFIM")

    # Fixed settings
    p.add_argument("--num-qubits",   type=int, default=5,   help="Number of qubits")
    p.add_argument("--per-family",   type=int, default=10,  help="Hamiltonians per family")
    p.add_argument("--epochs",       type=int, default=1000, help="Training epochs")
    p.add_argument("--window",       type=int, default=10,  help="Early‐stop window")
    p.add_argument("--tolerance",    type=float, default=5e-4, help="Convergence tolerance")
    p.add_argument("--delta-t",      type=float, default=0.025,  help="Δt for time steps")
    p.add_argument("--output-dir",   type=str, required=True,
                   help="Where to dump all outputs")
    args = p.parse_args()

    # Expand comma‐lists into Python lists
    sweep = {
        "alphas":       [float(x) for x in args.alphas.split(",")],
        "spreadings":   [int(x)   for x in args.spreadings.split(",")],
        "measurements": [int(x)   for x in args.measurements.split(",")],
        "shots":        [int(x)   for x in args.shots.split(",")],
        "steps":        [int(x)   for x in args.steps.split(",")],
    }
    # Parse families as list of strings (stripping whitespace)
    fixed_families = [fam.strip() for fam in args.families.split(",") if fam.strip()]

    # Fixed run settings
    fixed = {
        "num_qubits":          args.num_qubits,
        "per_family":          args.per_family,
        "epochs":              args.epochs,
        "window":              args.window,
        "tolerance":           args.tolerance,
        "delta_t":             args.delta_t,
        "families":            fixed_families,
        "coupling_type":       "anisotropic_normal",
        "h_field_type":        "random",
        "include_transverse":  True,
        "hidden_layers":       [200, 200, 200],
        "ACTIVATION":          nn.Tanh,
        "nn_seed":             99901,
        "device":              torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    print(f"Using device: {fixed['device']}")
    print(f"Sweeping families: {fixed['families']}")

    # Prepare top‐level run folder
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(args.output_dir, f"run_{ts}")
    os.makedirs(run_root, exist_ok=True)

    # Build all parameter combinations
    combos = list(product(
        sweep["alphas"], sweep["spreadings"],
        sweep["measurements"], sweep["shots"],
        sweep["steps"]
    ))

    overall = tqdm(combos, desc="Scaling Sweep")
    for (alpha, spreading, meas, shot, step) in overall:
        params = {
          "alpha":       alpha,
          "spreading":   spreading,
          "measurements": meas,
          "shots":       shot,
          "steps":       step
        }

        # Create subdir and ham_list
        subdir, ham_list, current_times = run_single_run(run_root, params, fixed)
        total_hams = len(ham_list)

        # Train each Hamiltonian in this combo
        for idx, info in enumerate(ham_list, start=1):
            overall.set_postfix({
                "α": f"{alpha:.3f}",
                "spread": spreading,
                "ham": f"{idx}/{total_hams}"
            })

            H = generate_hamiltonian(
                family=info["family"],
                num_qubits=fixed["num_qubits"],
                device=fixed["device"],        
                **info["params"]
            )

            dg = DataGen(
                num_qubits=fixed["num_qubits"],
                times=current_times,
                num_measurements=params["measurements"],
                shots=params["shots"],
                spreadings=params["spreading"],
                initial_state_indices=[0],
                seed=fixed["nn_seed"],
                hamiltonian=H,
            )
            targets, times_t, basis, init = dg.generate_dataset()
            ds = TensorDataset(targets, times_t, basis, init)
            loader = DataLoader(
                ds,
                batch_size=get_max_batch_size(fixed["num_qubits"]),
                shuffle=True
            )

            torch.manual_seed(fixed["nn_seed"])
            d = 2 ** fixed["num_qubits"]
            tri_len = d * (d + 1) // 2

            # Pass `device` to Predictor so it creates tensors on CUDA
            predictor = Predictor(
                input_size=tri_len,
                output_size=tri_len,
                hidden_layers=fixed["hidden_layers"],
                activation_fn=fixed["ACTIVATION"],
                ignore_input=True,
                device=fixed["device"]             # ← added
            )
            criterion = Loss(num_qubits=fixed["num_qubits"])
            optimizer = optim.AdamW(predictor.parameters())

            loss_hist = []
            for epoch in range(fixed["epochs"]):
                predictor.train()
                optimizer.zero_grad()
                total_loss = 0.0
                for xb, tb, bb, ib in loader:
                    xb, tb, bb, ib = (t.to(fixed["device"]) for t in (xb, tb, bb, ib))
                    loss = criterion(predictor, tb, ib, xb, bb)
                    loss.backward()
                    total_loss += loss.item()
                optimizer.step()

                avg = total_loss / len(loader)
                loss_hist.append(avg)

                if epoch >= fixed["window"] + 5:
                    recent = np.mean(loss_hist[-fixed["window"]:])
                    prev   = np.mean(loss_hist[-(fixed["window"]+5):-5])
                    if abs(recent - prev) < fixed["tolerance"]:
                        break

            codename = info["name"]
            model_filename = f"embedding_{codename}.pth"
            loss_filename  = f"embedding_{codename}_loss.json"

            torch.save(predictor.state_dict(), os.path.join(subdir, model_filename))
            save_json({"loss_history": loss_hist}, os.path.join(subdir, loss_filename))

            del ds, predictor, xb, tb, bb, ib
            torch.cuda.empty_cache()
            gc.collect()

        overall.update()

    print("All sweeps completed.")


if __name__ == "__main__":
    main()
