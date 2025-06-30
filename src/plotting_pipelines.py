#!/usr/bin/env python3
"""
plotting_pipelines.py

Each “sweep” pipeline now takes an explicit `cache_dir` (instead of inferring
it from __file__).  That way, whatever folder your composite script lives in
can choose exactly which “cached_errors/” to use.
"""

import os
import pickle
import numpy as np

from extraction_and_evalution import (
    collect_recovery_errors_from_data,
    compute_betas_from_errors
)

from plotting_utils import (
    plot_errors_by_spreadings,
    plot_errors_for_outer,
    plot_beta_trends_per_family,
    plot_betas_vs_alpha_alternative,
    plot_betas_vs_alpha_per_family,
    plot_dbetadalpha,
    plot_beta_trends_overlay,
    plot_errors_by_qubit_number
)



def run_sweep1_pipeline(base1: str, cache_dir: str):
    """
    Sweep 1 (time‐scaling by spreading).  Expects
      base1
    to point to first_parameter_sweep_data/.  It will look for
      {cache_dir}/sweep1_errors.pkl
    first, and only if missing will it recompute.
    """
    print("\n=== Running Sweep 1 Pipeline ===")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "sweep1_errors.pkl")

    if os.path.exists(cache_path):
        print(f"  → Loading cached errors from {cache_path}")
        with open(cache_path, "rb") as f:
            combined_errors = pickle.load(f)
    else:
        print("  → No cached file; collecting errors from raw embeddings…")
        run_dirs1 = sorted(d for d in os.listdir(base1) if d.startswith("run_"))
        combined_errors = {}

        for d in run_dirs1:
            run_dir = os.path.join(base1, d)
            if not os.path.isdir(run_dir):
                print(f"  [WARN] Skipping {run_dir}: not found")
                continue

            errs = collect_recovery_errors_from_data(
                run_dir,
                scaling_param="times",
                group_by="spreading"
            )
            for time_tuple, triplets in errs.items():
                combined_errors.setdefault(time_tuple, []).extend(triplets)

        if not combined_errors:
            print("  No data found for Sweep 1. Skipping.\n")
            return

        with open(cache_path, "wb") as f:
            pickle.dump(combined_errors, f)
        print(f"  → Wrote out {len(combined_errors)} keys to {cache_path}")

    # 1) Error vs ∑(time), colored/fitted by spreading
    plot_errors_by_spreadings(
        combined_errors,
        include_families=["XYZ3"],
        exclude_x_scale=[],
        label_prefix="P"
    )

    # 2) Compute β vs spreading and plot
    results = compute_betas_from_errors(
        combined_errors,
        scaling_param="times",
        include_families=["XYZ","XYZ2","XYZ3"],
        exclude_x_scale= None,
        exclude_above_one =True
    )
    plot_beta_trends_per_family(
        results,
        label_prefix="P"
    )    

    # 3) Print β and its 1σ uncertainty for each family & key
    print("\nSpreadings & β ± δβ by family:")
    for fam, (keys, betas, errs) in results.items():
        print(f"\nFamily: {fam}")
        for k, b, err in zip(keys, betas, errs):
            # ensure we have plain floats
            k_val   = float(k)
            b_val   = float(b)
            err_val = float(err)
            print(f"  {k_val:.3f} → {b_val:.3f} ± {err_val:.3f}")

    print("=== Finished Sweep 1 Pipeline ===\n")



def run_sweep2_pipeline(base_time: str, cache_dir: str):
    """
    Sweep 2 (time‐scaling by alpha).  Expects
      base_time
    to point to second_parameter_sweep_data/.  It will look for
      {cache_dir}/sweep2_errors.pkl first.
    """
    print("\n=== Running Sweep 2 Pipeline ===")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "sweep2_errors.pkl")

    if os.path.exists(cache_path):
        print(f"  → Loading cached errors from {cache_path}")
        with open(cache_path, "rb") as f:
            combined_errors = pickle.load(f)
    else:
        print("  → No cached file; collecting errors from raw embeddings…")
        run_dirs_time = sorted(d for d in os.listdir(base_time) if d.startswith("run_"))
        combined_errors = {}

        for d in run_dirs_time:
            run_dir = os.path.join(base_time, d)
            if not os.path.isdir(run_dir):
                print(f"  [WARN] Skipping {run_dir}: not found")
                continue

            errs = collect_recovery_errors_from_data(
                run_dir,
                scaling_param="times",
                group_by="alpha"
            )
            for time_tuple, triplets in errs.items():
                combined_errors.setdefault(time_tuple, []).extend(triplets)

        if not combined_errors:
            print("  No Sweep 2 data found; skipping.\n")
            return None

        with open(cache_path, "wb") as f:
            pickle.dump(combined_errors, f)
        print(f"  → Wrote out {len(combined_errors)} keys to {cache_path}")

    # Fit β(α)
    alphas_time, betas_time, beta_errs_time = compute_betas_from_errors(
        combined_errors,
        scaling_param="times",
        include_families=None,
        exclude_x_scale=None
    )

    # Plot alternative β vs α
    plot_betas_vs_alpha_alternative(
        alphas=alphas_time,
        betas=betas_time,
        beta_errs=beta_errs_time,
        scaling_param="times",
    )
    
    # Print α, β, and uncertainty
    print("Times: \nα & β(α) ± δβ:")
    for a, b, err in zip(alphas_time, betas_time, beta_errs_time):
        a_val = float(a) if isinstance(a, np.ndarray) else a
        b_val = float(b) if isinstance(b, np.ndarray) else b
        err_val = float(err) if isinstance(err, np.ndarray) else err
        print(f"  {a_val:.3f} → {b_val:.3f} ± {err_val:.3f}")


    print("=== Finished Sweep 2 Pipeline ===\n")
    return alphas_time, betas_time, beta_errs_time




def run_sweep2_pipeline(base_time: str, cache_dir: str):
    """
    Sweep 2 (time‐scaling by alpha).  Expects base_time
    to point to second_parameter_sweep_data/.  It will look for
      {cache_dir}/sweep2_errors.pkl first.
    """
    print("\n=== Running Sweep 2 Pipeline ===")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "sweep2_errors.pkl")

    # (1) Load or collect raw errors
    if os.path.exists(cache_path):
        print(f"  → Loading cached errors from {cache_path}")
        with open(cache_path, "rb") as f:
            combined_errors = pickle.load(f)
    else:
        print("  → No cached file; collecting errors from raw embeddings…")
        run_dirs_time = sorted(d for d in os.listdir(base_time) if d.startswith("run_"))
        combined_errors = {}

        for d in run_dirs_time:
            run_dir = os.path.join(base_time, d)
            if not os.path.isdir(run_dir):
                print(f"  [WARN] Skipping {run_dir}: not found")
                continue

            errs = collect_recovery_errors_from_data(
                run_dir,
                scaling_param="times",
                group_by="alpha"
            )
            for time_tuple, triplets in errs.items():
                combined_errors.setdefault(time_tuple, []).extend(triplets)

        if not combined_errors:
            print("  No Sweep 2 data found; skipping.\n")
            return None

        with open(cache_path, "wb") as f:
            pickle.dump(combined_errors, f)
        print(f"  → Wrote out {len(combined_errors)} keys to {cache_path}")

    # (2) Compute per‐family β(α)
    results = compute_betas_from_errors(
        combined_errors,
        scaling_param="times",
        include_families=["XYZ","XYZ2","XYZ3"],
        exclude_x_scale=None,
        exclude_above_one=True
    )

    # (3) Plot using the new per‐family routine
    plot_betas_vs_alpha_per_family(
        results,
        #label_prefix="α"   # or "P" if you prefer that notation
    )
    
    # (4) Print β ± σ for each family, with your chosen label_prefix
    print("\nα & β(α) ± δβ by family:")
    for fam, (keys, betas, errs) in results.items():
        print(f"\nFamily: {fam}")
        for k, b, err in zip(keys, betas, errs):
            k_val   = float(k)
            b_val   = float(b)
            err_val = float(err)
            # prefix the key
            print(f"  α={k_val:.3f} → {b_val:.3f} ± {err_val:.3f}")


    print("\n=== Finished Sweep 2 Pipeline ===\n")
    return results


def run_sweep2_outer(base_time: str, cache_dir: str):
    """
    Sweep 2 Outer: pick one α (largest) and plot Error vs ∑(time).
    Expects cached errors under {cache_dir}/sweep2_errors.pkl.
    """
    print("\n=== Running Sweep 2 Outer (Error vs ∑(time) for one α) ===")
    cache_path = os.path.join(cache_dir, "sweep2_errors.pkl")

    if os.path.exists(cache_path):
        print(f"  → Loading cached errors from {cache_path}")
        with open(cache_path, "rb") as f:
            combined_errors = pickle.load(f)
    else:
        print("  → No cached file; cannot run Sweep 2 Outer.")
        return

    unique_alphas = sorted({
        triplet[2]
        for triplets in combined_errors.values()
        for triplet in triplets
    })
    print("  Available α (Sweep 2 Outer):", unique_alphas)
    alpha_value = unique_alphas[-1]

    plot_errors_for_outer(
        errors_by_scaling=combined_errors,
        scaling_param="times",
        group_by="alpha",
        outer_value=alpha_value,
        include_families=None,
        exclude_x_scale=None,
        show_theory=True
    )
    print("=== Finished Sweep 2 Outer ===\n")


def run_sweep3_pipeline(base_pert: str, cache_dir: str):
    """
    Sweep 3 (spreading‐scaling by alpha).  Expects
      base_pert
    to point to third_parameter_sweep_data/.  It will look for
      {cache_dir}/sweep3_errors.pkl first.
    """
    print("\n=== Running Sweep 3 Pipeline ===")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "sweep3_errors.pkl")

    if os.path.exists(cache_path):
        print(f"  → Loading cached errors from {cache_path}")
        with open(cache_path, "rb") as f:
            combined_errors = pickle.load(f)
    else:
        print("  → No cached file; collecting errors from raw embeddings…")
        run_dirs_pert = sorted(d for d in os.listdir(base_pert) if d.startswith("run_"))
        combined_errors = {}

        for d in run_dirs_pert:
            run_dir = os.path.join(base_pert, d)
            if not os.path.isdir(run_dir):
                print(f"  [WARN] Skipping {run_dir}: not found")
                continue

            errs = collect_recovery_errors_from_data(
                run_dir,
                scaling_param="spreading",
                group_by="alpha"
            )
            for spreading_tuple, triplets in errs.items():
                combined_errors.setdefault(spreading_tuple, []).extend(triplets)

        if not combined_errors:
            print("  No Sweep 3 data found; skipping.\n")
            return None

        with open(cache_path, "wb") as f:
            pickle.dump(combined_errors, f)
        print(f"  → Wrote out {len(combined_errors)} keys to {cache_path}")

    alphas_pert, betas_pert, beta_errs_pert = compute_betas_from_errors(
        combined_errors,
        scaling_param="spreading",
        include_families=None,
        exclude_x_scale=None
    )

    # Plot alternative β vs α
    plot_betas_vs_alpha_alternative(
        alphas=alphas_pert,
        betas=betas_pert,
        beta_errs=beta_errs_pert,
        scaling_param="spreading",
    )
    
    # Print α, β, and uncertainty
    print("Spreadings :\nα & β(α) ± δβ:")
    for a, b, err in zip(alphas_pert, betas_pert, beta_errs_pert):
        a_val = float(a) if isinstance(a, np.ndarray) else a
        b_val = float(b) if isinstance(b, np.ndarray) else b
        err_val = float(err) if isinstance(err, np.ndarray) else err
        print(f"  {a_val:.3f} → {b_val:.3f} ± {err_val:.3f}")


    print("=== Finished Sweep 3 Pipeline ===\n")
    return alphas_pert, betas_pert, beta_errs_pert

import os
import pickle
import numpy as np

def run_sweep3_pipeline(base_pert: str, cache_dir: str):
    """
    Sweep 3 (spreading‐scaling by alpha). Expects base_pert
    to point to third_parameter_sweep_data/.  It will look for
      {cache_dir}/sweep3_errors.pkl first.
    """
    print("\n=== Running Sweep 3 Pipeline ===")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "sweep3_errors.pkl")

    # (1) Load or collect raw errors
    if os.path.exists(cache_path):
        print(f"  → Loading cached errors from {cache_path}")
        with open(cache_path, "rb") as f:
            combined_errors = pickle.load(f)
    else:
        print("  → No cached file; collecting errors from raw embeddings…")
        run_dirs_pert = sorted(d for d in os.listdir(base_pert) if d.startswith("run_"))
        combined_errors = {}

        for d in run_dirs_pert:
            run_dir = os.path.join(base_pert, d)
            if not os.path.isdir(run_dir):
                print(f"  [WARN] Skipping {run_dir}: not found")
                continue

            errs = collect_recovery_errors_from_data(
                run_dir,
                scaling_param="spreading",
                group_by="alpha"
            )
            for spreading_tuple, triplets in errs.items():
                combined_errors.setdefault(spreading_tuple, []).extend(triplets)

        if not combined_errors:
            print("  No Sweep 3 data found; skipping.\n")
            return None

        with open(cache_path, "wb") as f:
            pickle.dump(combined_errors, f)
        print(f"  → Wrote out {len(combined_errors)} keys to {cache_path}")

    # (2) Compute per‐family β(α) under spreading scaling
    results = compute_betas_from_errors(
        combined_errors,
        scaling_param="spreading",
        include_families=None,
        exclude_x_scale=[8],
        exclude_above_one=False
    )

    # (3) Plot β vs α per family
    plot_betas_vs_alpha_per_family(
        results,
        exclude_alphas=[]
        #label_prefix="α"
    )

    # (4) Print α → β ± σ for each family
    print(f"\nα & β(α) ± δβ by family:")
    for fam, (alphas, betas, errs) in results.items():
        print(f"\nFamily: {fam}")
        for a, b, err in zip(alphas, betas, errs):
            a_val   = float(a)
            b_val   = float(b)
            err_val = float(err)
            print(f"  α={a_val:.3f} → {b_val:.3f} ± {err_val:.3f}")

    print("\n=== Finished Sweep 3 Pipeline ===\n")
    return results



def run_sweep3_outer(base_pert: str, cache_dir: str, alpha_index=-1):
    """
    Sweep 3 Outer: pick one α (largest) and plot Error vs ∑(spreading).
    Expects cached errors under {cache_dir}/sweep3_errors.pkl.
    """
    print("\n=== Running Sweep 3 Outer (Error vs ∑(spreading) for one α) ===")
    cache_path = os.path.join(cache_dir, "sweep3_errors.pkl")

    if os.path.exists(cache_path):
        print(f"  → Loading cached errors from {cache_path}")
        with open(cache_path, "rb") as f:
            combined_errors = pickle.load(f)
    else:
        print("  → No cached file; cannot run Sweep 3 Outer.")
        return

    unique_alphas = sorted({
        triplet[2]
        for triplets in combined_errors.values()
        for triplet in triplets
    })
    print("  Available α (Sweep 3 Outer):", unique_alphas)
    alpha_value = unique_alphas[alpha_index]

    plot_errors_for_outer(
        errors_by_scaling=combined_errors,
        scaling_param="spreading",
        group_by="alpha",
        outer_value=alpha_value,
        include_families= None,
        exclude_x_scale=[1,2,4,8,16],
        show_theory=True
    )
    print("=== Finished Sweep 3 Outer ===\n")


def run_derivative_pipeline(time_res: tuple, pert_res: tuple):
    """
    If both time_res and pert_res are not None, extract (alphas_time, betas_time)
    and (alphas_pert, betas_pert), find their common α values, and call plot_dbetadalpha.
    """
    if (time_res is None) or (pert_res is None):
        print("Skipping derivative comparison: missing data.")
        return

    alphas_time, betas_time, beta_errs_time = time_res
    alphas_pert, betas_pert, beta_errs_pert = pert_res

    common_alphas = np.array(
        sorted(set(alphas_time).intersection(set(alphas_pert))),
        dtype=float
    )
    if common_alphas.size < 2:
        print("Insufficient α overlap for derivative plot.")
        return

    idx_time = [np.where(alphas_time == a)[0][0] for a in common_alphas]
    idx_pert = [np.where(alphas_pert == a)[0][0] for a in common_alphas]
    bt_common = betas_time[idx_time]
    bp_common = betas_pert[idx_pert]

    plot_dbetadalpha(
        alphas=common_alphas,
        betas_time=bt_common,
        betas_spreading=bp_common
    )



def run_derivative_pipeline(
    time_res: dict,
    pert_res: dict,
    include_families: list = None,
    show_uncertainty_band: bool = True,
    band_half_width: float = 1/16
):
    """
    Given two result‐dicts:
      time_res[fam] → (alphas_T, betas_T, errs_T)
      pert_res[fam] → (alphas_R, betas_R, errs_R)

    Optionally only for a subset of families via include_families.
    For each selected family present in both dicts, finds common α’s
    and calls plot_dbetadalpha with family‐specific legend labels.
    """
    if time_res is None or pert_res is None:
        print("Skipping derivative comparison: missing data.")
        return

    # base set of families present in both
    common_fams = set(time_res).intersection(pert_res)
    if include_families is not None:
        common_fams &= set(include_families)

    if not common_fams:
        print("No overlapping families to plot derivatives for.")
        return

    for fam in sorted(common_fams):
        alphas_time, betas_time, _ = time_res[fam]
        alphas_pert, betas_pert, _ = pert_res[fam]

        # find common α values
        common_alphas = np.array(
            sorted(set(alphas_time).intersection(alphas_pert)),
            dtype=float
        )
        if common_alphas.size < 2:
            print(f"  Skipping {fam}: insufficient α overlap ({common_alphas.size} points).")
            continue

        # indices into each array
        idx_t = [np.where(alphas_time == a)[0][0] for a in common_alphas]
        idx_p = [np.where(alphas_pert   == a)[0][0] for a in common_alphas]

        bt_common = betas_time[idx_t]
        bp_common = betas_pert[idx_p]

        # call the derivative plot for this family
        plot_dbetadalpha(
            alphas=common_alphas,
            betas_time=bt_common,
            betas_spreading=bp_common,
            label_time=f"Empirical $d\\beta_T/d\\alpha$ ({fam})",
            label_spreading=f"Empirical $d\\beta_R/d\\alpha$ ({fam})",
            show_uncertainty_band=show_uncertainty_band,
            band_half_width=band_half_width
        )


import numpy as np
import matplotlib.pyplot as plt

def run_derivative_pipeline(
    time_res: dict,
    pert_res: dict,
    include_families: list = None,
    show_uncertainty_band: bool = True,
    band_half_width: float = 1/16
):
    """
    Overlay dβ/dα for all (or a subset of) families on one plot.

    time_res[fam] → (alphas_T, betas_T, errs_T)
    pert_res[fam] → (alphas_R, betas_R, errs_R)
    """
    if time_res is None or pert_res is None:
        print("Skipping derivative comparison: missing data.")
        return

    # pick families in common (and filter if requested)
    fams = set(time_res).intersection(pert_res)
    if include_families is not None:
        fams &= set(include_families)
    if not fams:
        print("No families to plot.")
        return

    # create one shared axes
    fig, ax = plt.subplots(figsize=(8, 5))

    # for each family, extract common α and call plot_dbetadalpha onto ax
    for fam in sorted(fams):
        a_t, b_t, _ = time_res[fam]
        a_p, b_p, _ = pert_res[fam]

        # compute common α’s
        common = sorted(set(a_t).intersection(a_p))
        if len(common) < 2:
            print(f"  Skipping {fam}: insufficient α overlap ({len(common)} points).")
            continue

        # indices in each array
        idx_t = [np.where(a_t == α)[0][0] for α in common]
        idx_p = [np.where(a_p == α)[0][0] for α in common]

        a_common = np.array(common, dtype=float)
        bt_common = b_t[idx_t]
        bp_common = b_p[idx_p]

        # call the ax‐aware derivative plot
        plot_dbetadalpha(
            alphas=a_common,
            betas_time=bt_common,
            betas_spreading=bp_common,
            label_time=f"Empirical $d\\beta_T/d\\alpha$ ({fam})",
            label_spreading=f"Empirical $d\\beta_R/d\\alpha$ ({fam})",
            show_uncertainty_band=show_uncertainty_band,
            band_half_width=band_half_width,
            ax=ax
        )

    # finalize: axes labels, title, legend, grid
    ax.set_xlabel(r'$\alpha$', fontsize=15)
    ax.set_ylabel(r'$d\beta/d\alpha$', fontsize=15)
    ax.set_title(r'Derivative of Scaling Exponents vs. $\alpha$', fontsize=16)
    ax.tick_params(labelsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(loc='best', fontsize=12)
    plt.tight_layout()
    plt.show()

    return fig, ax


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def plot_dbetadalpha(
    alphas: np.ndarray,
    betas_time: np.ndarray,
    betas_spreading: np.ndarray,
    label_time: str = "Empirical $dβ_T/dα$",
    label_spreading: str = "Empirical $dβ_R/dα$",
    show_uncertainty_band: bool = True,
    band_half_width: float = 1/16,
    ax: plt.Axes = None
):
    """
    Plot dβ/dα for one family onto ax (or create one if ax is None).

    dβ_empirical/dα = m · (dβ_theory/dα), where m is the slope
    from fitting b vs. β_theory(α).

    Parameters
    ----------
    alphas : np.ndarray
    betas_time : np.ndarray
    betas_spreading : np.ndarray
    label_time : str
    label_spreading : str
    show_uncertainty_band : bool
    band_half_width : float
    ax : matplotlib Axes (optional)
    """
    # 1) Prepare axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # 2) Filter NaNs and sort
    mask = ~np.isnan(betas_time) & ~np.isnan(betas_spreading)
    a = alphas[mask].astype(float)
    bt = betas_time[mask].astype(float)
    bp = betas_spreading[mask].astype(float)
    if a.size < 2:
        print("Not enough data to compute derivatives.")
        return ax

    idx = np.argsort(a)
    a_sorted = a[idx]
    bt_sorted = bt[idx]
    bp_sorted = bp[idx]

    # 3) Theory derivative of β_theory(α)
    alphas_fine = np.linspace(a_sorted.min(), a_sorted.max(), 200)
    d_theory = -1.0 / (2.0 * (alphas_fine + 1.0)**2)

    # 4) Fit slopes m_t, m_p
    beta_theory_data = -((2 * a_sorted + 1) / (2 * (a_sorted + 1)))
    lr_t = LinearRegression().fit(
        beta_theory_data.reshape(-1, 1),
        bt_sorted.reshape(-1, 1)
    )
    m_t = float(lr_t.coef_[0, 0])
    lr_p = LinearRegression().fit(
        beta_theory_data.reshape(-1, 1),
        bp_sorted.reshape(-1, 1)
    )
    m_p = float(lr_p.coef_[0, 0])

    # 5) Empirical derivatives
    d_fit_t = m_t * d_theory
    d_fit_p = m_p * d_theory

    # 6a) Plot theoretical derivative
    ax.plot(
        alphas_fine, d_theory,
        '--', color='C0',
        label=r"Theoretical $d\beta_T/d\alpha$"
    )
    # 6b) Empirical time‐scaling derivative
    ax.plot(
        alphas_fine, d_fit_t,
        '-', linewidth=2, color='C3',
        label=label_time
    )
    # 6c) Empirical spreading‐scaling derivative
    ax.plot(
        alphas_fine, d_fit_p,
        '-', linewidth=2, color='C4',
        label=label_spreading
    )
    # 6d) Uncertainty band
    if show_uncertainty_band:
        ax.fill_between(
            alphas_fine,
            d_fit_t - band_half_width,
            d_fit_t + band_half_width,
            color='C3', alpha=0.3,
            label=r'$\pm$ band around Empirical $d\beta_T/d\alpha$'
        )

    return ax


def run_sweep4_pipeline(base1: str, cache_dir: str):
    """
    Sweep 1 — but grouping by qubit number instead of spreading.
    Looks in `base1` (e.g., first_parameter_sweep_data/) and caches results in `cache_dir/sweep1_by_qubit_errors.pkl`.
    """
    print("\n=== Running Sweep 1 Pipeline (Grouped by Qubit Number) ===")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "sweep1_by_qubit_errors.pkl")

    if os.path.exists(cache_path):
        print(f"  → Loading cached errors from {cache_path}")
        with open(cache_path, "rb") as f:
            combined_errors = pickle.load(f)
    else:
        print("  → No cached file; collecting errors from raw embeddings…")
        run_dirs1 = sorted(d for d in os.listdir(base1) if d.startswith("run_"))
        combined_errors = {}

        for d in run_dirs1:
            run_dir = os.path.join(base1, d)
            if not os.path.isdir(run_dir):
                print(f"  [WARN] Skipping {run_dir}: not found")
                continue

            errs = collect_recovery_errors_from_data(
                run_dir,
                scaling_param="times",
                group_by="num_qubits"
            )
            for time_tuple, triplets in errs.items():
                combined_errors.setdefault(time_tuple, []).extend(triplets)

        if not combined_errors:
            print("  No data found for Sweep 1 by qubit. Skipping.\n")
            return

        with open(cache_path, "wb") as f:
            pickle.dump(combined_errors, f)
        print(f"  → Wrote out {len(combined_errors)} keys to {cache_path}")

    # 1) Error vs ∑(time), grouped/fitted by num_qubits
    plot_errors_by_qubit_number(
        combined_errors,
        include_families=["XYZ3"],  # Adjust as needed
        exclude_x_scale=[],
        label_prefix="Q"
    )

    # 2) Compute β vs qubit number and plot
    results = compute_betas_from_errors(
        combined_errors,
        scaling_param="times",
        include_families=["XYZ", "XYZ2", "XYZ3"],
        exclude_x_scale=None,
        exclude_above_one=True
    )
    plot_beta_trends_per_family(
        results,
        label_prefix="Q"
    )

    # 3) Print β and its 1σ uncertainty for each family & key (qubit number)
    print("\nQubits & β ± δβ by family:")
    for fam, (keys, betas, errs) in results.items():
        print(f"\nFamily: {fam}")
        for k, b, err in zip(keys, betas, errs):
            print(f"  {int(k)} qubits → {b:.3f} ± {err:.3f}")

    print("=== Finished Sweep 1 Pipeline (Grouped by Qubit Number) ===\n")
