# src/plotting_utils.py

import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import scoreatpercentile
from collections import defaultdict

from sklearn.linear_model import LinearRegression

from extraction_and_evalution import collect_recovery_errors_from_data

# ──────────────────────────────────────────────────────────────────────────────
#  Plotting Functions
# ──────────────────────────────────────────────────────────────────────────────

# Define family style maps
FAMILY_MARKERS = {
    "XYZ": "o",
    "XYZ2":        "s",
    "XYZ3":         "D",
    # Add more families as needed...
}
FAMILY_COLORS = {
    "XYZ": "red",
    "XYZ2":        "green",
    "XYZ3":         "purple",
    # Add more families as needed...
}

def plot_errors_by_spreadings(
    errors_by_time: dict,
    include_families: list = None,
    exclude_x_scale: set = None,
    label_prefix: str = "P"       # <<< new: pass "P" (default) or "α" for Figure 2
):
    """
    Plot “error vs. sum_of_time_stamps”, coloring/fitting by the third‐tuple value.
    By default, we label curves with f"{label_prefix}={value}".  If you call with
      label_prefix="α", then legends will read “α=….”
    """
    def power_law(x, a, b):
        return a * x**b

    plt.figure(figsize=(10, 7))
    plt.xscale('log')
    plt.yscale('log')

    # Gather all unique third‐tuple values (either spreadingations or alphas)
    unique_keys = sorted({key for _, errs in errors_by_time.items() for _, _, key in errs})
    cmap = plt.cm.viridis(np.linspace(0, 1, len(unique_keys)))
    color_map = {k: cmap[i] for i, k in enumerate(unique_keys)}

    fit_data = {}

    for time_stamps, errors in sorted(errors_by_time.items(), key=lambda x: round(sum(x[0]), 8)):
        ssum = round(sum(time_stamps), 8)
        bucket = {}
        for rel_err, fam, key in errors:
            if (include_families is not None) and (fam not in include_families):
                continue
            bucket.setdefault(key, []).append(rel_err)

        for key, rez in bucket.items():
            if exclude_x_scale and (ssum in exclude_x_scale):
                continue

            # (1) Scatter all raw points
            label_str = f"{label_prefix}={key:.2f}" if label_prefix == "α" else f"{label_prefix}={key}"
            already_in_legend = label_str in plt.gca().get_legend_handles_labels()[1]

            plt.scatter(
                [ssum] * len(rez),
                rez,
                color=color_map[key],
                edgecolor='black',
                alpha=0.5,
                s=80,
                label=label_str if not already_in_legend else None
            )

            # (2) Prefilter for fitting: keep only rel_err < 1.0
            pref = [e for e in rez if e < 1.0]
            if len(pref) < 2:
                continue

            q0 = scoreatpercentile(pref, 0)
            q50 = scoreatpercentile(pref, 50)
            filt = [e for e in pref if (q0 <= e <= q50)]

            fit_data.setdefault(key, {"x": [], "y": []})
            fit_data[key]["x"].extend([ssum] * len(filt))
            fit_data[key]["y"].extend(filt)

    # Fit & plot one curve per key
    for key, data in fit_data.items():
        fx = np.array(data["x"])
        fy = np.array(data["y"])
        if len(fx) < 2 or len(fy) < 2:
            continue

        popt, pcov = curve_fit(power_law, fx, fy, p0=(1, -0.5))
        a, b = popt
        a_err, b_err = np.sqrt(np.diag(pcov))

        def round_sig(val, err):
            sig_digit = -int(np.floor(np.log10(err)))
            return round(val, sig_digit), round(err, sig_digit)

        a_r, a_err_r = round_sig(a, a_err)
        b_r, b_err_r = round_sig(b, b_err)

        x_fit = np.linspace(min(fx), max(fx), 100)
        y_fit = power_law(x_fit, a, b)
        label_str = f"{label_prefix}={key:.2f}" if label_prefix == "α" else f"{label_prefix}={key}"
        plt.plot(
            x_fit,
            y_fit,
            '--',
            color=color_map[key],
            label=f"{label_str} fit: y=({a_r}±{a_err_r}) x^{b_r}±{b_err_r}"
        )

    plt.xlabel("Total Experiment Time (log)", fontsize=16)
    plt.ylabel("Error (log)", fontsize=16)
    title_prefix = "Error vs Total Experiment Time"
    if label_prefix == "α":
        plt.title(f"{title_prefix} (grouped by α)", fontsize=18)
    else:
        plt.title(f"Effect of Number of Spreadings on Learning Rate", fontsize=18)

    # Increase tick label sizes
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.grid(True, which='major', linestyle='-', linewidth=0.5, color='black', alpha=0.7)
    plt.grid(True, which='minor', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
    #plt.legend(fontsize=12)  # Legend size is already ≥ 15 elsewhere if uncommented
    plt.tight_layout()
    plt.show()


def plot_errors_by_spreadings(
    errors_by_time: dict,
    include_families: list = None,   # e.g. ["XYZ3"]
    exclude_x_scale: set = None,
    label_prefix: str = "P"
):
    """
    Plot “error vs. sum_of_time_stamps”, grouping/fitting by the third‐tuple key
    but only for the given family or families.
    """
    def power_law(x, a, b):
        return a * x**b

    # === NEW: filter out any families not in include_families ===
    if include_families is not None:
        filtered = {}
        for t, recs in errors_by_time.items():
            kept = [ (e,f,k) for (e,f,k) in recs if f in include_families ]
            if kept:
                filtered[t] = kept
        errors_by_time = filtered
        if not errors_by_time:
            print("No data for families:", include_families)
            return
    # === end new ===

    plt.figure(figsize=(10, 7))
    plt.xscale('log')
    plt.yscale('log')

    # (unchanged) Gather all unique third‐tuple values (either spreadings or α's)
    unique_keys = sorted({ key for _, errs in errors_by_time.items() for _, _, key in errs })
    cmap = plt.cm.viridis(np.linspace(0, 1, len(unique_keys)))
    color_map = {k: cmap[i] for i, k in enumerate(unique_keys)}

    fit_data = {}

    # (unchanged) scatter & bucket
    for time_stamps, errors in sorted(
        errors_by_time.items(),
        key=lambda x: round(sum(x[0]), 8)
    ):
        ssum = round(sum(time_stamps), 8)
        bucket = {}
        for rel_err, fam, key in errors:
            # no need to re-check fam; we've pre-filtered
            bucket.setdefault(key, []).append(rel_err)

        for key, rez in bucket.items():
            if exclude_x_scale and (ssum in exclude_x_scale):
                continue

            # (1) scatter
            label_str = (
                f"{label_prefix}={key:.2f}"
                if label_prefix=="α"
                else f"{label_prefix}={key}"
            )
            already = label_str in plt.gca().get_legend_handles_labels()[1]

            plt.scatter(
                [ssum]*len(rez),
                rez,
                color=color_map[key],
                edgecolor='black',
                alpha=0.5,
                s=80,
                label=None #if already else label_str
            )

            # (2) build fit_data
            if ssum==0.1:
                pref = [e for e in rez if e < 1.25]
            else:
                pref = [e for e in rez if e < 1.0]
            if len(pref) < 1:
                continue
            q0 = scoreatpercentile(pref, 0)
            q50 = scoreatpercentile(pref, 50)
            filt = [e for e in pref if q0 <= e <= q50]


            fit_data.setdefault(key, {"x": [], "y": []})
            fit_data[key]["x"].extend([ssum]*len(filt))
            fit_data[key]["y"].extend(filt)

    # (unchanged) Fit & plot one curve per key
    for key, data in fit_data.items():
        fx = np.array(data["x"], float)
        fy = np.array(data["y"], float)
        if fx.size < 2 or fy.size < 2:
            continue

        popt, pcov = curve_fit(power_law, fx, fy, p0=(1, -0.5))
        a, b = popt
        a_err, b_err = np.sqrt(np.diag(pcov))

        def round_sig(v, e):
            sig = -int(np.floor(np.log10(e))) if e>0 else 2
            return round(v, sig), round(e, sig)
        a_r, a_er = round_sig(a, a_err)
        b_r, b_er = round_sig(b, b_err)

        x_fit = np.linspace(fx.min(), fx.max(), 100)
        y_fit = power_law(x_fit, a, b)
        label_str = (
            f"{label_prefix}={key:.2f}"
            if label_prefix=="α"
            else f"{label_prefix}={key}"
        )

        plt.plot(
            x_fit, y_fit, '--',
            color=color_map[key],
            label=f"{label_str} fit: y=({a_r}±{a_er})·x^({b_r}±{b_er})"
        )

    # (unchanged) finalize
    plt.xlabel("Total Experiment Time (log)", fontsize=16)
    plt.ylabel("Error (log)", fontsize=16)
    if label_prefix == "α":
        plt.title("Error vs Total Experiment Time (grouped by α)", fontsize=18)
    else:
        plt.title("Effect of Number of Spreadings on Learning Rate", fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True, which='major', linestyle='-', linewidth=0.5, color='black', alpha=0.7)
    plt.grid(True, which='minor', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_beta_trends(
    keys: np.ndarray,
    betas: np.ndarray,
    beta_errs: np.ndarray = None,
    label_prefix: str = "P"
):
    """
    Plot fitted β values (learning‐rate exponents) against their corresponding keys
    (either spreadingation values or α values) on a log‐scale x‐axis.

    Parameters:
    -----------
    keys : np.ndarray
        Array of keys (e.g. spreadingation values or α values). Can be any shape; will be flattened.
    betas : np.ndarray
        Array of fitted β exponents corresponding to each key. Can be any shape; will be flattened.
    beta_errs : np.ndarray or None
        (Optional) Array of uncertainties σ_b for each fitted β. If provided,
        error bars will be drawn. Can be any shape; will be flattened.
    label_prefix : str
        If plotting vs. α, pass "α" so that axes and titles label appropriately;
        otherwise (default) it is interpreted as “spreadingation.”

    Behavior:
    ---------
    - Converts inputs to 1D numpy arrays.
    - Scatter the (key, β) points in black with a thin edge.
    - Optionally draw vertical error bars at each point if beta_errs is given.
    - Connect the points in ascending‐key order with a gray line to show trend.
    - Draw a horizontal reference line at β = –0.75 (theoretical prediction)
      and a shaded band of width ±1/8 around it.
    - Use log‐scale for the x‐axis.
    """
    # (0) Ensure inputs are 1D numpy arrays
    keys = np.asarray(keys, dtype=float).ravel()
    betas = np.asarray(betas, dtype=float).ravel()
    if beta_errs is not None:
        beta_errs = np.asarray(beta_errs, dtype=float).ravel()

    # (1) Sort by keys so that the connecting line is monotonic
    idx = np.argsort(keys)
    keys_sorted = keys[idx]
    betas_sorted = betas[idx]
    if beta_errs is not None:
        beta_errs_sorted = beta_errs[idx]
    else:
        beta_errs_sorted = None

    plt.figure(figsize=(10, 7))

    # (2) Scatter or errorbar plot of (key, beta)
    if beta_errs_sorted is not None:
        plt.errorbar(
            keys_sorted,
            betas_sorted,
            yerr=beta_errs_sorted,
            fmt='o',
            color='black',
            ecolor='black',
            elinewidth=1,
            capsize=4,
            alpha=0.8,
            markersize=8,
            markeredgecolor='k',
            markeredgewidth=0.5,
            label="Fitted β ± σ"
        )
    else:
        plt.scatter(
            keys_sorted,
            betas_sorted,
            c='black',
            s=80,
            alpha=0.8,
            edgecolor='k',
            linewidth=0.5,
            label="Fitted β"
        )

    # (3) Trend line connecting the points
    plt.plot(
        keys_sorted,
        betas_sorted,
        '-', color='gray', alpha=0.6,
        label="Data trend"
    )

    # (4) Theoretical β = –0.75 and shaded band ±1/8
    theoretical_beta = -0.75
    half_width = 1 / 8
    plt.axhline(
        y=theoretical_beta,
        color='green',
        linestyle=':',
        linewidth=3,
        label=r"Theoretical β = –0.75"
    )
    plt.axhspan(
        theoretical_beta - half_width,
        theoretical_beta + half_width,
        color='green',
        alpha=0.2,
        label=r"O(m_t⁻¹) band around –0.75"
    )

    # (5) Axis scales and labels
    #plt.xscale('log')

    if label_prefix == "α":
        xlabel = "α (log)"
        plt.title("Error Scaling β vs. α", fontsize=16)
    else:
        xlabel = "State Spreadings (log)"
        plt.title("Error Scaling β for Number of State Spreading", fontsize=16)

    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel("Error Scaling β", fontsize=15)

    # Increase tick label sizes
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.legend(fontsize=15)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()
   

def plot_beta_trends_per_family(
    results: dict,
    label_prefix: str = "P"
):
    """
    Plot β vs key for multiple families on a log‐x axis.

    Parameters
    ----------
    results : dict
        family_name → (keys_array, betas_array, beta_errs_array or None)
    label_prefix : str
        If "α", axis/title will label α; otherwise it's “State Spreadings”.
    """
    plt.figure(figsize=(10, 7))

    # (0) For each family, scatter+trend
    for fam, (keys, betas, beta_errs) in results.items():
        # flatten & sort
        k = np.asarray(keys, float).ravel()
        b = np.asarray(betas, float).ravel()
        idx = np.argsort(k)
        k, b = k[idx], b[idx]
        errs = None
        if beta_errs is not None:
            errs = np.asarray(beta_errs, float).ravel()[idx]

        marker = FAMILY_MARKERS.get(fam, "o")
        color  = FAMILY_COLORS.get(fam, "black")

        if errs is not None:
            plt.errorbar(
                k, b, yerr=errs,
                fmt=marker,
                markersize=8,
                markeredgecolor='k',
                markerfacecolor=color,
                ecolor=color,
                elinewidth=1,
                capsize=4,
                alpha=0.8,
                label=fam
            )
        else:
            plt.scatter(
                k, b,
                marker=marker,
                s=80,
                edgecolor='k',
                facecolor=color,
                alpha=0.8,
                label=fam
            )

        # trend line
        plt.plot(
            k, b,
            '-', color=color, alpha=0.6
        )

    # (1) Theoretical β = –0.75 and ±1/8 band
    theo = -0.75
    half = 1/8
    plt.axhline(theo, color='green', linestyle=':', linewidth=3, label=r"Theoretical β = –0.75")
    plt.axhspan(theo-half, theo+half, color='green', alpha=0.2, label=r"O(m_t⁻¹) band")

    # (2) Axis scale & labels
    #plt.xscale('log')
    if label_prefix == "α":
        plt.xlabel("α (log)", fontsize=15)
        plt.title("Error‐Scaling β vs. α", fontsize=16)
    else:
        plt.xlabel("State Spreadings", fontsize=15)
        plt.title("Error‐Scaling β vs. Number of Spreadings", fontsize=16)

    plt.ylabel("Error‐Scaling β", fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=12, loc='best')
    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

def plot_beta_trends_overlay(
    results: dict,
    label_prefix: str = "P"
):
    """
    Overlay β vs. key curves for multiple families on one log–x plot.

    Parameters
    ----------
    results : dict
        family_name → (keys_array, betas_array, beta_errs_array)
    label_prefix : str
        "α" for α‐axis labeling, otherwise treated as P/spreadings.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xscale('log')

    # Plot each family
    for fam, (keys, betas, beta_errs) in results.items():
        k = np.asarray(keys, float).ravel()
        b = np.asarray(betas, float).ravel()
        sort_idx = np.argsort(k)
        k, b = k[sort_idx], b[sort_idx]
        errs = None
        if beta_errs is not None:
            errs = np.asarray(beta_errs, float).ravel()[sort_idx]

        marker = FAMILY_MARKERS.get(fam, 'o')
        color  = FAMILY_COLORS.get(fam, 'black')

        if errs is not None:
            ax.errorbar(
                k, b, yerr=errs,
                fmt=marker,
                markersize=8,
                markeredgecolor='k',
                markerfacecolor=color,
                ecolor=color,
                elinewidth=1,
                capsize=4,
                alpha=0.8,
                label=fam
            )
        else:
            ax.scatter(
                k, b,
                marker=marker,
                s=80,
                edgecolor='k',
                facecolor=color,
                alpha=0.8,
                label=fam
            )

        ax.plot(k, b, '-', color=color, alpha=0.6)

    # Add theoretical reference line and band
    theo = -0.75
    half_width = 1 / 8
    ax.axhline(theo, color='green', linestyle=':', linewidth=3,
               label=r"Theoretical β = –0.75")
    ax.axhspan(theo - half_width, theo + half_width,
               color='green', alpha=0.2,
               label=r"O(m_t⁻¹) band around –0.75")

    # Labels and title
    if label_prefix == "α":
        ax.set_xlabel(r'$\alpha$ (log)', fontsize=15)
        ax.set_title("Error‐Scaling β vs. α", fontsize=16)
    else:
        ax.set_xlabel("State Spreadings", fontsize=15)
        ax.set_title("Error‐Scaling β vs. Number of Spreadings", fontsize=16)
    ax.set_ylabel("Error‐Scaling β", fontsize=15)
      # Set y-axis limits from –1.0 up to 0.05
    #ax.set_ylim(-1.0, 0.05)
    ax.tick_params(labelsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=12, loc='best')
    plt.tight_layout()
    plt.show()

    return fig, ax



def plot_errors_for_outer(
    errors_by_scaling: dict,
    scaling_param: str,
    group_by: str,
    outer_value,
    include_families: list = None,
    exclude_x_scale: set = None,
    show_theory: bool = True
):
    """
    Generalized “Error vs. sum(inner_tuple)” plot for a single outer_value, 
    with labels automatically derived from scaling_param and group_by.

    Parameters
    ----------
    errors_by_scaling : dict
        { inner_tuple → [ (rel_err, family, group_key), … ] }
        where `inner_tuple` was determined by scaling_param, and `group_key` by group_by.

    scaling_param : str
        Either "times" or "spreading". Indicates which meta‐parameter was used as the dict key.
        This affects the x‐axis label: 
          - "times" → "Sum of Time Stamps (log)"
          - "spreading" → "Sum of spreadingation (log)"

    group_by : str
        One of "alpha", "times", or "spreading". Indicates which meta‐parameter is used as the 
        “outer” grouping. Affects the plot title:
          - "alpha" → "α"
          - "times" → "Time Stamps"
          - "spreading" → "spreadingation"

    outer_value : scalar or tuple
        The specific group_key value to plot. All triplets whose third element == outer_value
        will be included.

    include_families : list or None
        If provided, only errors whose `family` is in this list are included.

    exclude_x_scale : set or None
        If provided, any ssum = round(sum(inner_tuple), 8) in this set will be scattered
        (all raw errs in gray) but excluded from fitting.

    show_theory : bool
        If True, overlay SQL (x^(-0.5)) and Heisenberg (x^(-1)) reference curves.

    Behavior
    --------
    1) Filters triplets in errors_by_scaling where group_key == outer_value.
    2) Groups those filtered errors by inner_tuple.
    3) For each inner_tuple (in ascending order of sum(inner_tuple)):
         a) Compute ssum = round(sum(inner_tuple), 8).
         b) If ssum ∈ exclude_x_scale: scatter raw errs (gray) & skip fitting.
         c) Else:
            • If ssum < 2: pref = errs.
            • Else: pref = [e for e in errs if e < 0.1].
            • If len(pref) < 2: scatter raw errs & skip fitting.
            • Else:
                – q0 = 0th percentile of pref, q50 = 50th percentile of pref.
                – filt = [e for e in pref if q0 ≤ e ≤ q50].
                – If len(filt) < 2: scatter raw errs & skip fitting.
                – Else:
                    • Scatter raw errs (gray), collect (ssum, e) for each e in filt into fit_x, fit_y.
    4) If len(fit_x) ≥ 2:
         • Fit fit_y ≈ a·(fit_x)^b via curve_fit.
         • Compute σ_a, σ_b from covariance.
         • Plot best‐fit line “y = (a±σ_a)·x^(b±σ_b)” in red dashed.
    5) If show_theory: overlay SQL (∝ x^(-0.5)) and Heisenberg (∝ x^(-1)) reference curves,
       normalized to pass through the first fitted point.

    Example usage
    -------------
    >>> # Suppose collect_recovery_errors_from_data(..., scaling_param="spreading", group_by="alpha")
    >>> errs_by_spreading = {
    ...     (0.1, 0.2): [ (0.05, "XY", 0.5), (0.08, "Heisenberg", 0.5), … ],
    ...     (0.2, 0.3): [ … ],
    ... }
    >>> plot_errors_for_outer(
    ...     errors_by_scaling=errs_by_spreading,
    ...     scaling_param="spreading",
    ...     group_by="alpha",
    ...     outer_value=0.5,
    ...     include_families=None,
    ...     exclude_x_scale=None,
    ...     show_theory=True
    ... )
    """
    # Validate arguments
    if scaling_param not in {"times", "spreading"}:
        raise ValueError("scaling_param must be 'times' or 'spreading'")
    if group_by not in {"alpha", "times", "spreading"}:
        raise ValueError("group_by must be 'alpha', 'times', or 'spreading'")
    if group_by == scaling_param:
        raise ValueError("group_by must differ from scaling_param")

    # Determine axis labels from scaling_param and group_by
    inner_label = "Total Experiment Time" if scaling_param == "times" else "Number of Spreadings"
    outer_label = {
        "alpha": "α",
        "times": "Total Experiment Time",
        "spreading": "Number of Spreadings"
    }[group_by]

    # (1) Filter triplets where group_key == outer_value
    filtered = []
    for inner_tuple, triplets in errors_by_scaling.items():
        for (rel_err, family, group_key) in triplets:
            if group_key == outer_value:
                filtered.append((rel_err, family, inner_tuple))

    if not filtered:
        print(f"No data for {outer_label} = {outer_value}")
        return

    # (2) Prepare figure
    plt.figure(figsize=(8, 7))
    plt.xscale("log")
    plt.yscale("log")

    # (3) Group by inner_tuple
    inner_groups = {}
    for (rel_err, family, inner_tuple) in filtered:
        if (include_families is not None) and (family not in include_families):
            continue
        inner_groups.setdefault(inner_tuple, []).append(rel_err)

    fit_x = []
    fit_y = []

    # (4) Iterate over sorted inner_tuples
    for inner_tuple, errs in sorted(
        inner_groups.items(),
        key=lambda item: round(sum(item[0]), 8)
    ):
        ssum = round(sum(inner_tuple), 8)

        # (4a) Exclude from fitting if requested
        if (exclude_x_scale is not None) and (ssum in exclude_x_scale):
            plt.scatter(
                [ssum] * len(errs),
                errs,
                color="gray",
                edgecolor="black",
                alpha=0.5,
                s=80,
                label=None
            )
            continue

        # (4b) Custom prefilter
        #if ssum < 2:
        #    pref = errs.copy()
        #else:
        #    pref = [e for e in errs if e < 0.1]
            
        if scaling_param == 'times':
            if ssum < 2:
                pref = errs.copy()
            else:
                pref = [e for e in errs if e < 0.1]
        else:
            #pref = [e for e in errs if e < 2.0]
            #pref = errs.copy()
            if ssum < 50:
                pref = errs.copy()
            else:
                pref = [e for e in errs if e < 0.1]


        if len(pref) < 2:
            plt.scatter(
                [ssum] * len(errs),
                errs,
                color="gray",
                edgecolor="black",
                alpha=0.5,
                s=80,
                label=None
            )
            continue

        # (4c) Percentile‐filter on pref
        q0  = scoreatpercentile(pref, 0)
        q50 = scoreatpercentile(pref, 50)
        filt = [e for e in pref if (q0 <= e <= q50)]
        if len(filt) < 2:
            plt.scatter(
                [ssum] * len(errs),
                errs,
                color="gray",
                edgecolor="black",
                alpha=0.5,
                s=80,
                label=None
            )
            continue

        # (4d-1) Scatter all raw errs in gray
        plt.scatter(
            [ssum] * len(errs),
            errs,
            color="gray",
            edgecolor="black",
            alpha=0.5,
            s=80,
            label=None
        )

        # (4d-2) Append filtered errs for fitting
        fit_x.extend([ssum] * len(filt))
        fit_y.extend(filt)

    # (5) Fit a power‐law if we have ≥2 points
    def _power_law(x, a, b):
        return a * np.power(x, b)

    if len(fit_x) >= 2 and len(fit_y) >= 2:
        fx = np.array(fit_x, dtype=float)
        fy = np.array(fit_y, dtype=float)

        # Sort by fx
        idx_sort = np.argsort(fx)
        fx = fx[idx_sort]
        fy = fy[idx_sort]

        try:
            (a_fit, b_fit), pcov = curve_fit(_power_law, fx, fy, p0=(1.0, -0.5))
            sigma_a, sigma_b = np.sqrt(np.diag(pcov))
        except Exception:
            a_fit = np.nan
            b_fit = np.nan
            sigma_a = np.nan
            sigma_b = np.nan

        # (6) Compute smooth fit curve
        x_fit = np.logspace(np.log10(fx.min()), np.log10(fx.max()), 200)
        y_fit = _power_law(x_fit, a_fit, b_fit)

        # (7) Overlay SQL & Heisenberg if requested
        if show_theory:
            y_sql = y_fit[0] * (x_fit / x_fit[0]) ** (-0.5)
            plt.plot(
                x_fit, y_sql,
                color="black", linestyle="-", linewidth=2, alpha=0.7,
                label="SQL ∝ x⁻⁰․⁵"
            )

        # (8) Plot the best‐fit line and annotate uncertainties
        def _round_sig(val, err):
            if np.isnan(err) or err == 0:
                return round(val, 2), round(err, 2)
            sig = -int(np.floor(np.log10(err)))
            return round(val, sig), round(err, sig)

        a_r, a_err_r = _round_sig(a_fit, sigma_a)
        b_r, b_err_r = _round_sig(b_fit, sigma_b)

        plt.plot(
            x_fit,
            y_fit,
            'r--',
            linewidth=2,
            label=f"Fit: y = ({a_r} ± {a_err_r})·x^({b_r} ± {b_err_r})",
            zorder=3,
            clip_on=False
        )
        
        if show_theory:
            y_heis = y_fit[0] * (x_fit / x_fit[0]) ** (-1.0)
            plt.plot(
                x_fit, y_heis,
                color="blue", linestyle="-", linewidth=2, alpha=0.7,
                label="Heisenberg ∝ x⁻¹"
            )

    # (9) Finalize labels and title
    plt.xlabel(f"{inner_label} (log)", fontsize=16)
    plt.ylabel("Error (log)", fontsize=16)
    plt.title(f"Error vs {inner_label} ( {outer_label} = {outer_value} )", fontsize=18)

    # Increase tick label sizes
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.grid(True, which="major", linestyle="-", linewidth=0.5, color="black", alpha=0.7)
    plt.grid(True, which="minor", linestyle="--", linewidth=0.5, color="gray", alpha=0.7)
    plt.legend(fontsize=15, loc="best")
    plt.tight_layout()
    plt.show()




def plot_errors_for_outer(
    errors_by_scaling: dict,
    scaling_param: str,
    group_by: str,
    outer_value,
    include_families: list = ["XYZ"],
    exclude_x_scale: set = None,
    show_theory: bool = True
):
    """
    Plot error vs. sum(inner_tuple) for a given outer_value, with one curve per family.
    """
    # Validate arguments
    if scaling_param not in {"times", "spreading"}:
        raise ValueError("scaling_param must be 'times' or 'spreading'")
    if group_by not in {"alpha", "times", "spreading"}:
        raise ValueError("group_by must be 'alpha', 'times', or 'spreading'")
    if group_by == scaling_param:
        raise ValueError("group_by must differ from scaling_param")

    # Labels
    inner_label = "Total Experiment Time" if scaling_param == "times" else "Number of Spreadings"
    outer_label = {"alpha": "α", "times": "Total Experiment Time", "spreading": "Number of Spreadings"}[group_by]

    # (1) Filter triplets for this outer_value
    filtered = []
    for inner_tuple, triplets in errors_by_scaling.items():
        for (rel_err, family, group_key) in triplets:
            if group_key == outer_value:
                filtered.append((inner_tuple, family, rel_err))
    if not filtered:
        print(f"No data for {outer_label} = {outer_value}")
        return

    # Group by inner_tuple
    inner_groups = {}
    for inner_tuple, family, rel_err in filtered:
        if include_families and family not in include_families:
            continue
        inner_groups.setdefault(inner_tuple, []).append((family, rel_err))

    # Prepare figure
    plt.figure(figsize=(8, 7))
    plt.xscale("log")
    plt.yscale("log")

    fit_x, fit_y = [], []
    plotted_families = set()

    # Iterate sorted by sum(inner_tuple)
    for inner_tuple in sorted(inner_groups.keys(), key=lambda t: round(sum(t), 8)):
        ssum = round(sum(inner_tuple), 8)
        fam_errs_list = inner_groups[inner_tuple]
        families_here = sorted({fam for fam, _ in fam_errs_list})
        center_offset = (len(families_here) - 1) / 2

        for i, family in enumerate(families_here):
            fam_errs = [err for fam, err in fam_errs_list if fam == family]
            # Prefilter
            if scaling_param == 'times':
                pref = fam_errs if ssum < 2 else [e for e in fam_errs if e < 0.1]
            else:
                pref = fam_errs if ssum < 50 else [e for e in fam_errs if e < 0.1]
            if len(pref) < 2:
                continue

            # Percentile filter
            q0, q50 = scoreatpercentile(pref, 0), scoreatpercentile(pref, 50)
            filt = [e for e in pref if q0 <= e <= q50]
            if len(filt) < 2:
                continue

            # Scatter
            offset = (i - center_offset) * ssum * 0.02
            x_vals = [ssum + offset] * len(fam_errs)
            plt.scatter(
                x_vals, fam_errs,
                marker=FAMILY_MARKERS.get(family, 'o'),
                color=FAMILY_COLORS.get(family, 'black'),
                edgecolor='black',
                alpha=0.7,
                s=100,
                label=family if family not in plotted_families else None
            )
            plotted_families.add(family)

            # Collect for fit, unless excluded
            if exclude_x_scale is None or ssum not in exclude_x_scale:
                fit_x.extend([ssum] * len(filt))
                fit_y.extend(filt)

    # Fit power-law
    def _power(x, a, b):
        return a * np.power(x, b)

    if len(fit_x) >= 2:
        fx, fy = np.array(fit_x), np.array(fit_y)
        idx = np.argsort(fx)
        fx, fy = fx[idx], fy[idx]

        try:
            (a_fit, b_fit), pcov = curve_fit(_power, fx, fy, p0=(1.0, -0.5))
            sigma_a, sigma_b = np.sqrt(np.diag(pcov))
        except:
            a_fit = b_fit = sigma_a = sigma_b = np.nan

        # Smooth curve
        x_fit = np.logspace(np.log10(fx.min()), np.log10(fx.max()), 200)
        y_fit = _power(x_fit, a_fit, b_fit)

        # Theory
        if show_theory:
            y_sql = y_fit[0] * (x_fit / x_fit[0])**(-0.5)
            plt.plot(x_fit, y_sql, '-', label="SQL ∝ x⁻⁰․⁵", linewidth=2, alpha=0.7)
            y_heis = y_fit[0] * (x_fit / x_fit[0])**(-1.0)
            plt.plot(x_fit, y_heis, '-', label="Heisenberg ∝ x⁻¹", color='blue', linewidth=2, alpha=0.7)

        # Fit line
        def round_sig(val, err):
            if np.isnan(err) or err == 0:
                return round(val, 2), round(err, 2)
            sig = -int(np.floor(np.log10(err)))
            return round(val, sig), round(err, sig)
        a_r, a_err_r = round_sig(a_fit, sigma_a)
        b_r, b_err_r = round_sig(b_fit, sigma_b)

        plt.plot(
            x_fit, y_fit, 'r--',
            label=f"Fit: y=({a_r}±{a_err_r})·x^({b_r}±{b_err_r})",
            linewidth=2, zorder=3, clip_on=False
        )

    # Finalize
    plt.xlabel(f"{inner_label} (log)", fontsize=16)
    plt.ylabel("Error (log)", fontsize=16)
    plt.title(f"Error vs {inner_label} ({outer_label}={outer_value})", fontsize=18)
    plt.xticks(fontsize=15); plt.yticks(fontsize=15)
    plt.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=14, loc='best')
    plt.tight_layout()
    plt.show()




def plot_errors_for_outer(
    errors_by_scaling: dict,
    scaling_param: str,
    group_by: str,
    outer_value,
    include_families: list = None,
    exclude_x_scale: set = None,
    show_theory: bool = True
):
    """
    Plot error vs. sum(inner_tuple) for a given outer_value, with one curve per family.
    """
    # Validate arguments
    if scaling_param not in {"times", "spreading"}:
        raise ValueError("scaling_param must be 'times' or 'spreading'")
    if group_by not in {"alpha", "times", "spreading"}:
        raise ValueError("group_by must be 'alpha', 'times', or 'spreading'")
    if group_by == scaling_param:
        raise ValueError("group_by must differ from scaling_param")

    # Labels
    inner_label = "Total Experiment Time" if scaling_param == "times" else "Number of Spreadings"
    outer_label = {"alpha": "α", "times": "Total Experiment Time", "spreading": "Number of Spreadings"}[group_by]

    # (1) Filter triplets for this outer_value
    filtered = []
    for inner_tuple, triplets in errors_by_scaling.items():
        for (rel_err, family, group_key) in triplets:
            if group_key == outer_value:
                filtered.append((inner_tuple, family, rel_err))
    if not filtered:
        print(f"No data for {outer_label} = {outer_value}")
        return

    # Group by inner_tuple
    inner_groups = {}
    for inner_tuple, family, rel_err in filtered:
        if include_families and family not in include_families:
            continue
        inner_groups.setdefault(inner_tuple, []).append((family, rel_err))

    # Prepare figure
    plt.figure(figsize=(8, 7))
    plt.xscale("log")
    plt.yscale("log")

    # instead of one global fit, collect per-family
    fit_data = {fam: ([], []) for fam in FAMILY_MARKERS.keys()}
    plotted_families = set()

    # Iterate sorted by sum(inner_tuple)
    for inner_tuple in sorted(inner_groups.keys(), key=lambda t: round(sum(t), 8)):
        ssum = round(sum(inner_tuple), 8)
        fam_errs_list = inner_groups[inner_tuple]
        families_here = sorted({fam for fam, _ in fam_errs_list})
        center_offset = (len(families_here) - 1) / 2

        for i, family in enumerate(families_here):
            fam_errs = [err for fam, err in fam_errs_list if fam == family]
            # Prefilter
            if scaling_param == 'times':
                pref = fam_errs if ssum < 2 else [e for e in fam_errs if e < 0.1]
            else:
                pref = fam_errs if ssum < 50 else [e for e in fam_errs if e < 0.1]
            if len(pref) < 2:
                continue

            # Percentile filter
            q0, q50 = scoreatpercentile(pref, 0), scoreatpercentile(pref, 50)
            filt = [e for e in pref if q0 <= e <= q50]
            if len(filt) < 2:
                continue

            # Scatter
            offset = (i - center_offset) * ssum * 0.02
            x_vals = [ssum + offset] * len(fam_errs)
            plt.scatter(
                x_vals, fam_errs,
                marker=FAMILY_MARKERS.get(family, 'o'),
                color=FAMILY_COLORS.get(family, 'black'),
                edgecolor='black',
                alpha=0.7,
                s=100,
                label=family if family not in plotted_families else None
            )
            plotted_families.add(family)

            # Collect for fit, unless excluded
            if exclude_x_scale is None or ssum not in exclude_x_scale:
                fx_list, fy_list = fit_data[family]
                fx_list.extend([ssum] * len(filt))
                fy_list.extend(filt)
                fit_data[family] = (fx_list, fy_list)

    # power-law model
    def _power(x, a, b):
        return a * np.power(x, b)

    # now do one power-law fit & plot per family
    for family, (fx_list, fy_list) in fit_data.items():
        if len(fx_list) < 2:
            continue

        fx = np.array(fx_list); fy = np.array(fy_list)
        idx = np.argsort(fx)
        fx, fy = fx[idx], fy[idx]

        try:
            (a_fit, b_fit), pcov = curve_fit(_power, fx, fy, p0=(1.0, -0.5))
            sigma_a, sigma_b = np.sqrt(np.diag(pcov))
        except:
            continue

        # generate smooth fit-curve
        x_fit = np.logspace(np.log10(fx.min()), np.log10(fx.max()), 200)
        y_fit = _power(x_fit, a_fit, b_fit)

        # optionally plot theory curves once (using first family’s fit as baseline)
        if show_theory and family == list(fit_data.keys())[0]:
            y_sql = y_fit[0] * (x_fit / x_fit[0])**(-0.5)
            plt.plot(x_fit, y_sql, '-', label="SQL ∝ x⁻⁰․⁵", linewidth=2, alpha=0.7)
            y_heis = y_fit[0] * (x_fit / x_fit[0])**(-1.0)
            plt.plot(x_fit, y_heis, '-', label="Heisenberg ∝ x⁻¹", color='blue', linewidth=2, alpha=0.7)

        # round to significant figures
        def round_sig(val, err):
            if np.isnan(err) or err == 0:
                return round(val, 2), round(err, 2)
            sig = -int(np.floor(np.log10(err)))
            return round(val, sig), round(err, sig)
        a_r, a_err_r = round_sig(a_fit, sigma_a)
        b_r, b_err_r = round_sig(b_fit, sigma_b)

        plt.plot(
            x_fit, y_fit, linestyle='--',
            label=f"{family} fit: y=({a_r}±{a_err_r})·x^({b_r}±{b_err_r})",
            linewidth=2, alpha=0.8
        )

    # Finalize
    plt.xlabel(f"{inner_label} (log)", fontsize=16)
    plt.ylabel("Error (log)", fontsize=16)
    plt.title(f"Error vs {inner_label} ({outer_label}={outer_value})", fontsize=18)
    plt.xticks(fontsize=15); plt.yticks(fontsize=15)
    plt.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=14, loc='best')
    plt.tight_layout()
    plt.show()


def plot_betas_vs_alpha_alternative(alphas, betas, beta_errs, scaling_param: str):
    """
    Plot “β vs α” in the style of the example’s first panel:
      • A dashed line for the theoretical β_theory(α) = –((2α+1)/(2(α+1))).
      • Error‐bar scatter of (α, fitted b ± σ_b).
      • A linear regression fit of b vs. β_theory(α), plotted as a solid line.

    Inputs:
      alphas    : 1D array of α values (shape (N,))
      betas     : 1D array of fitted exponent b for each α (shape (N,))
      beta_errs : 1D array of 1σ uncertainties in b (shape (N,))
    """
    
    inner_label = "Run-Time" if scaling_param == "times" else "State-Spreadings"
    inner_label_sign = r"T" if scaling_param == "times" else "R"

    # 1) Filter out any NaNs so we only fit actual data points
    mask = ~np.isnan(betas)
    a_data = np.array(alphas[mask], dtype=float)
    b_data = np.array(betas[mask], dtype=float)
    e_data = np.array(beta_errs[mask], dtype=float)

    if a_data.size == 0:
        print("No valid data points to plot.")
        return

    # 2) Compute theoretical curve: β_theory(α) = –((2α+1)/(2(α+1)))
    alphas_fine = np.linspace(a_data.min(), a_data.max(), 200)
    beta_theory_fine = -((2 * alphas_fine + 1) / (2 * (alphas_fine + 1)))
    beta_theory_data = -((2 * a_data + 1) / (2 * (a_data + 1)))

    # 3) Fit a linear model: b_data ≈ m·β_theory_data + c
    lr = LinearRegression()
    lr.fit(beta_theory_data.reshape(-1, 1), b_data.reshape(-1, 1))
    m = lr.coef_[0, 0]
    c = lr.intercept_[0]
    fit_line_fine = m * beta_theory_fine + c

    # --- Begin plotting ---
    plt.figure(figsize=(8, 5))

    # 1) Dashed theoretical curve
    plt.plot(
        alphas_fine,
        beta_theory_fine,
        '--',
        color='C0',
        label=r'Theoretical $\beta_T(\alpha)$'
    )

    # 2) Error‐bar scatter of fitted b(α)
    plt.errorbar(
        a_data,
        b_data,
        yerr=e_data,
        fmt='o',
        capsize=4,
        ecolor='black',
        elinewidth=1,
        alpha=0.8,
        markersize=8,
        markeredgecolor='k',
        color='C3',
        label=fr'Fitted $\beta_{inner_label_sign}(\alpha)\pm\sigma_\beta$'
    )

    # 3) Solid regression fit: b_fit(α) = m·β_theory(α) + c
    plt.plot(
        alphas_fine,
        fit_line_fine,
        '-',
        linewidth=2,
        color='C3',
        label=f'Fit'#: slope={m:.2f}, intercept={c:.2f}'
    )

    # Plot formatting
    plt.xlabel(r'$\alpha$', fontsize=15)
    plt.ylabel(r'Error‐scaling exponent $\beta$', fontsize=15)
    plt.title(fr'{inner_label} Error‐Scaling Exponent vs. $\alpha$', fontsize=16)

    # Increase tick label sizes
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=15, loc='best')
    plt.tight_layout()
    plt.show()
    

def plot_betas_vs_alpha_per_family(
    results: dict,
    scaling_param: str = "spreading",
    show_regression: bool = True
):
    """
    Plot β vs α for multiple families.

    Parameters
    ----------
    results : dict
        family → (alphas_array, betas_array, beta_errs_array)
    scaling_param : str
        "times" or "spreading" (affects title/labels)
    show_regression : bool
        if True, fit & plot m·β_theory + c for each family

    Behavior
    --------
    • Draws the theoretical curve β_theory(α) = –(2α+1)/[2(α+1)] once, dashed.
    • For each family in `results`:
        – scatter α vs β with error‐bars, using 
          FAMILY_MARKERS[family], FAMILY_COLORS[family]
        – optionally, fit β_data ≈ m·β_theory_data + c and plot 
          that fit in the family color, solid line.
    """
    inner_label = "Run-Time" if scaling_param == "times" else "State-Spreadings"
    inner_label_sign = "T" if scaling_param == "times" else "R"

    # 1) Compute overall theory curve
    # collect all α across families to get min/max
    all_a = np.hstack([res[0] for res in results.values()])
    a_min, a_max = np.nanmin(all_a), np.nanmax(all_a)
    alphas_fine = np.linspace(a_min, a_max, 300)
    beta_theory_fine = -((2 * alphas_fine + 1) / (2 * (alphas_fine + 1)))

    plt.figure(figsize=(8, 5))
    # plot theory once
    plt.plot(
        alphas_fine, beta_theory_fine,
        linestyle="--", linewidth=2,
        label=r"Theoretical $\beta(\alpha) = -\frac{2\alpha+1}{2(\alpha+1)}$"
    )

    # 2) loop families
    for fam, (alphas, betas, errs) in results.items():
        # mask out NaNs
        mask = ~np.isnan(betas)
        a_data = alphas[mask]
        b_data = betas[mask]
        e_data = errs[mask]
        if a_data.size == 0:
            continue

        # scatter + errorbars
        plt.errorbar(
            a_data, b_data, yerr=e_data,
            fmt=FAMILY_MARKERS.get(fam, "o"),
            markersize=8,
            markeredgecolor="k",
            markerfacecolor=FAMILY_COLORS.get(fam, "black"),
            ecolor=FAMILY_COLORS.get(fam, "black"),
            elinewidth=1.5,
            capsize=4,
            alpha=0.8,
            label=f"{fam} & Fit"
        )

        if show_regression:
            # compute beta_theory at data points
            beta_theory_data = -((2 * a_data + 1) / (2 * (a_data + 1)))

            # fit linear: b_data ≈ m·beta_theory_data + c
            lr = LinearRegression().fit(
                beta_theory_data.reshape(-1, 1),
                b_data.reshape(-1, 1)
            )
            m, c = lr.coef_[0, 0], lr.intercept_[0]

            # plot fit line over fine grid
            fit_line = m * beta_theory_fine + c
            plt.plot(
                alphas_fine, fit_line,
                linestyle="-", linewidth=2,
                color=FAMILY_COLORS.get(fam, "black"),
                alpha=0.7,
                label=f"{fam} fit (m={m:.2f}, c={c:.2f})"
            )

    # formatting
    plt.xlabel(r"$\alpha$", fontsize=15)
    plt.ylabel(r"$\beta$", fontsize=15)
    plt.title(f"{inner_label} Error-Scaling Exponent vs. $\\alpha$", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=12, loc="best")
    plt.tight_layout()
    plt.show()


def plot_betas_vs_alpha_per_family(
    results: dict,
    scaling_param: str = "spreading",
    show_regression: bool = True
):
    inner_label = "Run-Time" if scaling_param == "times" else "State-Spreadings"

    # 1) Theoretical curve
    all_a = np.hstack([res[0] for res in results.values()])
    fine = np.linspace(np.nanmin(all_a), np.nanmax(all_a), 300)
    beta_theory_fine = -((2 * fine + 1) / (2 * (fine + 1)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        fine, beta_theory_fine,
        '--', color='C0', linewidth=2,
        label=r"Theoretical $\beta(\alpha)$"
    )

    # We'll collect real ErrorbarContainer handles for the legend
    legend_handles = [ax.lines[-1]]   # the theoretical line handle
    legend_labels  = [r"Theoretical $\beta(\alpha)$"]

    # 2) Per‐family plotting
    for fam, (alphas, betas, errs) in results.items():
        mask = ~np.isnan(betas)
        a = alphas[mask]
        b = betas[mask]
        e = errs[mask]
        if a.size == 0:
            continue

        marker = FAMILY_MARKERS.get(fam, "o")
        color  = FAMILY_COLORS.get(fam, "black")

        # Draw the real data
        eb = ax.errorbar(
            a, b, yerr=e,
            fmt=marker,
            linestyle='None',
            markersize=8,
            markeredgecolor='k',
            markerfacecolor=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=4,
            alpha=0.8
        )

        # Regression line (unlabeled)
        if show_regression:
            beta_th = -((2 * a + 1) / (2 * (a + 1)))
            lr = LinearRegression().fit(
                beta_th.reshape(-1,1), b.reshape(-1,1)
            )
            m, c = lr.coef_[0,0], lr.intercept_[0]
            ax.plot(
                fine, m*beta_theory_fine + c,
                '-', linewidth=2, color=color, alpha=0.7
            )

        # Add the ErrorbarContainer to our legend handles
        legend_handles.append(eb)
        legend_labels.append(f'{fam} & Fit')

    # Reference band
    #theo = -0.75; half = 1/8
    #ax.axhline(theo, color='green', linestyle=':', linewidth=3)
    #ax.axhspan(theo-half, theo+half, color='green', alpha=0.2)

    # Formatting
    ax.set_xlabel(r"$\alpha$", fontsize=15)
    ax.set_ylabel(r"$\beta$", fontsize=15)
    ax.set_title(f"{inner_label} Error‐Scaling Exponent vs. $\\alpha$", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Build legend from our collected handles, which include real ErrorbarContainers
    ax.legend(legend_handles, legend_labels, fontsize=12, loc='best')

    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_combined_betas_vs_alpha_two_panels(
    time_res, pert_res
):
    """
    Create a two‐panel figure:
      • Top panel: Sweep 2 (time) “β vs α” with theoretical curve, error bars, and empirical fit.
      • Bottom panel: Sweep 3 (spreading) “β vs α” with the same theoretical curve, error bars, and empirical fit.

    Parameters
    ----------
    time_res : tuple or None
        If not None, should be (alphas_time, betas_time, beta_errs_time).
    pert_res : tuple or None
        If not None, should be (alphas_pert, betas_pert, beta_errs_pert).
    """
    if (time_res is None) or (pert_res is None):
        print("Skipping combined two‐panel plot: missing data from one of the sweeps.")
        return

    alphas_time, betas_time, beta_errs_time = time_res
    alphas_pert, betas_pert, beta_errs_pert = pert_res

    # --- 1) Filter out NaNs for each dataset ---
    mask_time = ~np.isnan(betas_time)
    a_time = np.array(alphas_time)[mask_time].astype(float)
    b_time = np.array(betas_time)[mask_time].astype(float)
    e_time = np.array(beta_errs_time)[mask_time].astype(float)

    mask_pert = ~np.isnan(betas_pert)
    a_pert = np.array(alphas_pert)[mask_pert].astype(float)
    b_pert = np.array(betas_pert)[mask_pert].astype(float)
    e_pert = np.array(beta_errs_pert)[mask_pert].astype(float)

    if a_time.size == 0 and a_pert.size == 0:
        print("No valid data in either Sweep 2 or Sweep 3.")
        return

    # --- 2) Build a combined α range for theoretical curve ---
    all_alphas = np.hstack([a_time, a_pert]) if (a_time.size and a_pert.size) else (a_time if a_pert.size == 0 else a_pert)
    amin, amax = all_alphas.min(), all_alphas.max()
    alphas_fine = np.linspace(amin, amax, 200)

    # Theoretical β(α) = –((2α+1)/(2(α+1))) for any α
    def beta_theory(alpha_array):
        return -((2 * alpha_array + 1) / (2 * (alpha_array + 1)))

    expected_fine_neg = beta_theory(alphas_fine)

    # Compute theoretical values at discrete α for fitting
    beta_theory_time = beta_theory(a_time) if a_time.size > 0 else np.array([])
    beta_theory_pert = beta_theory(a_pert) if a_pert.size > 0 else np.array([])

    # --- 3) Fit linear models ---
    # Sweep 2 (time)
    if a_time.size > 0:
        lr_time = LinearRegression()
        lr_time.fit(beta_theory_time.reshape(-1, 1), b_time.reshape(-1, 1))
        m_time, c_time = float(lr_time.coef_[0, 0]), float(lr_time.intercept_[0])
        fit_t_fine = m_time * expected_fine_neg + c_time
    else:
        fit_t_fine = None

    # Sweep 3 (spreading)
    if a_pert.size > 0:
        lr_pert = LinearRegression()
        lr_pert.fit(beta_theory_pert.reshape(-1, 1), b_pert.reshape(-1, 1))
        m_pert, c_pert = float(lr_pert.coef_[0, 0]), float(lr_pert.intercept_[0])
        fit_p_fine = m_pert * expected_fine_neg + c_pert
    else:
        fit_p_fine = None

    # --- 4) Create two stacked subplots ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    # --- Top panel: Sweep 2 (time) ---
    ax1.plot(
        alphas_fine,
        expected_fine_neg,
        '--',
        color='C0',
        label=r'Theoretical $β_T(\alpha)$'
    )
    if a_time.size > 0:
        ax1.errorbar(
            a_time,
            b_time,
            yerr=e_time,
            fmt='o',
            capsize=4,
            ecolor='black',
            elinewidth=1,
            alpha=0.8,
            markersize=8,
            markeredgecolor='k',
            color='C3',
            label=r'$β_T(\alpha)$ fitted for particular α`s '
        )
        ax1.plot(
            alphas_fine,
            fit_t_fine,
            '-',
            linewidth=2,
            color='C3',
            label=r'Fit: empirical $β_T(\alpha)$ trend '
        )
    ax1.set_ylabel(r'Error Scaling $\beta$', fontsize=15)
    ax1.set_title('Run-time Error Scaling vs α', fontsize=16)
    ax1.legend(fontsize=12, loc='best')
    ax1.grid(True)
    ax1.tick_params(axis='both', labelsize=15)

    # --- Bottom panel: Sweep 3 (spreading) ---
    ax2.plot(
        alphas_fine,
        expected_fine_neg,
        '--',
        color='C0',
        label=r'Theoretical $β_T(\alpha)$'
    )
    if a_pert.size > 0:
        ax2.errorbar(
            a_pert,
            b_pert,
            yerr=e_pert,
            fmt='o',
            capsize=4,
            ecolor='black',
            elinewidth=1,
            alpha=0.8,
            markersize=8,
            markeredgecolor='k',
            color='C4',
            label='β(R) fitted for particular α`s '
        )
        ax2.plot(
            alphas_fine,
            fit_p_fine,
            '-',
            linewidth=2,
            color='C4',
            label='Fit: empirical β(R) trend  '
        )
    ax2.set_xlabel('α', fontsize=15)
    ax2.set_ylabel(r'Error Scaling $\beta$', fontsize=15)
    ax2.set_title('State-Spreading Error Scaling vs α', fontsize=16)
    ax2.legend(fontsize=12, loc='best')
    ax2.grid(True)
    ax2.tick_params(axis='both', labelsize=15)

    plt.tight_layout()
    plt.show()


def plot_dbetadalpha(
    alphas: np.ndarray,
    betas_time: np.ndarray,
    betas_spreading: np.ndarray,
    label_time: str = "Empirical $dβ_T/dα$",
    label_spreading: str = "Empirical $dβ_R/dα$",
    show_uncertainty_band: bool = True,
    band_half_width: float = 1/16
):
    """
    Plot dβ/dα for both “time‐scaling” and “spreading‐scaling” fits,
    using linear‐fit slopes m_t and m_p from b vs. β_theory(α), so that:
      dβ_empirical/dα = m · (dβ_theory/dα).

    Inputs:
      alphas         : 1D array of α values (shape (N,))
      betas_time     : 1D array of fitted exponent b_T for time‐scaling
      betas_spreading  : 1D array of fitted exponent b_R for spreading‐scaling
    """
    # (1) Filter out any NaNs and sort
    mask = ~np.isnan(betas_time) & ~np.isnan(betas_spreading)
    a_data = alphas[mask].astype(float)
    bt_data = betas_time[mask].astype(float)
    bp_data = betas_spreading[mask].astype(float)

    if a_data.size < 2:
        print("Not enough data to compute derivatives.")
        return

    idx = np.argsort(a_data)
    a_sorted = a_data[idx]
    bt_sorted = bt_data[idx]
    bp_sorted = bp_data[idx]

    # (2) Theoretical derivative of β_theory(α)
    alphas_fine = np.linspace(a_sorted.min(), a_sorted.max(), 200)
    d_theory = -1.0 / (2.0 * (alphas_fine + 1.0)**2)

    # (3) Fit linear model b_T ≈ m_t * β_theory(α) + c_t
    beta_theory_data = -((2 * a_sorted + 1) / (2 * (a_sorted + 1)))
    lr_t = LinearRegression().fit(
        beta_theory_data.reshape(-1, 1),
        bt_sorted.reshape(-1, 1)
    )
    m_t = float(lr_t.coef_[0, 0])

    # (4) Fit linear model b_R ≈ m_p * β_theory(α) + c_p
    lr_p = LinearRegression().fit(
        beta_theory_data.reshape(-1, 1),
        bp_sorted.reshape(-1, 1)
    )
    m_p = float(lr_p.coef_[0, 0])

    # (5) Empirical derivatives: dβ_empirical/dα = m · dβ_theory/dα
    d_fit_t = m_t * d_theory
    d_fit_p = m_p * d_theory

    # (6) Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    # (6a) Theoretical derivative (dashed, C0)
    ax.plot(
        alphas_fine,
        d_theory,
        '--',
        color='C0',
        label=r"Theoretical $d\beta_T/d\alpha$"
    )

    # (6b) Empirical time‐scaling derivative (solid, C3)
    ax.plot(
        alphas_fine,
        d_fit_t,
        '-',
        linewidth=2,
        color='C3',
        label=f"{label_time} "#(slope={m_t:.2f})"
    )

    # (6c) Empirical spreading‐scaling derivative (solid, C4)
    ax.plot(
        alphas_fine,
        d_fit_p,
        '-',
        linewidth=2,
        color='C4',
        label=f"{label_spreading} "#(slope={m_p:.2f})"
    )

    # (6d) ±band around time‐scaling derivative if requested
    if show_uncertainty_band:
        ax.fill_between(
            alphas_fine,
            d_fit_t - band_half_width,
            d_fit_t + band_half_width,
            color='C3',
            alpha=0.3,
            label=r'$\pm\,(\Omega(m_t^{-1}))/2$ around Empirical $d\beta_T/d\alpha$'
        )

    ax.set_xlabel(r'Scheduling exponent $\alpha$', fontsize=15)
    ax.set_ylabel(r'Sensitivity $dβ/dα$', fontsize=15)
    ax.set_title(r'Derivative of Scaling Exponents vs. $\alpha$', fontsize=16)

    # Increase tick label sizes
    ax.tick_params(axis='both', labelsize=15)

    ax.legend(loc='best', frameon=True, fontsize=13)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()



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
