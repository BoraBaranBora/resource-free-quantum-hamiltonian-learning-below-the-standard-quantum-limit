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
import matplotlib.lines as mlines

from sklearn.linear_model import LinearRegression

from extraction_and_evalution import collect_recovery_errors_from_data

# ──────────────────────────────────────────────────────────────────────────────
#  Plotting Functions
# ──────────────────────────────────────────────────────────────────────────────

# Define family style maps
FAMILY_MARKERS = {
    "Heisenberg": "o",
    "XYZ": "o",
    "XYZ2":        "s",
    "XYZ3":         "D",
    # Add more families as needed...
}
FAMILY_COLORS = {
    "Heisenberg": "red",
    "XYZ": "red",
    "XYZ2":        "green",
    "XYZ3":         "purple",
    # Add more families as needed...
}


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
        return a * x**-b

    # === NEW: filter out any families not in include_families ===
    if include_families is not None:
        filtered = {}
        for t, recs in errors_by_time.items():
            kept = [
                (e, f if f != "XXYGL" else "XXZ", k)
                for (e, f, k) in recs
                if (f if f != "XXYGL" else "XXZ") in include_families
            ]
            if kept:
                filtered[t] = kept
        errors_by_time = filtered
        if not errors_by_time:
            print("No data for families:", include_families)
            return

    # === end new ===

    plt.figure(figsize=(8, 7))
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
            if ssum==0.01:
                pref = [e for e in rez if e < 20.0]
            else:
                pref = [e for e in rez if e < 20.0]
            if len(pref) < 1:
                continue
            q0 = scoreatpercentile(pref, 0)
            q50 = scoreatpercentile(pref, 100)
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

        popt, pcov = curve_fit(power_law, fx, fy, p0=(1, 0.5))
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
        p_number = int(label_str[label_str.find('(')+1 : label_str.find(',')])

        plt.plot(
            x_fit, y_fit, '--',
            color=color_map[key],
            label=f"{p_number}",#f"{label_str} fit: y=({a_r}±{a_er})·x^({b_r}±{b_er})"
        )

    # (unchanged) finalize
    plt.xlabel("Total Experiment Time ", fontsize=20)
    plt.ylabel("Error ", fontsize=20)
    if label_prefix == "α":
        plt.title("Error vs Total Experiment Time (grouped by α)", fontsize=20)
    else:
        plt.title(f"Effect of Spreadings on Error Scaling; ({include_families[0]})", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, which='major', linestyle='-', linewidth=0.5, color='black', alpha=0.7)
    plt.grid(True, which='minor', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
    plt.legend(fontsize=15)
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
    plt.figure(figsize=(8, 7))

    # (0) For each family, scatter+trend
    for orig_fam, (keys, betas, beta_errs) in results.items():
        fam = "XXZ" if orig_fam == "XXYGL" else orig_fam
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
    theo = 0.75
    half = 1/8
    plt.axhline(theo, color='green', linestyle=':', linewidth=3, label=r"Theoretical β(α=1.0) = 0.75")
    plt.axhspan(theo-half, theo+half, color='green', alpha=0.2, label=r"O(m_t⁻¹) band")

    # add horizontal line at 0.5 labeled SQL
    plt.axhline(1.0, color='blue', linestyle=':', linewidth=2, label='Heisenberg Limit')

    # add horizontal line at 0.5 labeled SQL
    plt.axhline(0.5, color='black', linestyle=':', linewidth=2, label='SQL')
    
    
    # (2) Axis scale & labels
    #plt.xscale('log')
    if label_prefix == "α":
        plt.xlabel("α ", fontsize=20)
        plt.title("Error‐Scaling β vs. α", fontsize=20)
    else:
        plt.xlabel("State Spreadings", fontsize=20)
        plt.title(r"Error‐Scaling β vs. Number of Spreadings ($\alpha$ =1.0)", fontsize=20)

    plt.ylabel("Error‐Scaling β", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=15, loc='best')
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
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import curve_fit
    from scipy.stats import scoreatpercentile

    # Validate arguments
    if scaling_param not in {"times", "spreading"}:
        raise ValueError("scaling_param must be 'times' or 'spreading'")
    if group_by not in {"alpha", "times", "spreading"}:
        raise ValueError("group_by must be 'alpha', 'times', or 'spreading'")
    if group_by == scaling_param:
        raise ValueError("group_by must differ from scaling_param")

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

    fit_data = {}  # Collect only actual families with data
    plotted_families = []  # Preserve plotting order

    # Iterate sorted by inner_tuple
    for inner_tuple in sorted(inner_groups.keys()):
        fam_errs_list = inner_groups[inner_tuple]
        families_here = sorted({fam for fam, _ in fam_errs_list})
        center_offset = (len(families_here) - 1) / 2

        x_base = sum(inner_tuple)

        for i, family in enumerate(families_here):
            fam_errs = [err for fam, err in fam_errs_list if fam == family]

            if scaling_param == 'times':
                pref = fam_errs if x_base < 2 else [e for e in fam_errs if e < 50.0]
            else:
                pref = fam_errs if x_base < 10 else [e for e in fam_errs if e < 50.0]
            if len(pref) < 1:
                continue

            q0, q50 = scoreatpercentile(pref, 0), scoreatpercentile(pref, 50)
            filt = [e for e in pref if q0 <= e <= q50]
            if len(filt) < 1:
                continue

            offset = (i - center_offset) * x_base * 0.01
            x_vals = [x_base + offset] * len(fam_errs)

            display_family = "XXZ" if family == "XXYGL" else family

            plt.scatter(
                x_vals, fam_errs,
                marker=FAMILY_MARKERS.get(display_family, 'o'),
                color=FAMILY_COLORS.get(display_family, 'black'),
                edgecolor='black',
                alpha=0.65,
                s=100,
                label=display_family if display_family not in plotted_families else None
            )

            if display_family not in plotted_families:
                plotted_families.append(display_family)

            if exclude_x_scale is None or x_base not in exclude_x_scale:
                if display_family not in fit_data:
                    fit_data[display_family] = ([], [])
                fx_list, fy_list = fit_data[display_family]
                fx_list.extend([x_base] * len(filt))
                fy_list.extend(filt)

    # Power-law model
    def _power(x, a, b):
        return a * np.power(x, -b)

    theory_plotted = False  # Only plot theory lines once

    for display_family in plotted_families:
        if display_family not in fit_data or len(fit_data[display_family][0]) < 1:
            continue

        fx = np.array(fit_data[display_family][0])
        fy = np.array(fit_data[display_family][1])
        idx = np.argsort(fx)
        fx, fy = fx[idx], fy[idx]

        try:
            (a_fit, b_fit), pcov = curve_fit(_power, fx, fy, p0=(1.0, 0.5))
            sigma_a, sigma_b = np.sqrt(np.diag(pcov))
        except Exception:
            continue

        x_fit = np.logspace(np.log10(fx.min()), np.log10(fx.max()), 200)
        y_fit = _power(x_fit, a_fit, b_fit)

        if show_theory and not theory_plotted:
            y_sql = y_fit[0] * (x_fit / x_fit[0])**(-0.5)
            plt.plot(x_fit, y_sql, '-', label="SQL ∝ x⁰․⁵", linewidth=2, alpha=0.7)
            y_heis = y_fit[0] * (x_fit / x_fit[0])**(-1.0)
            plt.plot(x_fit, y_heis, '-', label="Heisenberg ∝ x¹", color='blue', linewidth=2, alpha=0.7)
            theory_plotted = True

        def round_sig(val, err):
            if np.isnan(err) or err == 0:
                return round(val, 2), round(err, 2)
            sig = -int(np.floor(np.log10(err)))
            return round(val, sig), round(err, sig)

        a_r, a_err_r = round_sig(a_fit, sigma_a)
        b_r, b_err_r = round_sig(b_fit, sigma_b)

        plt.plot(
            x_fit, y_fit, linestyle='--',
            color=FAMILY_COLORS.get(display_family, 'black'),
            label=f"{display_family} fit: y=({a_r}±{a_err_r})·x^({b_r}±{b_err_r})",
            linewidth=2, alpha=0.8
        )

    plt.xlabel(f"{inner_label} ", fontsize=20)
    plt.ylabel("Error ", fontsize=20)
    title = f"Error vs {inner_label} ({outer_label}={outer_value})"
    if include_families and len(include_families) == 1:
        title += f" - {display_family}"
    plt.title(title, fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=15, loc='best')
    plt.tight_layout()
    plt.show()


def plot_each_family_separately(
    errors_by_scaling: dict,
    scaling_param: str,
    group_by: str,
    outer_value,
    families: list,
    exclude_x_scale: set = None,
    show_theory: bool = True
):
    for family in families:
        print(f"\nPlotting for family: {family}")
        plot_errors_for_outer(
            errors_by_scaling=errors_by_scaling,
            scaling_param=scaling_param,
            group_by=group_by,
            outer_value=outer_value,
            include_families=[family],
            exclude_x_scale=exclude_x_scale,
            show_theory=show_theory
        )



def plot_betas_vs_alpha_per_family(
    results: dict,
    scaling_param: str = "times",
    show_regression: bool = True,
    exclude_alphas: list = []
):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from scipy.optimize import curve_fit

    def beta_model(alpha, gamma0, B):
        return 0.5 * (alpha * gamma0 + 1) / (alpha + 1) + B

    inner_label = "Total Exp. Time" if scaling_param == "times" else "State-Spreadings"
    exclude_alphas = set(exclude_alphas or [])

    all_a = np.hstack([res[0] for res in results.values()])
    fine = np.linspace(np.nanmin(all_a), np.nanmax(all_a), 300)
    beta_theory_fine = beta_model(fine, gamma0=1.0, B=0.0)

    fig, ax = plt.subplots(figsize=(8, 6))

    legend_handles = []
    legend_labels = []

    for fam, (alphas, betas, errs) in results.items():
        mask = ~np.isnan(betas)
        a = alphas[mask]
        b = betas[mask]
        e = errs[mask]

        if exclude_alphas:
            include_mask = ~np.isin(a, list(exclude_alphas))
            a, b, e = a[include_mask], b[include_mask], e[include_mask]

        if a.size == 0:
            continue

        color = FAMILY_COLORS.get(fam, "black")
        marker = FAMILY_MARKERS.get(fam, "o")

        label = fam
        if show_regression:
            try:
                popt, pcov = curve_fit(
                    beta_model,
                    a, b,
                    #sigma=e,
                    #absolute_sigma=True,
                    p0=[1.0, 0.0],  # initial guess: gamma_0 = 1, B = 0
                    bounds=([-5.0, -2.0], [5.0, 2.0])
                )
                gamma0_fit, B_fit = popt
                gamma0_err, B_err = np.sqrt(np.diag(pcov))

                label = (
                    fr"{fam} & fit: "
                    fr"$\gamma_0 = {gamma0_fit:.2f} \pm {gamma0_err:.2f},\; "
                    fr"B = {B_fit:.2f} \pm {B_err:.2f}$"
                )

                alpha_grid = np.linspace(a.min(), a.max(), 300)
                fit_curve = beta_model(alpha_grid, *popt)
                ax.plot(
                    alpha_grid, fit_curve,
                    '-', linewidth=2, color=color, alpha=0.8, label=None
                )
            except Exception as err:
                print(f"[WARN] Fit failed for {fam}: {err}")
                label = f"{fam} (fit failed)"

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
            alpha=0.8,
            label=label
        )

        legend_handles.append(eb)
        legend_labels.append(label)

    ax.set_xlabel(r"$\alpha$", fontsize=20)
    ax.set_ylabel(r"$\beta_{\mathrm{T}_{\mathrm{tot}}}$", fontsize=20)
    ax.set_title(f"{inner_label} Error‐Scaling Exponent vs. $\\alpha$", fontsize=20)
    ax.tick_params(labelsize=18)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(handles=legend_handles, labels=legend_labels, fontsize=13, loc='best')

    plt.tight_layout()
    plt.show()
    return fig, ax




def plot_betas_vs_alpha_per_family(
    results: dict,
    scaling_param: str = "times",
    gamma0_theory: float = 2.0,
    finite_mt_offset: float = -0.09,
    exclude_alphas: list = []
):
    import numpy as np
    import matplotlib.pyplot as plt

    def beta_theory(alpha, gamma0):
        return 0.5 * (alpha * gamma0 + 1) / (alpha + 1)

    inner_label = "Total Exp. Time" if scaling_param == "times" else "State-Spreadings"
    exclude_alphas = set(exclude_alphas or [])

    all_a = np.hstack([res[0] for res in results.values()])
    fine = np.linspace(np.nanmin(all_a), np.nanmax(all_a), 300)
    theory_vals = beta_theory(fine, gamma0=gamma0_theory)
    theory_shifted = theory_vals + finite_mt_offset

    fig, ax = plt.subplots(figsize=(8, 6))
    data_handles = []
    data_labels = []

    for orig_fam, (alphas, betas, errs) in results.items():
        fam = "XXZ" if orig_fam == "XXYGL" else orig_fam

        mask = ~np.isnan(betas)
        a = alphas[mask]
        b = betas[mask]
        e = errs[mask]

        if exclude_alphas:
            include_mask = ~np.isin(a, list(exclude_alphas))
            a, b, e = a[include_mask], b[include_mask], e[include_mask]

        if a.size == 0:
            continue

        color = FAMILY_COLORS.get(fam, "black")
        marker = FAMILY_MARKERS.get(fam, "o")

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
            alpha=0.8,
            label=fam
        )

        data_handles.append(eb)
        data_labels.append(fam)

    # Plot theoretical curves
    theory_main, = ax.plot(
        fine, theory_vals,
        '-', linewidth=2.5, color='black', alpha=0.8,
        label=fr"Prediction"
    )
    theory_shifted_line, = ax.plot(
        fine, theory_shifted,
        '--', linewidth=2.5, color='gray', alpha=0.8,
        label=fr"Prediction, shifted"
    )

    # Legend for data families
    data_legend = ax.legend(
        handles=data_handles,
        labels=data_labels,
        fontsize=13,
        loc='upper left',
        title="Data families"
    )
    ax.add_artist(data_legend)

    # Separate legend for theory lines
    ax.legend(
        handles=[theory_main, theory_shifted_line],
        loc='lower right',
        fontsize=15,
        title='Theoretical curves'
    )

    ax.set_xlabel(r"$\alpha$", fontsize=20)
    ax.set_ylabel(r"$\beta_{\mathrm{T}_{\mathrm{tot}}}$", fontsize=20)
    ax.set_title(f"{inner_label} Error‐Scaling Exponent vs. $\\alpha$", fontsize=20)
    ax.tick_params(labelsize=18)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.show()
    return fig, ax
