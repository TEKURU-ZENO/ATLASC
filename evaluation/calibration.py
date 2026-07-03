"""
calibration.py

Epistemic calibration analysis for ATLAS ensemble predictions.

Calibration is defined as: model's uncertainty estimate U_traj should be
monotonically correlated with actual empirical error (MSE).
A perfectly calibrated model means higher U_traj cells actually have higher MSE.
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
from typing import Tuple, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def compute_calibration_curve(
    uncertainties: np.ndarray,
    mses: np.ndarray,
    num_bins: int = 10,
) -> dict:
    """
    Bin uncertainties into deciles and compute mean MSE per bin.

    Args:
        uncertainties: [N] per-cell uncertainty values.
        mses: [N] per-cell MSE values.
        num_bins: Number of equal-frequency bins.

    Returns:
        dict with keys:
        - 'bin_centers': [num_bins] mean U per bin.
        - 'bin_mses': [num_bins] mean MSE per bin.
        - 'bin_errors': [num_bins] SEM per bin.
        - 'slope': linear trend slope.
        - 'pearson_r': overall Pearson r.
        - 'p_value': significance.
    """
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(uncertainties, percentiles)

    bin_centers, bin_mses, bin_errors = [], [], []
    for i in range(num_bins):
        mask = (uncertainties >= bin_edges[i]) & (uncertainties <= bin_edges[i + 1])
        if np.sum(mask) > 0:
            bin_centers.append(np.mean(uncertainties[mask]))
            bin_mses.append(np.mean(mses[mask]))
            bin_errors.append(np.std(mses[mask]) / np.sqrt(np.sum(mask)))

    bin_centers = np.array(bin_centers)
    bin_mses = np.array(bin_mses)

    # Trend line slope
    if len(bin_centers) >= 2:
        z = np.polyfit(bin_centers, bin_mses, 1)
        slope = float(z[0])
    else:
        slope = 0.0

    pearson_r, p_value = stats.pearsonr(uncertainties, mses)

    return {
        "bin_centers": bin_centers,
        "bin_mses": bin_mses,
        "bin_errors": np.array(bin_errors),
        "slope": slope,
        "pearson_r": float(pearson_r),
        "p_value": float(p_value),
    }


def plot_calibration_curve(
    cal_results: dict,
    title: str = "Epistemic Calibration Curve",
    output_path: str = "experiments/results/calibration_curve.png",
    domain_label: str = "In-Domain",
) -> None:
    """
    Plot an epistemic calibration curve from precomputed bin statistics.
    """
    bin_centers = cal_results["bin_centers"]
    bin_mses = cal_results["bin_mses"]
    bin_errors = cal_results["bin_errors"]
    slope = cal_results["slope"]
    r = cal_results["pearson_r"]
    p = cal_results["p_value"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        bin_centers, bin_mses, yerr=bin_errors,
        fmt='-o', color='royalblue', ecolor='lightcoral',
        elinewidth=3, capsize=5, markerfacecolor='darkblue', markersize=8,
        label=f"{domain_label} (r={r:.3f}, p={p:.2e})"
    )

    # Trend line
    if len(bin_centers) >= 2:
        z = np.polyfit(bin_centers, bin_mses, 1)
        p_fn = np.poly1d(z)
        ax.plot(
            bin_centers, p_fn(bin_centers), "r--",
            alpha=0.8, linewidth=2,
            label=f"Trend (slope={slope:.4f})"
        )

    ax.set_xlabel("Predicted Trajectory Uncertainty ($U_{traj}$)", fontsize=14)
    ax.set_ylabel("Empirical Mean Squared Error (MSE)", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=12)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Calibration curve saved to {output_path}")


def assess_calibration(cal_results: dict) -> str:
    """
    Assess calibration quality from slope and correlation.

    Returns:
        'GOOD', 'MARGINAL', or 'POOR' calibration verdict.
    """
    slope = cal_results["slope"]
    r = cal_results["pearson_r"]
    p = cal_results["p_value"]

    if slope > 0 and r > 0.2 and p < 0.05:
        return "GOOD"
    elif slope > 0 and r > 0.0:
        return "MARGINAL"
    else:
        return "POOR"


def run_calibration_analysis(
    uncertainties: np.ndarray,
    mses: np.ndarray,
    label: str = "Norman In-Domain",
    output_prefix: str = "experiments/results/calibration",
) -> dict:
    """
    End-to-end calibration analysis with plot and verdict.

    Args:
        uncertainties: [N] per-cell uncertainty.
        mses: [N] per-cell MSE.
        label: Name for this evaluation set.
        output_prefix: Output path prefix (no extension).

    Returns:
        Calibration result dict.
    """
    cal = compute_calibration_curve(uncertainties, mses)
    verdict = assess_calibration(cal)

    print(f"\nCalibration Analysis [{label}]")
    print(f"  Pearson r: {cal['pearson_r']:.4f}  (p={cal['p_value']:.2e})")
    print(f"  Calibration Slope (beta): {cal['slope']:.4f}")
    print(f"  Verdict: {verdict}")

    output_path = f"{output_prefix}_{label.lower().replace(' ', '_')}.png"
    plot_calibration_curve(cal, title=f"Calibration — {label}", output_path=output_path, domain_label=label)

    cal["verdict"] = verdict
    cal["label"] = label
    return cal
