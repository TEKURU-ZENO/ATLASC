"""
visualization_utils.py

Shared plotting utilities for ATLAS evaluation and explainability outputs.

All plots use Agg backend (non-interactive) and save to disk.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Optional, Dict


def plot_uncertainty_vs_mse(
    uncertainties: np.ndarray,
    mses: np.ndarray,
    title: str = "Uncertainty vs MSE",
    output_path: str = "experiments/results/u_vs_mse.png",
    color: str = "royalblue",
    alpha: float = 0.4,
) -> None:
    """
    Scatter plot of per-cell U_traj vs per-cell MSE with a trend line.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(uncertainties, mses, alpha=alpha, s=8, color=color, label="Cells")

    z = np.polyfit(uncertainties, mses, 1)
    p = np.poly1d(z)
    x_line = np.linspace(uncertainties.min(), uncertainties.max(), 200)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2,
            label=f"Trend (slope={z[0]:.4f})")

    ax.set_xlabel("Epistemic Uncertainty (U_traj)", fontsize=13)
    ax.set_ylabel("Empirical MSE", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    _save(output_path)


def plot_gene_importance_bar(
    attributions: np.ndarray,
    gene_names: List[str],
    top_k: int = 20,
    title: str = "Top Driver Genes",
    output_path: str = "experiments/results/gene_importance.png",
) -> None:
    """
    Horizontal bar chart of top-k genes by attribution magnitude.
    """
    abs_attr = np.abs(attributions)
    sorted_idx = np.argsort(abs_attr)[::-1][:top_k]
    top_genes = [gene_names[i] for i in sorted_idx]
    top_values = attributions[sorted_idx]

    colors = ['tomato' if v < 0 else 'steelblue' for v in top_values]

    top_values_rev = list(reversed(top_values.tolist()))
    top_genes_rev = list(reversed(top_genes))
    colors_rev = list(reversed(colors))

    fig, ax = plt.subplots(figsize=(9, max(5, top_k // 2)))
    y_pos = np.arange(len(top_genes))
    ax.barh(y_pos, top_values_rev, color=colors_rev, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_genes_rev, fontsize=10)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel("Attribution Score", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold')
    plt.tight_layout()
    _save(output_path)


def plot_pathway_importance_bar(
    pathway_scores: Dict[str, float],
    title: str = "Pathway-Level Attribution",
    output_path: str = "experiments/results/pathway_importance.png",
    top_k: int = 10,
) -> None:
    """
    Horizontal bar chart of top-k pathways by attribution magnitude.
    """
    sorted_pathways = sorted(pathway_scores.items(), key=lambda x: abs(x[1]), reverse=True)
    top_pathways = [item for i, item in enumerate(sorted_pathways) if i < top_k]
    names = [p[0] for p in top_pathways]
    values = [p[1] for p in top_pathways]
    colors = ['tomato' if v < 0 else 'mediumseagreen' for v in values]

    names_rev = list(reversed(names))
    values_rev = list(reversed(values))
    colors_rev = list(reversed(colors))

    fig, ax = plt.subplots(figsize=(10, max(5, top_k // 2)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values_rev, color=colors_rev, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_rev, fontsize=10)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel("Aggregated Attribution Score", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold')
    plt.tight_layout()
    _save(output_path)


def plot_latent_norm_trajectory(
    norms_per_t: np.ndarray,
    title: str = "Latent State Norm Over Rollout",
    output_path: str = "experiments/results/latent_norm_trajectory.png",
) -> None:
    """
    Line plot of mean latent L2 norm at each rollout timestep.
    """
    T = len(norms_per_t)
    timesteps = np.arange(1, T + 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(timesteps, norms_per_t, '-o', color='darkorange', linewidth=2, markersize=8)
    ax.set_xlabel("Rollout Timestep", fontsize=13)
    ax.set_ylabel("Mean Latent L2 Norm", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.set_xticks(timesteps)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    _save(output_path)


def plot_mse_distribution(
    mse_norman: np.ndarray,
    mse_dixit: np.ndarray,
    output_path: str = "experiments/results/mse_distribution.png",
) -> None:
    """
    Side-by-side KDE of MSE distributions for Norman (in-domain) and Dixit (OOD).
    """
    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=(9, 5))
    for data, label, color in [
        (mse_norman, "Norman (In-Domain)", "royalblue"),
        (mse_dixit, "Dixit (OOD)", "tomato"),
    ]:
        kde = gaussian_kde(data, bw_method='scott')
        x_range = np.linspace(data.min(), data.max(), 300)
        ax.plot(x_range, kde(x_range), linewidth=2, color=color, label=label)
        ax.fill_between(x_range, 0, kde(x_range), alpha=0.15, color=color)

    ax.set_xlabel("Per-Cell MSE", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title("MSE Distribution: In-Domain vs Out-of-Domain", fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    _save(output_path)


def plot_domain_shift_summary(
    metrics: Dict[str, float],
    output_path: str = "experiments/results/domain_shift_summary.png",
) -> None:
    """
    Simple bar chart showing key transfer metrics: ΔMSE, ΔU, corr, β slope.
    """
    keys = ["delta_mse", "delta_u", "pearson_r", "beta_slope"]
    labels = ["ΔMSE", "ΔU", "corr(U,MSE)", "β slope"]
    values = [metrics.get(k, 0.0) for k in keys]
    colors = ['steelblue' if v >= 0 else 'tomato' for v in values]

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(labels))
    ax.bar(x_pos, values, color=colors, edgecolor='black', linewidth=0.6, width=0.5)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Metric Value", fontsize=13)
    ax.set_title("Domain Shift Evaluation Summary", fontsize=15, fontweight='bold')
    plt.tight_layout()
    _save(output_path)


def _save(output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")
