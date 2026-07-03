"""
trajectory_analysis.py

Analysis of ATLAS predicted gene expression trajectories.

Provides tools to:
- Compute gene expression fold change from baseline to final predicted state.
- Identify top differentially expressed genes between conditions.
- Summarize trajectory smoothness (latent norm over time).
- Compare predicted vs actual expression across timepoints.
"""

import os
import sys
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def compute_fold_change(
    predictions: torch.Tensor,
    baseline: torch.Tensor,
    log_scale: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute per-gene fold change: predicted vs baseline expression.

    Args:
        predictions: [N, genes] predicted post-perturbation expression.
        baseline: [genes] or [N, genes] baseline (unperturbed) expression.
        log_scale: If True, return log2 fold change (appropriate if data is log-normalized).
        eps: Small constant to avoid div/log(0).

    Returns:
        [N, genes] fold change tensor.
    """
    if baseline.dim() == 1:
        baseline = baseline.unsqueeze(0)
    if log_scale:
        # If data is already log1p,  ΔFC = pred - baseline
        fc = predictions - baseline
    else:
        fc = (predictions + eps) / (baseline + eps)
    return fc


def compute_top_deg(
    fold_changes: torch.Tensor,
    gene_names: List[str],
    top_k: int = 20,
    direction: str = "both",
) -> List[dict]:
    """
    Identify top differentially expressed genes.

    Args:
        fold_changes: [N, genes] fold change per cell per gene.
        gene_names: List of gene name strings.
        top_k: Number of top DEGs to return.
        direction: 'up', 'down', or 'both'.

    Returns:
        List of dicts with 'gene', 'mean_fc', 'std_fc', 'direction'.
    """
    mean_fc = fold_changes.mean(dim=0).cpu().numpy()  # [genes]
    std_fc = fold_changes.std(dim=0).cpu().numpy()     # [genes]

    if direction == "up":
        indices = np.argsort(mean_fc)[::-1][:top_k]
    elif direction == "down":
        indices = np.argsort(mean_fc)[:top_k]
    else:
        indices = np.argsort(np.abs(mean_fc))[::-1][:top_k]

    results = []
    for i, idx in enumerate(indices):
        results.append({
            "rank": i + 1,
            "gene": gene_names[idx] if idx < len(gene_names) else f"Gene_{idx}",
            "mean_fc": float(mean_fc[idx]),
            "std_fc": float(std_fc[idx]),
            "direction": "up" if mean_fc[idx] > 0 else "down",
        })
    return results


def compute_trajectory_smoothness(
    trajectory: torch.Tensor,
) -> np.ndarray:
    """
    Measure the L2 norm of latent state vectors at each rollout timestep.

    Args:
        trajectory: [T, K, batch, genes, latent] rollout trajectory tensor.

    Returns:
        [T] array of mean latent L2 norm across K, batch, and genes.
    """
    T = trajectory.size(0)
    norms = []
    for t in range(T):
        z_t = trajectory[t]  # [K, batch, genes, latent]
        norm_t = torch.norm(z_t, dim=-1).mean().item()
        norms.append(norm_t)
    return np.array(norms)


def compute_prediction_error_per_gene(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    gene_mask: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """
    Compute per-gene MSE across all cells.

    Args:
        predictions: [N, genes] predicted values.
        targets: [N, genes] ground truth values.
        gene_mask: Optional [genes] boolean mask.

    Returns:
        [genes] or [masked_genes] array of per-gene MSE.
    """
    if gene_mask is not None:
        predictions = predictions[:, gene_mask]
        targets = targets[:, gene_mask]
    gene_mse = ((predictions - targets) ** 2).mean(dim=0).cpu().numpy()
    return gene_mse


def report_top_deg(
    fold_changes: torch.Tensor,
    gene_names: List[str],
    top_k: int = 15,
) -> None:
    """Print a formatted top DEG report."""
    degs = compute_top_deg(fold_changes, gene_names, top_k=top_k)
    print(f"\n{'Rank':<6} {'Gene':<20} {'Mean FC':>10} {'Std FC':>10} {'Dir':>6}")
    print("-" * 58)
    for d in degs:
        arrow = "↑" if d["direction"] == "up" else "↓"
        print(f"{d['rank']:<6} {d['gene']:<20} {d['mean_fc']:>10.4f} {d['std_fc']:>10.4f} {arrow:>6}")


def summarize_trajectory(
    trajectory: torch.Tensor,
    ensemble,
    targets: torch.Tensor,
    gene_names: List[str],
    top_k_deg: int = 10,
) -> dict:
    """
    High-level summary of a rollout trajectory.

    Returns:
        dict with 'norms_per_t', 'top_degs', 'per_gene_mse'.
    """
    norms = compute_trajectory_smoothness(trajectory)  # [T]

    # Final step predictions
    final_z_k = trajectory[-1]  # [K, batch, genes, latent]
    preds_k = ensemble.decode(final_z_k)  # [K, batch, genes]
    mean_preds = preds_k.mean(dim=0)      # [batch, genes]

    baseline = mean_preds.mean(dim=0)     # [genes] — mean pred as baseline
    fc = compute_fold_change(mean_preds, baseline, log_scale=True)

    degs = compute_top_deg(fc, gene_names, top_k=top_k_deg)
    per_gene_mse = compute_prediction_error_per_gene(mean_preds, targets)

    print(f"\nTrajectory Latent Norms per Timestep: {[round(n, 3) for n in norms.tolist()]}")
    print("\nTop Differentially Expressed Genes (Predicted):")
    report_top_deg(fc, gene_names, top_k=top_k_deg)

    return {
        "norms_per_t": norms,
        "top_degs": degs,
        "per_gene_mse": per_gene_mse,
    }
