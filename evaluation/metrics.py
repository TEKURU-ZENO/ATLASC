"""
metrics.py

Core evaluation metrics for ATLAS perturbation predictions.
Provides reusable functions for MSE, correlation, and uncertainty-error calibration.
"""

import torch
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from typing import Tuple, Dict, Optional


def compute_mse_per_cell(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    gene_mask: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """
    Per-cell Mean Squared Error.

    Args:
        predictions: [N, genes] predicted expression.
        targets: [N, genes] target expression.
        gene_mask: Optional boolean mask [genes] to restrict evaluation.

    Returns:
        [N] array of per-cell MSE values.
    """
    if gene_mask is not None:
        predictions = predictions[:, gene_mask]
        targets = targets[:, gene_mask]
    return ((predictions - targets) ** 2).mean(dim=-1).cpu().numpy()


def compute_pearson_correlation(
    uncertainty: np.ndarray,
    mse: np.ndarray,
) -> Tuple[float, float]:
    """
    Pearson correlation between epistemic uncertainty and empirical MSE.

    Args:
        uncertainty: [N] U_traj values.
        mse: [N] MSE values.

    Returns:
        (r, p_value) Pearson r and two-tailed p-value.
    """
    r, p = stats.pearsonr(uncertainty, mse)
    return float(r), float(p)


def compute_spearman_correlation(
    uncertainty: np.ndarray,
    mse: np.ndarray,
) -> Tuple[float, float]:
    """
    Spearman rank correlation between U_traj and MSE.
    """
    r, p = stats.spearmanr(uncertainty, mse)
    return float(r), float(p)


def compute_calibration_slope(
    uncertainty: np.ndarray,
    mse: np.ndarray,
) -> Tuple[float, float]:
    """
    Fit linear regression: MSE = alpha + beta * U_traj.

    Args:
        uncertainty: [N] epistemic uncertainty values.
        mse: [N] empirical MSE values.

    Returns:
        (beta, alpha) — slope and intercept of linear calibration.
    """
    X = uncertainty.reshape(-1, 1)
    y = mse.reshape(-1, 1)
    reg = LinearRegression().fit(X, y)
    beta = float(reg.coef_[0][0])
    alpha = float(reg.intercept_[0])
    return beta, alpha


def compute_delta_metrics(
    u_source: np.ndarray,
    u_target: np.ndarray,
    mse_source: np.ndarray,
    mse_target: np.ndarray,
) -> Dict[str, float]:
    """
    Compute ΔMSE and ΔU between source (Norman) and target (Dixit) domains.

    Returns:
        dict with keys: 'delta_mse', 'delta_u', 'mean_mse_source',
                        'mean_mse_target', 'mean_u_source', 'mean_u_target'.
    """
    return {
        "mean_mse_source": float(mse_source.mean()),
        "mean_mse_target": float(mse_target.mean()),
        "delta_mse": float(mse_target.mean() - mse_source.mean()),
        "mean_u_source": float(u_source.mean()),
        "mean_u_target": float(u_target.mean()),
        "delta_u": float(u_target.mean() - u_source.mean()),
    }


def compute_r2_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    gene_mask: Optional[torch.Tensor] = None,
) -> float:
    """
    Compute global R² (coefficient of determination) for predictions vs targets.
    """
    if gene_mask is not None:
        predictions = predictions[:, gene_mask]
        targets = targets[:, gene_mask]
    y = targets.cpu().numpy().flatten()
    y_hat = predictions.cpu().numpy().flatten()
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def print_metrics_summary(metrics: Dict[str, float]) -> None:
    """Print a formatted metrics table."""
    print(f"\n{'Metric':<35} {'Value':>12}")
    print("-" * 50)
    for k, v in metrics.items():
        print(f"{k:<35} {v:>12.5f}")
