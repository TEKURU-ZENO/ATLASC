"""
uncertainty.py
--------------
Standalone uncertainty quantification utilities for ATLAS.

Provides:
  - Aleatoric uncertainty via Monte-Carlo Dropout
  - Epistemic uncertainty via Deep Ensemble disagreement
  - Calibration helpers (Expected Calibration Error)
  - Confidence-interval construction
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# 1.  Monte-Carlo Dropout  (aleatoric proxy)
# ---------------------------------------------------------------------------

def mc_dropout_predict(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    n_samples: int = 30,
    dropout_p: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate predictive mean and aleatoric variance via MC-Dropout.

    The model must have at least one ``nn.Dropout`` module.  We temporarily
    force dropout to stay active in eval mode by patching it.

    Returns:
        mean:  [batch, genes]
        var:   [batch, genes]  (aleatoric variance)
    """
    # Patch all Dropout layers to stay open at p=dropout_p
    original_states: dict = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Dropout):
            original_states[name] = (m.p, m.training)
            m.p = dropout_p
            m.train()

    samples: List[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(n_samples):
            z = model.encode(x)           # [batch, genes, latent]
            z_out = model.forward(z, edge_index)  # after GNN
            pred = model.decode(z_out)    # [batch, genes]
            samples.append(pred)

    # Restore
    for name, m in model.named_modules():
        if isinstance(m, nn.Dropout) and name in original_states:
            m.p, m.training = original_states[name]

    stacked = torch.stack(samples, dim=0)   # [S, batch, genes]
    mean = stacked.mean(dim=0)
    var  = stacked.var(dim=0, unbiased=True)
    return mean, var


# ---------------------------------------------------------------------------
# 2.  Deep-Ensemble epistemic uncertainty
# ---------------------------------------------------------------------------

def ensemble_epistemic_variance(
    predictions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and variance across K ensemble members.

    Args:
        predictions: [K, batch, genes]

    Returns:
        mean: [batch, genes]
        epistemic_var: [batch, genes]
    """
    mean = predictions.mean(dim=0)
    epistemic_var = predictions.var(dim=0, unbiased=True)
    return mean, epistemic_var


def pairwise_disagreement(
    predictions: torch.Tensor,
) -> torch.Tensor:
    """
    Mean pairwise L2 disagreement across ensemble members.

    Args:
        predictions: [K, batch, genes]

    Returns:
        score: scalar Tensor
    """
    K = predictions.size(0)
    if K < 2:
        return torch.tensor(0.0, device=predictions.device)

    total = 0.0
    pairs = 0
    for i in range(K):
        for j in range(i + 1, K):
            diff = (predictions[i] - predictions[j]).pow(2).sum(-1).sqrt()
            total = total + diff.mean()
            pairs += 1

    return total / pairs


# ---------------------------------------------------------------------------
# 3.  Trajectory-Weighted Uncertainty  (from ensemble rollout)
# ---------------------------------------------------------------------------

def trajectory_uncertainty(
    trajectory: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Time-weighted epistemic uncertainty from a full rollout trajectory.

    Args:
        trajectory: [T, K, batch, genes, latent]

    Returns:
        U_traj:   [batch, genes]  cumulative time-weighted variance
        D_t:      [T]             mean pairwise disagreement per step
    """
    T, K, B, G, L = trajectory.shape
    device = trajectory.device

    # Variance across K at each timestep: [T, B, G, L]
    var_z = trajectory.var(dim=1, unbiased=True)
    # Sum over latent → [T, B, G]
    var_node = var_z.sum(dim=-1)

    # Time weights α_t = t/T
    alpha_t = torch.arange(1, T + 1, dtype=torch.float32, device=device) / T
    weighted = var_node * alpha_t.view(-1, 1, 1)
    U_traj = weighted.sum(dim=0)   # [B, G]

    # Pairwise disagreement per step
    D_t_list: List[float] = []
    for t in range(T):
        # trajectory[t] : [K, B, G, L] → decode to [K, B, G] approx via norm
        z_t = trajectory[t]            # [K, B, G, L]
        z_norm = z_t.norm(dim=-1)      # [K, B, G]
        D_t_list.append(pairwise_disagreement(z_norm).item())

    D_t = torch.tensor(D_t_list, device=device)
    return U_traj, D_t


# ---------------------------------------------------------------------------
# 4.  Calibration · Expected Calibration Error
# ---------------------------------------------------------------------------

def expected_calibration_error(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    variance: torch.Tensor,
    n_bins: int = 10,
) -> float:
    """
    Scalar ECE: how well predictive variance matches actual squared error.

    We bin samples by their predicted standard deviation percentile and
    compare each bin's mean variance to its empirical MSE.

    Args:
        y_pred:   [N] or [batch, genes] – predicted mean
        y_true:   same shape – ground truth
        variance: same shape – predictive variance (σ²)
        n_bins:   number of percentile bins

    Returns:
        ece: float
    """
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    variance = variance.reshape(-1)

    sq_error = (y_pred - y_true).pow(2)

    quantiles = torch.linspace(0.0, 1.0, n_bins + 1, device=y_pred.device)
    bin_edges = torch.quantile(variance, quantiles)

    ece = 0.0
    N = y_pred.numel()

    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        mask = (variance >= lo) & (variance <= hi)
        if mask.sum() == 0:
            continue
        mean_var = variance[mask].mean().item()
        mean_err = sq_error[mask].mean().item()
        weight = mask.float().sum().item() / N
        ece += weight * abs(mean_var - mean_err)

    return ece


# ---------------------------------------------------------------------------
# 5.  Confidence-interval construction
# ---------------------------------------------------------------------------

def gaussian_confidence_interval(
    mean: torch.Tensor,
    variance: torch.Tensor,
    z_score: float = 1.96,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric Gaussian CI:  mean ± z_score * std

    Returns:
        lower: same shape as mean
        upper: same shape as mean
    """
    std = variance.clamp(min=0.0).sqrt()
    lower = mean - z_score * std
    upper = mean + z_score * std
    return lower, upper


def coverage_probability(
    lower: torch.Tensor,
    upper: torch.Tensor,
    y_true: torch.Tensor,
) -> float:
    """
    Fraction of ground-truth values that fall within the predicted interval.
    Ideal value = the nominal confidence level (e.g. 0.95 for 95 % CI).
    """
    within = ((y_true >= lower) & (y_true <= upper)).float()
    return within.mean().item()
