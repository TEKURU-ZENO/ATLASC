"""
loss.py
-------
Custom loss functions for ATLAS GNN training.

Available losses:
  1. TrajectoryReconstructionLoss  – MSE over multi-step rollout
  2. EpistemicRegularizationLoss   – encourages ensemble diversity
  3. StructuralConsistencyLoss     – penalises predictions that violate
                                     known GRN edge directions
  4. CombinedATLASLoss             – weighted sum of all three
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# 1.  Trajectory Reconstruction Loss
# ---------------------------------------------------------------------------

class TrajectoryReconstructionLoss(nn.Module):
    """
    Mean-squared error between model predictions and ground-truth expression
    signals across all rollout steps, with optional temporal discount.

    Args:
        gamma   : temporal discount factor in [0, 1].
                  1.0 → equal weight at every step (standard MSE).
                  <1 → earlier steps weighted more.
        reduction: 'mean' | 'sum'
    """

    def __init__(self, gamma: float = 1.0, reduction: str = "mean"):
        super().__init__()
        assert 0.0 < gamma <= 1.0
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        predictions: torch.Tensor,   # [T, batch, genes]  or  [batch, genes]
        targets: torch.Tensor,        # same shape
    ) -> torch.Tensor:

        if predictions.dim() == 2:
            # Single-step case
            return F.mse_loss(predictions, targets, reduction=self.reduction)

        T = predictions.size(0)
        device = predictions.device
        weights = torch.tensor(
            [self.gamma ** (T - t - 1) for t in range(T)],
            dtype=torch.float32,
            device=device,
        ).view(-1, 1, 1)  # [T, 1, 1]

        per_step_mse = ((predictions - targets) ** 2) * weights
        loss = per_step_mse.mean() if self.reduction == "mean" else per_step_mse.sum()
        return loss


# ---------------------------------------------------------------------------
# 2.  Epistemic Regularisation Loss
# ---------------------------------------------------------------------------

class EpistemicRegularizationLoss(nn.Module):
    """
    Encourages diversity among K ensemble member predictions by maximising
    pairwise prediction disagreement (negative loss = push apart).

    We minimise the *negative* of pairwise MSE, i.e. we subtract this
    from the total loss (or equivalently, add it with a negative weight).

    Args:
        temperature: softens the disagreement to prevent collapse to
                     adversarially different models.
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, predictions_k: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions_k: [K, batch, genes]
        Returns:
            loss: scalar – negative mean pairwise disagreement
        """
        K = predictions_k.size(0)
        if K < 2:
            return torch.tensor(0.0, device=predictions_k.device)

        total = torch.tensor(0.0, device=predictions_k.device)
        pairs = 0
        for i in range(K):
            for j in range(i + 1, K):
                diff = (predictions_k[i] - predictions_k[j]) ** 2
                total = total + diff.mean() / self.temperature
                pairs += 1

        # Negative: minimising the loss → maximising disagreement
        return -(total / pairs)


# ---------------------------------------------------------------------------
# 3.  Structural Consistency Loss
# ---------------------------------------------------------------------------

class StructuralConsistencyLoss(nn.Module):
    """
    Penalises predictions that contradict known regulatory sign information.

    For each directed edge (src → dst) in the GRN, if we know the regulatory
    sign (activation = +1, repression = -1), the predicted expression change
    at dst should correlate with that sign.

    If no sign information is available, this loss is zero.

    Args:
        margin: hinge margin – predictions within margin are not penalised
    """

    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        delta_expr: torch.Tensor,     # [batch, genes]  – predicted expression change
        edge_index: torch.Tensor,     # [2, E]
        edge_signs: Optional[torch.Tensor] = None,  # [E]  values in {-1, +1}
    ) -> torch.Tensor:
        """
        If edge_signs is None, returns 0.
        """
        if edge_signs is None:
            return torch.tensor(0.0, device=delta_expr.device)

        src, dst = edge_index  # each [E]
        # Predicted change at destination: [batch, E]
        pred_change = delta_expr[:, dst]
        # Expected sign broadcast: [1, E] → [batch, E]
        expected_sign = edge_signs.float().unsqueeze(0)

        # Hinge: penalise when sign disagrees
        # Loss = max(0, margin - sign * pred_change)
        signed_pred = expected_sign * pred_change
        hinge = F.relu(self.margin - signed_pred)
        return hinge.mean()


# ---------------------------------------------------------------------------
# 4.  Combined ATLAS Loss
# ---------------------------------------------------------------------------

class CombinedATLASLoss(nn.Module):
    """
    Weighted combination:
        L = λ_rec * L_rec  +  λ_struct * L_struct  -  λ_epist * L_epist
    """

    def __init__(
        self,
        lambda_rec: float = 1.0,
        lambda_struct: float = 0.1,
        lambda_epist: float = 0.01,
        gamma: float = 1.0,
    ):
        super().__init__()
        self.lambda_rec = lambda_rec
        self.lambda_struct = lambda_struct
        self.lambda_epist = lambda_epist

        self.rec_loss = TrajectoryReconstructionLoss(gamma=gamma)
        self.epist_loss = EpistemicRegularizationLoss()
        self.struct_loss = StructuralConsistencyLoss()

    def forward(
        self,
        predictions: torch.Tensor,             # [batch, genes] or [T, batch, genes]
        targets: torch.Tensor,
        predictions_k: Optional[torch.Tensor] = None,   # [K, batch, genes]
        delta_expr: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_signs: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Returns:
            dict with keys: 'total', 'reconstruction', 'epistemic', 'structural'
        """
        l_rec = self.rec_loss(predictions, targets)

        l_epist = torch.tensor(0.0, device=predictions.device)
        if predictions_k is not None and self.lambda_epist > 0:
            l_epist = self.epist_loss(predictions_k)

        l_struct = torch.tensor(0.0, device=predictions.device)
        if delta_expr is not None and edge_index is not None and self.lambda_struct > 0:
            l_struct = self.struct_loss(delta_expr, edge_index, edge_signs)

        total = (
            self.lambda_rec * l_rec
            + self.lambda_struct * l_struct
            + self.lambda_epist * l_epist   # already negative internally
        )

        return {
            "total": total,
            "reconstruction": l_rec,
            "epistemic": l_epist,
            "structural": l_struct,
        }
