"""
experiment_ranker.py
--------------------
Unified Experiment Ranker supporting multiple operational modes:
  1. DISCOVERY: Epistemic trajectory uncertainty + Diversity + Hub regularizer
  2. SYNTHETIC_LETHALITY: Prioritizes targets inducing selective phenotypic disruption
  3. BAYESIAN_OPTIMIZATION: Upper Confidence Bound (UCB) and Thompson Sampling on GRN latent manifold
  4. COST_AWARE: Normalizes information value by assay / reagent cost
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple, Union
import torch
import numpy as np

from .acquisition import normalize_scores
from .budget_constraints import ExperimentCost


class UnifiedExperimentRanker:
    """
    Ranks active learning candidate perturbations across diverse biological objectives.
    """
    def __init__(
        self,
        cost_model: Optional[ExperimentCost] = None,
        beta_ucb: float = 2.0,
    ):
        self.cost_model = cost_model or ExperimentCost()
        self.beta_ucb = beta_ucb

    def rank_discovery(
        self,
        candidates: List[int],
        u_traj_raw: Dict[int, float],
        diversity_scores: Dict[int, float],
        hub_penalties: Dict[int, float],
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        cost_aware: bool = False,
    ) -> List[Tuple[int, float, Dict[str, float]]]:
        """
        Ranks candidates under the ATLAS full exploration objective.
        Score = U_norm + lambda1 * Div_norm + lambda2 * Hub_norm [/ Cost]
        """
        u_norm = normalize_scores(u_traj_raw)
        div_norm = normalize_scores(diversity_scores)
        hub_norm = normalize_scores(hub_penalties)

        ranked = []
        for c in candidates:
            score = u_norm.get(c, 0.0) + lambda1 * div_norm.get(c, 0.0) + lambda2 * hub_norm.get(c, 0.0)
            cost = self.cost_model.estimate(1)
            if cost_aware:
                score = score / max(cost, 1e-4)
            breakdown = {
                "u_norm": u_norm.get(c, 0.0),
                "div_norm": div_norm.get(c, 0.0),
                "hub_norm": hub_norm.get(c, 0.0),
                "cost": cost,
            }
            ranked.append((c, score, breakdown))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def rank_synthetic_lethality(
        self,
        candidates: List[int],
        predicted_deltas: Dict[int, torch.Tensor],
        essential_gene_indices: List[int],
        u_traj_raw: Dict[int, float],
        lethality_weight: float = 2.0,
    ) -> List[Tuple[int, float, Dict[str, float]]]:
        """
        Ranks candidate interventions by their capacity to induce selective phenotypic collapse
        (downregulation of essential survival genes) combined with epistemic confidence.
        """
        ranked = []
        u_norm = normalize_scores(u_traj_raw)

        for c in candidates:
            delta = predicted_deltas.get(c)
            if delta is not None and len(essential_gene_indices) > 0:
                # Essential gene suppression: negative delta on essential genes indicates lethality
                suppression = -delta[essential_gene_indices].mean().item()
            else:
                suppression = 0.0

            # Favor high suppression with calibrated epistemic uncertainty
            score = lethality_weight * suppression + u_norm.get(c, 0.0)
            breakdown = {
                "essential_suppression": suppression,
                "uncertainty_norm": u_norm.get(c, 0.0),
            }
            ranked.append((c, score, breakdown))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def rank_bayesian_optimization_ucb(
        self,
        candidates: List[int],
        predicted_means: Dict[int, float],
        predicted_stds: Dict[int, float],
    ) -> List[Tuple[int, float, Dict[str, float]]]:
        """
        Upper Confidence Bound (UCB) non-myopic acquisition:
            alpha_UCB(x) = mu(x) + beta * sigma(x)
        """
        ranked = []
        for c in candidates:
            mu = predicted_means.get(c, 0.0)
            sigma = predicted_stds.get(c, 1.0)
            score = mu + self.beta_ucb * sigma
            breakdown = {"mean": mu, "std": sigma, "beta": self.beta_ucb}
            ranked.append((c, score, breakdown))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked


def rank_candidates(candidates: List[int], model=None, **kwargs) -> List[int]:
    """Compatibility interface returning ranked candidate indices."""
    ranker = UnifiedExperimentRanker()
    # Default fallback: return candidates in incoming order or scored if metrics passed
    if "u_traj_raw" in kwargs:
        ranked = ranker.rank_discovery(
            candidates, kwargs["u_traj_raw"],
            kwargs.get("diversity_scores", {c: 0.0 for c in candidates}),
            kwargs.get("hub_penalties", {c: 0.0 for c in candidates})
        )
        return [r[0] for r in ranked]
    return candidates
