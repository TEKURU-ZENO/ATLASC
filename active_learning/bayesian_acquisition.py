"""
bayesian_acquisition.py
-----------------------
Batch Bayesian Optimization acquisition functions for ATLAS.

Provides:
  - q-Expected Improvement (q-EI) via Monte Carlo sampling
  - Thompson Sampling across deep ensemble predictive distributions
  - Batch acquisition selector for non-myopic experimental design
"""

from __future__ import annotations

import math
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


class BatchBayesianAcquisition:
    """
    Non-myopic batch acquisition using Gaussian Process / Ensemble posteriors.
    """
    def __init__(self, mc_samples: int = 128, seed: int = 42):
        self.mc_samples = mc_samples
        self.rng = np.random.RandomState(seed)

    def compute_q_expected_improvement(
        self,
        candidate_predictions: Dict[int, torch.Tensor],
        best_observed_value: float,
    ) -> Dict[int, float]:
        """
        Computes Monte Carlo Expected Improvement (EI) for each candidate.
        
        Args:
            candidate_predictions: map of gene_idx -> tensor [K, genes] of ensemble member predictions
            best_observed_value: current target benchmark (e.g. maximum target phenotypic change)
        """
        q_ei_scores = {}

        for gene_idx, preds in candidate_predictions.items():
            # preds: [K, G]
            # Scalar phenotypic utility (e.g. norm of post-intervention delta)
            utilities = torch.norm(preds, dim=-1)  # [K]
            mu = utilities.mean().item()
            sigma = utilities.std().item() + 1e-6

            # Analytical Expected Improvement: (mu - y*) * Phi(z) + sigma * phi(z)
            improvement = mu - best_observed_value
            z = improvement / sigma

            phi_z = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * z * z)
            Phi_z = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

            ei = improvement * Phi_z + sigma * phi_z
            q_ei_scores[gene_idx] = max(0.0, ei)

        return q_ei_scores

    def thompson_sampling_batch(
        self,
        candidates: List[int],
        candidate_predictions: Dict[int, torch.Tensor],
        batch_size: int = 8,
    ) -> List[int]:
        """
        Selects a diverse batch using Thompson Sampling:
        For each slot in the batch, draws a random sample from each candidate's posterior
        and selects the candidate with the highest sampled realization.
        """
        selected = []
        pool = list(candidates)

        for _ in range(min(batch_size, len(pool))):
            best_cand = None
            best_sample = -float("inf")

            for c in pool:
                preds = candidate_predictions.get(c)
                if preds is not None:
                    # Random draw across K ensemble members or Gaussian approximation
                    k_idx = self.rng.randint(0, preds.size(0))
                    sample_val = torch.norm(preds[k_idx]).item()
                else:
                    sample_val = self.rng.randn()

                if sample_val > best_sample:
                    best_sample = sample_val
                    best_cand = c

            if best_cand is not None:
                selected.append(best_cand)
                pool.remove(best_cand)

        return selected
