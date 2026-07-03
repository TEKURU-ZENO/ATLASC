"""
simulator.py
------------
High-level simulation interface that wraps the lower-level RolloutEngine.

Provides:
  - Simulator.run()       : run a full perturbation simulation, decode output
  - Simulator.run_many()  : batch simulate multiple perturbations
  - Simulator.perturb_and_compare() : overlay factual vs perturbed trajectory

This module is what the API service layer calls, so it hides all the
latent-space internals and returns decoded gene-expression profiles.
"""

from __future__ import annotations

import torch
from typing import Dict, List, Optional, Union

from simulation.rollout_engine import RolloutEngine
from causal.do_operator import apply_intervention


class Simulator:
    """
    Stateful simulation wrapper.

    Args:
        ensemble   : EnsembleWrapper (encoders + GNNs)
        edge_index : [2, E] GRN edge tensor (on target device)
        device     : 'cpu' or 'cuda'
        rollout_steps : default number of simulation steps
    """

    def __init__(
        self,
        ensemble,
        edge_index: torch.Tensor,
        device: Union[torch.device, str] = "cpu",
        rollout_steps: int = 5,
    ):
        self.ensemble = ensemble
        self.edge_index = edge_index
        self.device = torch.device(device) if isinstance(device, str) else device
        self.rollout_steps = rollout_steps

        self._engine = RolloutEngine(ensemble, execution_mode="deterministic")

    # ------------------------------------------------------------------
    # Single perturbation simulation
    # ------------------------------------------------------------------

    def run(
        self,
        x: torch.Tensor,
        perturbations: Dict[int, float],
        steps: Optional[int] = None,
    ) -> dict:
        """
        Simulate the effect of one or more gene knockouts/overexpressions.

        Args:
            x             : [batch, genes]  cell expression matrix
            perturbations : {gene_idx: value}  e.g. {42: 0.0}
            steps         : override default rollout_steps

        Returns:
            dict:
                'mean_expression'  : [batch, genes]  posterior mean
                'uncertainty'      : [batch, genes]  epistemic variance
                'trajectory_mean'  : [T, batch, genes]  decoded at each step
                'steps_run'        : int
        """
        steps = steps or self.rollout_steps
        x = x.to(self.device)

        self.ensemble.eval()
        with torch.no_grad():
            # Encode all K models
            z0_k = self.ensemble.encode(x)   # [K, B, G, L]

            # Apply each perturbation
            for gene_idx, value in perturbations.items():
                interventions = torch.tensor(
                    [[gene_idx, -1]] * x.size(0), dtype=torch.long, device=self.device
                )
                z0_k = apply_intervention(z0_k, interventions, value=value)

            # Roll out
            _, trajectory = self._engine.rollout(
                z0_k, self.edge_index, steps=steps,
                log_trajectory=True, enforce_stability=True
            )
            # trajectory: [T, K, B, G, L]

            # Decode
            K, B, G = z0_k.shape[0], x.size(0), x.size(1)
            T = trajectory.shape[0]

            decoded_steps = []
            for t in range(T):
                # [K, B, G, L] -> decode -> [K, B, G]
                expr_k = self.ensemble.decode(trajectory[t])
                decoded_steps.append(expr_k)

            traj_expr = torch.stack(decoded_steps, dim=0)   # [T, K, B, G]
            mean_expr = traj_expr.mean(dim=0).mean(dim=0)   # [B, G]
            epist_var = traj_expr.mean(dim=0).var(dim=0, unbiased=True)  # [B, G]  approx

        return {
            "mean_expression": mean_expr,
            "uncertainty":     epist_var,
            "trajectory_mean": traj_expr.mean(dim=1),   # [T, B, G]
            "steps_run":       T,
        }

    # ------------------------------------------------------------------
    # Batch simulation (many perturbations)
    # ------------------------------------------------------------------

    def run_many(
        self,
        x: torch.Tensor,
        perturbation_list: List[Dict[int, float]],
        steps: Optional[int] = None,
    ) -> List[dict]:
        """
        Run run() for each set of perturbations in perturbation_list.

        Returns:
            List of result dicts, one per perturbation set.
        """
        return [self.run(x, p, steps) for p in perturbation_list]

    # ------------------------------------------------------------------
    # Factual vs perturbed comparison
    # ------------------------------------------------------------------

    def perturb_and_compare(
        self,
        x: torch.Tensor,
        perturbations: Dict[int, float],
        steps: Optional[int] = None,
    ) -> dict:
        """
        Run the factual (unperturbed) and perturbed simulations and return
        the comparison.

        Returns:
            dict:
                'factual'      : result dict from run() with empty perturbations
                'perturbed'    : result dict from run() with perturbations
                'delta_mean'   : [batch, genes]  mean expression difference
        """
        factual   = self.run(x, {}, steps)
        perturbed = self.run(x, perturbations, steps)

        delta = perturbed["mean_expression"] - factual["mean_expression"]
        return {
            "factual":   factual,
            "perturbed": perturbed,
            "delta_mean": delta,
        }
