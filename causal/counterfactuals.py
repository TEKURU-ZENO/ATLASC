"""
counterfactuals.py
------------------
Generate counterfactual cell states by asking:
  "What would gene expression look like if gene G had been perturbed differently?"

This uses the trained ensemble + rollout engine to simulate alternative
trajectories and compute the counterfactual delta.
"""

from __future__ import annotations

import torch
from typing import List, Optional, Tuple


def compute_counterfactual(
    ensemble,
    simulator,
    edge_index: torch.Tensor,
    x_factual: torch.Tensor,
    target_gene: int,
    factual_value: float,
    counterfactual_value: float,
    rollout_steps: int = 3,
    device: torch.device | str = "cpu",
) -> dict:
    """
    Compute the counterfactual effect of changing a single gene's expression.

    Args:
        ensemble            : EnsembleWrapper (eval mode)
        simulator           : RolloutEngine
        edge_index          : [2, E]  GRN topology
        x_factual           : [batch, genes]  actual cell states
        target_gene         : gene index to intervene on
        factual_value       : the actual measured value (for reference)
        counterfactual_value: the hypothetical intervention value
        rollout_steps       : number of GNN rollout steps
        device              : 'cpu' or 'cuda'

    Returns:
        dict with keys:
            'factual_trajectory'       : [T, K, batch, genes, latent]
            'counterfactual_trajectory': [T, K, batch, genes, latent]
            'delta_expression_mean'    : [batch, genes]  mean expression difference
            'uncertainty_delta'        : [batch, genes]  uncertainty in the delta
    """
    from causal.do_operator import apply_intervention

    device = torch.device(device) if isinstance(device, str) else device
    ensemble.eval()

    B = x_factual.size(0)
    x = x_factual.to(device)

    with torch.no_grad():
        # ------------------------------------------------------------------
        # Factual rollout
        # ------------------------------------------------------------------
        z0_k = ensemble.encode(x)   # [K, B, G, L]

        _, traj_factual = simulator.rollout(
            z0_k, edge_index, steps=rollout_steps, log_trajectory=True, enforce_stability=False
        )  # [T, K, B, G, L]

        # Decode mean expression at each step → average over T for a summary
        factual_expr = ensemble.decode(traj_factual.mean(dim=0))  # [K, B, G]
        factual_mean = factual_expr.mean(dim=0)                   # [B, G]

        # ------------------------------------------------------------------
        # Counterfactual rollout  (do(target_gene = counterfactual_value))
        # ------------------------------------------------------------------
        interventions = torch.tensor(
            [[target_gene, -1]] * B, dtype=torch.long, device=device
        )
        z0_cf_k = apply_intervention(z0_k, interventions, value=counterfactual_value)

        _, traj_cf = simulator.rollout(
            z0_cf_k, edge_index, steps=rollout_steps, log_trajectory=True, enforce_stability=False
        )

        cf_expr = ensemble.decode(traj_cf.mean(dim=0))    # [K, B, G]
        cf_mean = cf_expr.mean(dim=0)                     # [B, G]

        # ------------------------------------------------------------------
        # Delta + uncertainty in delta
        # ------------------------------------------------------------------
        delta_k = cf_expr - factual_expr                  # [K, B, G]
        delta_mean = delta_k.mean(dim=0)                  # [B, G]
        delta_var  = delta_k.var(dim=0, unbiased=True)    # [B, G]

    return {
        "factual_trajectory":        traj_factual,
        "counterfactual_trajectory": traj_cf,
        "factual_expression_mean":   factual_mean,
        "cf_expression_mean":        cf_mean,
        "delta_expression_mean":     delta_mean,
        "uncertainty_delta":         delta_var,
    }


def batch_counterfactuals(
    ensemble,
    simulator,
    edge_index: torch.Tensor,
    x_factual: torch.Tensor,
    candidate_genes: List[int],
    cf_value: float = 0.0,
    rollout_steps: int = 3,
    device: torch.device | str = "cpu",
) -> List[dict]:
    """
    Run compute_counterfactual for each gene in candidate_genes.

    Returns a list of result dicts (one per candidate).
    """
    results = []
    for gene in candidate_genes:
        result = compute_counterfactual(
            ensemble=ensemble,
            simulator=simulator,
            edge_index=edge_index,
            x_factual=x_factual,
            target_gene=gene,
            factual_value=None,         # not needed at the moment
            counterfactual_value=cf_value,
            rollout_steps=rollout_steps,
            device=device,
        )
        result["gene"] = gene
        results.append(result)
    return results
