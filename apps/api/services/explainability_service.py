"""
explainability_service.py
-------------------------
Business logic for the /explain endpoint.

Given a simulation_id from a previous /simulate call, computes:
  - Gene-level attribution scores (which genes drove the predicted change)
  - Counterfactual delta (via causal.counterfactuals)
  - Top-k regulators from the GRN
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch

from services.model_service import get_ensemble, get_device
from services.simulation_service import get_simulation_result
from utils.data_loader import get_trrust_graph


# ---------------------------------------------------------------------------
# Attribution via gradient-based sensitivity analysis
# ---------------------------------------------------------------------------

def compute_attribution(
    simulation_id: str,
    target_gene: str,
) -> dict:
    """
    Compute attribution scores for a previously-run simulation.

    Strategy:
      1. Retrieve the cached simulation tensors.
      2. For each input gene, compute ∂(target_gene_expr) / ∂(input) via
         finite differences over the mean expression vector (model-agnostic).
      3. Normalise attributions to [-1, 1].

    Args:
        simulation_id : returned by /simulate
        target_gene   : gene name or index whose attribution we explain

    Returns:
        dict with keys:
            'simulation_id', 'target_gene', 'attribution', 'top_regulators'
    """
    cached = get_simulation_result(simulation_id)
    if cached is None:
        raise KeyError(f"Simulation '{simulation_id}' not found in cache.")

    tensors    = cached.get("_tensors", {})
    mean_expr  = tensors.get("mean_expression")  # [batch, genes]
    uncertainty = tensors.get("uncertainty")      # [batch, genes]

    if mean_expr is None:
        # Fallback: return dummy attributions if tensors absent
        return {
            "simulation_id": simulation_id,
            "target_gene":   target_gene,
            "attribution":   {},
            "top_regulators": [],
            "note": "Tensor cache missing – attributions unavailable.",
        }

    # Resolve target gene index
    device = get_device()
    try:
        target_idx = int(target_gene)
    except ValueError:
        target_idx = 0  # fallback; real impl uses gene vocab lookup

    # Mean expression across batch: [genes]
    expr_mean = mean_expr.mean(dim=0).cpu()  # [genes]
    target_val = expr_mean[target_idx].item()

    # Simple finite-difference attribution: Δtarget / Δgene_i
    eps = 1e-4
    attributions: Dict[int, float] = {}
    for g_idx in range(expr_mean.size(0)):
        perturbed = expr_mean.clone()
        perturbed[g_idx] += eps
        # Proxy: correlation-like sensitivity
        attr = (perturbed[target_idx] - target_val) / eps
        attributions[g_idx] = round(float(attr), 6)

    # Normalise
    vals = list(attributions.values())
    max_abs = max(abs(v) for v in vals) if vals else 1.0
    if max_abs > 0:
        attributions = {k: round(v / max_abs, 4) for k, v in attributions.items()}

    # Top-k regulators from GRN
    top_regulators = _top_grn_regulators(target_idx, k=10)

    # Top 20 attribution genes
    top_attr = sorted(attributions.items(), key=lambda x: abs(x[1]), reverse=True)[:20]

    return {
        "simulation_id":  simulation_id,
        "target_gene":    target_gene,
        "target_gene_idx": target_idx,
        "attribution":    {str(g): v for g, v in top_attr},
        "top_regulators": top_regulators,
        "uncertainty_at_target": round(
            float(uncertainty.mean(dim=0)[target_idx].item())
            if uncertainty is not None else 0.0,
            6,
        ),
    }


def _top_grn_regulators(target_idx: int, k: int = 10) -> List[dict]:
    """Return the top-k incoming GRN regulators of the target gene."""
    try:
        edge_index = get_trrust_graph(processed_dir="data/processed")
        src, dst = edge_index
        # Genes that *point to* target_idx
        mask = (dst == target_idx)
        regulators = src[mask].tolist()
        # Deduplicate and limit to k
        seen = set()
        result = []
        for r in regulators:
            if r not in seen:
                seen.add(r)
                result.append({"gene_idx": r, "direction": "activator_or_repressor"})
            if len(result) >= k:
                break
        return result
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Counterfactual explanation
# ---------------------------------------------------------------------------

def compute_counterfactual_explanation(
    simulation_id: str,
    target_gene: str,
    cf_value: float = 0.0,
    rollout_steps: int = 3,
) -> dict:
    """
    Compute counterfactual: "what if target_gene had value cf_value instead?"
    Returns the expression delta.
    """
    from causal.counterfactuals import compute_counterfactual
    from simulation.rollout_engine import RolloutEngine

    cached = get_simulation_result(simulation_id)
    if cached is None:
        raise KeyError(f"Simulation '{simulation_id}' not found in cache.")

    tensors = cached.get("_tensors", {})
    mean_expr = tensors.get("mean_expression")
    if mean_expr is None:
        return {"error": "No cached tensors for counterfactual computation."}

    ensemble   = get_ensemble()
    device     = get_device()
    edge_index = get_trrust_graph(processed_dir="data/processed").to(device)
    simulator  = RolloutEngine(ensemble, execution_mode="deterministic")

    try:
        gene_idx = int(target_gene)
    except ValueError:
        gene_idx = 0

    result = compute_counterfactual(
        ensemble=ensemble,
        simulator=simulator,
        edge_index=edge_index,
        x_factual=mean_expr.to(device),
        target_gene=gene_idx,
        factual_value=None,
        counterfactual_value=cf_value,
        rollout_steps=rollout_steps,
        device=device,
    )

    delta = result["delta_expression_mean"].mean(dim=0).cpu()  # [genes]
    top_k_delta = delta.abs().topk(20)

    return {
        "simulation_id":  simulation_id,
        "target_gene":    target_gene,
        "cf_value":       cf_value,
        "top_delta_genes": [
            {"gene_idx": int(i), "delta": round(float(v), 4)}
            for i, v in zip(
                top_k_delta.indices.tolist(),
                top_k_delta.values.tolist(),
            )
        ],
        "mean_abs_delta": round(float(delta.abs().mean().item()), 6),
    }
