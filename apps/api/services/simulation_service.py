"""
simulation_service.py
---------------------
Business logic for the /simulate endpoint.

Loads the ensemble + GRN, runs the Simulator, and returns a
serialisable result dict that the route can return as JSON.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Dict, List, Optional

import torch

from services.model_service import get_ensemble, get_device, get_num_genes, _PROJECT_ROOT
from utils.data_loader import get_trrust_graph
from simulation.simulator import Simulator
from causal.perturbation_parser import parse_perturbation, resolve_gene_names

# ---------------------------------------------------------------------------
# In-memory simulation result cache (keyed by simulation_id)
# ---------------------------------------------------------------------------
_SIM_CACHE: Dict[str, dict] = {}

# Gene vocabulary (populated lazily)
_GENE_VOCAB: Dict[str, int] = {}
_GENE_NAMES: list = []


def _get_gene_vocab() -> Dict[str, int]:
    """Build gene-index vocabulary from processed data (lazy, absolute path)."""
    global _GENE_VOCAB, _GENE_NAMES
    if _GENE_VOCAB:
        return _GENE_VOCAB
    names_path = os.path.join(_PROJECT_ROOT, "data", "processed", "gene_names.pt")
    if os.path.exists(names_path):
        import torch as _torch
        _GENE_NAMES = _torch.load(names_path, map_location="cpu", weights_only=False)
        _GENE_VOCAB = {name: idx for idx, name in enumerate(_GENE_NAMES)}
    return _GENE_VOCAB


def run_simulation(
    perturbation_specs: List[str],
    steps: int = 5,
    dataset_id: str = "norman",
) -> dict:
    """
    Execute a perturbation simulation.

    Args:
        perturbation_specs : list of string specs, e.g. ["TP53:0", "MYC:+1.5"]
        steps              : number of rollout steps
        dataset_id         : for logging/metadata only

    Returns:
        Serialisable result dict with simulation_id and expression summaries.
    """
    t0 = time.time()

    ensemble = get_ensemble()
    device   = get_device()
    num_genes = get_num_genes()

    # Build GRN with absolute path
    grn_dir = os.path.join(_PROJECT_ROOT, "data", "processed")
    edge_index = get_trrust_graph(processed_dir=grn_dir).to(device)

    # Parse perturbation specs -> gene indices
    parsed = []
    vocab  = _get_gene_vocab()
    for spec in perturbation_specs:
        try:
            items = parse_perturbation(spec)
            if vocab:
                items = resolve_gene_names(items, vocab)
            parsed.extend(items)
        except Exception as e:
            # Fallback: try to interpret as int:value
            try:
                gene_str, val_str = spec.split(":")
                parsed.append({"gene_idx": int(gene_str), "value": float(val_str), "mode": "zero"})
            except Exception:
                raise ValueError(f"Cannot parse perturbation spec '{spec}': {e}")

    perturbations: Dict[int, float] = {p["gene_idx"]: p["value"] for p in parsed}

    # Build basal state: use basal_state.pt (mean of controls, small) or zeros
    basal_path = os.path.join(_PROJECT_ROOT, "data", "processed", "basal_state.pt")
    if os.path.exists(basal_path):
        import torch as _torch
        basal = _torch.load(basal_path, map_location="cpu", weights_only=True)
        x = basal.unsqueeze(0).repeat(8, 1).to(device)  # [8, genes] batch
    else:
        x = torch.zeros(1, num_genes, device=device)

    simulator = Simulator(ensemble, edge_index, device=device, rollout_steps=steps)
    result    = simulator.run(x, perturbations, steps=steps)

    # Serialise tensors
    mean_expr     = result["mean_expression"].mean(dim=0).cpu().tolist()  # [genes]
    uncertainty   = result["uncertainty"].mean(dim=0).cpu().tolist()
    top_perturbed = _top_affected_genes(result["mean_expression"], k=20)

    sim_id = str(uuid.uuid4())[:8]
    payload = {
        "simulation_id":   sim_id,
        "status":          "completed",
        "dataset_id":      dataset_id,
        "steps":           result["steps_run"],
        "perturbations":   perturbations,
        "mean_expression_summary": {
            "top_affected_genes": top_perturbed,
            "total_genes":        num_genes,
        },
        "uncertainty_mean":  float(sum(uncertainty) / max(len(uncertainty), 1)),
        "elapsed_s":         round(time.time() - t0, 3),
    }
    _SIM_CACHE[sim_id] = {**payload, "_tensors": result}
    return payload


def get_simulation_result(simulation_id: str) -> Optional[dict]:
    """Retrieve a cached simulation result by ID."""
    return _SIM_CACHE.get(simulation_id)


def _top_affected_genes(mean_expr: torch.Tensor, k: int = 20) -> List[dict]:
    """Return the top-k most changed genes (by absolute expression mean)."""
    flat = mean_expr.mean(dim=0).abs()   # [genes]
    top_vals, top_idx = flat.topk(min(k, flat.size(0)))
    return [
        {"gene_idx": int(i), "abs_expression": round(float(v), 4)}
        for i, v in zip(top_idx.tolist(), top_vals.tolist())
    ]
