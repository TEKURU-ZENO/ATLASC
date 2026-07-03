"""
intervention_utils.py
---------------------
Utility helpers for constructing and validating intervention tensors
before passing them to do_operator.apply_intervention.
"""

from __future__ import annotations

import torch
from typing import Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Intervention tensor construction
# ---------------------------------------------------------------------------

def build_intervention_tensor(
    gene_value_map: Dict[int, float],
    batch_size: int,
    device: torch.device | str = "cpu",
    max_interventions: Optional[int] = None,
) -> Tuple[torch.Tensor, List[float]]:
    """
    Convert a {gene_idx: value} mapping into the padded tensor format
    expected by apply_intervention.

    Args:
        gene_value_map   : {gene_index: intervention_value}
        batch_size       : number of cells in the batch
        device           : target device
        max_interventions: pad to this width; defaults to len(gene_value_map)

    Returns:
        gene_indices : LongTensor [batch_size, max_interventions]  (padded with -1)
        values       : list of float intervention values, aligned with columns
    """
    genes  = list(gene_value_map.keys())
    values = [gene_value_map[g] for g in genes]
    n = max_interventions or len(genes)

    # Pad if needed
    pad_n = n - len(genes)
    genes_padded  = genes  + [-1] * pad_n
    values_padded = values + [0.0] * pad_n

    gene_indices = torch.tensor(
        [genes_padded] * batch_size, dtype=torch.long, device=device
    )
    return gene_indices, values_padded


def apply_scalar_intervention(
    z: torch.Tensor,
    gene_idx: int,
    value: float,
    mode: str = "zero",
) -> torch.Tensor:
    """
    Apply a scalar intervention to a specific gene's latent slice.

    Modes:
        'zero'    : set the full latent vector to value  (default)
        'additive': add value to each dimension of the latent vector
        'scale'   : multiply each latent dimension by value

    Args:
        z        : [batch, genes, latent]  or  [K, batch, genes, latent]
        gene_idx : which gene to intervene on
        value    : the intervention magnitude
        mode     : intervention type

    Returns:
        z_modified (cloned, gradient-detached)
    """
    z_out = z.clone()
    if mode == "zero":
        z_out[..., gene_idx, :] = value
    elif mode == "additive":
        z_out[..., gene_idx, :] = z_out[..., gene_idx, :] + value
    elif mode == "scale":
        z_out[..., gene_idx, :] = z_out[..., gene_idx, :] * value
    else:
        raise ValueError(f"Invalid intervention mode: '{mode}'. Choose from 'zero', 'additive', 'scale'.")
    return z_out


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_gene_indices(
    gene_indices: Union[List[int], torch.Tensor],
    num_genes: int,
) -> None:
    """Raise ValueError if any gene index is out of range [0, num_genes)."""
    if isinstance(gene_indices, torch.Tensor):
        gene_indices = gene_indices.tolist()

    invalid = [g for g in gene_indices if g != -1 and not (0 <= g < num_genes)]
    if invalid:
        raise ValueError(
            f"Gene indices out of range [0, {num_genes}): {invalid}"
        )


def is_valid_intervention(
    intervention: dict,
    num_genes: int,
) -> bool:
    """
    Lightweight check that an intervention dict (from the API) is valid.

    Expected format: {"gene_idx": int, "value": float}  or
                     {"gene_idxs": [int, ...], "values": [float, ...]}
    """
    if "gene_idx" in intervention:
        g = intervention["gene_idx"]
        return isinstance(g, int) and 0 <= g < num_genes

    if "gene_idxs" in intervention:
        for g in intervention["gene_idxs"]:
            if not (isinstance(g, int) and 0 <= g < num_genes):
                return False
        return True

    return False


# ---------------------------------------------------------------------------
# Perturbation combinatorics
# ---------------------------------------------------------------------------

def enumerate_single_gene_interventions(
    candidate_genes: List[int],
    value: float = 0.0,
) -> List[Dict[int, float]]:
    """Return one single-gene intervention dict per candidate gene."""
    return [{g: value} for g in candidate_genes]


def enumerate_pairwise_interventions(
    candidate_genes: List[int],
    value: float = 0.0,
) -> List[Dict[int, float]]:
    """
    Return all unique pairs of genes as simultaneous interventions.
    Warning: O(n²) – use only for small candidate pools.
    """
    pairs = []
    for i, g1 in enumerate(candidate_genes):
        for g2 in candidate_genes[i + 1 :]:
            pairs.append({g1: value, g2: value})
    return pairs
