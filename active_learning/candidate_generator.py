"""
candidate_generator.py
----------------------
Generates candidate gene perturbations for active-learning iterations.

Strategies:
  1. AllGenes       – enumerate all genes as candidates (exhaustive)
  2. HighVariance   – select genes with highest transcriptomic variance
  3. HubGenes       – select high-degree nodes in the GRN
  4. RandomSample   – uniform random sample (baseline / ablation)
  5. Combined       – union of multiple strategies with deduplication
"""

from __future__ import annotations

import random
from typing import List, Optional, Set

import torch


# ---------------------------------------------------------------------------
# 1. All genes
# ---------------------------------------------------------------------------

def all_genes(num_genes: int) -> List[int]:
    """Return every gene index as a candidate."""
    return list(range(num_genes))


# ---------------------------------------------------------------------------
# 2. High-variance genes
# ---------------------------------------------------------------------------

def high_variance_genes(
    cell_data: torch.Tensor,
    top_k: int,
    exclude: Optional[Set[int]] = None,
) -> List[int]:
    """
    Select genes with the highest expression variance across cells.

    Args:
        cell_data : [n_cells, n_genes]  raw or normalised expression matrix
        top_k     : number of genes to select
        exclude   : set of already-selected gene indices to skip

    Returns:
        List of top-k gene indices sorted by descending variance.
    """
    exclude = exclude or set()
    var = cell_data.var(dim=0)                    # [n_genes]
    sorted_indices = var.argsort(descending=True).tolist()
    candidates = [g for g in sorted_indices if g not in exclude]
    return candidates[:top_k]


# ---------------------------------------------------------------------------
# 3. Hub genes (high GRN degree)
# ---------------------------------------------------------------------------

def hub_genes(
    edge_index: torch.Tensor,
    num_genes: int,
    top_k: int,
    exclude: Optional[Set[int]] = None,
) -> List[int]:
    """
    Select genes with the highest total degree (in + out) in the GRN.

    Args:
        edge_index : [2, E]  directed edge tensor
        num_genes  : total number of genes
        top_k      : number of genes to return
        exclude    : genes to skip

    Returns:
        List of top-k gene indices by degree, descending.
    """
    exclude = exclude or set()
    degree = torch.zeros(num_genes, dtype=torch.long)

    if edge_index.numel() > 0:
        src, dst = edge_index
        degree.scatter_add_(0, src, torch.ones_like(src))
        degree.scatter_add_(0, dst, torch.ones_like(dst))

    sorted_indices = degree.argsort(descending=True).tolist()
    candidates = [g for g in sorted_indices if g not in exclude]
    return candidates[:top_k]


# ---------------------------------------------------------------------------
# 4. Random sample
# ---------------------------------------------------------------------------

def random_sample(
    num_genes: int,
    k: int,
    exclude: Optional[Set[int]] = None,
    seed: Optional[int] = None,
) -> List[int]:
    """
    Uniform random sample of k gene indices.

    Args:
        num_genes : total number of genes
        k         : how many to sample
        exclude   : genes to skip
        seed      : optional RNG seed for reproducibility
    """
    exclude = exclude or set()
    pool = [g for g in range(num_genes) if g not in exclude]
    if seed is not None:
        random.seed(seed)
    k = min(k, len(pool))
    return random.sample(pool, k)


# ---------------------------------------------------------------------------
# 5. Combined strategy
# ---------------------------------------------------------------------------

def combined_candidates(
    num_genes: int,
    cell_data: Optional[torch.Tensor] = None,
    edge_index: Optional[torch.Tensor] = None,
    top_k_variance: int = 50,
    top_k_hub: int = 50,
    random_k: int = 20,
    exclude: Optional[Set[int]] = None,
    seed: Optional[int] = 42,
) -> List[int]:
    """
    Union of high-variance + hub + random candidates, deduplicated.

    Falls back gracefully if cell_data or edge_index is None.

    Returns:
        Deduplicated list of candidate gene indices.
    """
    exclude = exclude or set()
    candidates: Set[int] = set()

    if cell_data is not None:
        candidates.update(high_variance_genes(cell_data, top_k_variance, exclude))

    if edge_index is not None:
        candidates.update(hub_genes(edge_index, num_genes, top_k_hub, exclude))

    candidates.update(random_sample(num_genes, random_k, exclude | candidates, seed))

    return sorted(candidates - exclude)


# ---------------------------------------------------------------------------
# Convenience: filter already-tested genes
# ---------------------------------------------------------------------------

def filter_tested(
    candidates: List[int],
    tested: Set[int],
) -> List[int]:
    """Remove genes that have already been selected in previous iterations."""
    return [g for g in candidates if g not in tested]
