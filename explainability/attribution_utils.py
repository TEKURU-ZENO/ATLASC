"""
attribution_utils.py

Shared utility helpers for gradient-based and perturbation-based attribution
methods used across the ATLAS explainability module.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional


def compute_integrated_gradients(
    model_fn,
    baseline: torch.Tensor,
    input_tensor: torch.Tensor,
    target_dim: int,
    steps: int = 50,
) -> torch.Tensor:
    """
    Compute Integrated Gradients for a model function w.r.t. a scalar target output.

    Args:
        model_fn: Callable that takes an input tensor and returns a scalar or vector.
        baseline: [genes] reference baseline (typically zeros).
        input_tensor: [genes] the actual input.
        target_dim: Index of the output dimension to attribute.
        steps: Number of interpolation steps.

    Returns:
        attributions: [genes] integrated gradient attribution scores.
    """
    baseline = baseline.unsqueeze(0)  # [1, genes]
    input_tensor = input_tensor.unsqueeze(0)  # [1, genes]

    # Interpolated inputs along the straight line from baseline to input
    alphas = torch.linspace(0.0, 1.0, steps).to(input_tensor.device)
    interpolated = [baseline + alpha * (input_tensor - baseline) for alpha in alphas]
    interpolated = torch.cat(interpolated, dim=0)  # [steps, genes]
    interpolated.requires_grad_(True)

    outputs = model_fn(interpolated)  # [steps, output_dim]
    if outputs.dim() > 1:
        target_output = outputs[:, target_dim].sum()
    else:
        target_output = outputs.sum()

    target_output.backward()

    gradients = interpolated.grad  # [steps, genes]
    avg_gradients = gradients.mean(dim=0)  # [genes]
    attributions = (input_tensor.squeeze(0) - baseline.squeeze(0)) * avg_gradients
    return attributions.detach()


def compute_gradient_times_input(
    model_fn,
    input_tensor: torch.Tensor,
    target_dim: int,
) -> torch.Tensor:
    """
    Simple Gradient × Input saliency. Faster than IG, less smooth.

    Args:
        model_fn: Callable returning scalar or vector.
        input_tensor: [genes] input.
        target_dim: Target output index.

    Returns:
        saliency: [genes] attribution scores.
    """
    x = input_tensor.clone().detach().unsqueeze(0).requires_grad_(True)
    output = model_fn(x)
    if output.dim() > 1:
        scalar = output[0, target_dim]
    else:
        scalar = output[0]
    scalar.backward()
    saliency = (x.grad.squeeze(0) * x.squeeze(0)).detach()
    return saliency


def normalize_attributions(attributions: torch.Tensor) -> torch.Tensor:
    """Min-max normalize attributions to [0, 1]."""
    a_min = attributions.min()
    a_max = attributions.max()
    if (a_max - a_min).abs() < 1e-9:
        return torch.zeros_like(attributions)
    return (attributions - a_min) / (a_max - a_min)


def rank_genes_by_attribution(
    attributions: torch.Tensor,
    gene_names: List[str],
    top_k: int = 20,
) -> List[dict]:
    """
    Rank genes by absolute attribution magnitude.

    Args:
        attributions: [num_genes] attribution tensor.
        gene_names: List of gene name strings aligned to attribution dimensions.
        top_k: Number of top genes to return.

    Returns:
        List of dicts with keys 'gene', 'attribution', 'rank'.
    """
    abs_attr = attributions.abs()
    sorted_indices = torch.argsort(abs_attr, descending=True)
    results = []
    for rank, idx in enumerate(sorted_indices[:top_k]):
        results.append({
            "rank": rank + 1,
            "gene": gene_names[idx.item()],
            "attribution": attributions[idx.item()].item(),
            "abs_attribution": abs_attr[idx.item()].item(),
        })
    return results


def aggregate_attributions_over_cells(
    attr_list: List[torch.Tensor],
    method: str = "mean",
) -> torch.Tensor:
    """
    Aggregate per-cell attribution tensors into a single population-level score.

    Args:
        attr_list: List of [genes] tensors.
        method: 'mean', 'median', or 'sum'.

    Returns:
        [genes] aggregated attributions.
    """
    stacked = torch.stack(attr_list, dim=0)  # [N_cells, genes]
    if method == "mean":
        return stacked.mean(dim=0)
    elif method == "median":
        return stacked.median(dim=0).values
    elif method == "sum":
        return stacked.sum(dim=0)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
