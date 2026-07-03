"""
gene_attribution.py

Computes per-gene importance scores using gradient × input attribution on the
ATLAS ensemble. Identifies which genes most strongly drive predicted perturbation
outcomes for a given cell.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from explainability.attribution_utils import (
    compute_gradient_times_input,
    normalize_attributions,
    rank_genes_by_attribution,
    aggregate_attributions_over_cells,
)
from causal.do_operator import apply_intervention


def compute_gene_importance(
    encoder,
    gnn,
    x: torch.Tensor,
    intervention_indices: torch.Tensor,
    target_gene_idx: int,
    device: torch.device,
    method: str = "gradient_times_input",
) -> torch.Tensor:
    """
    Compute gene-level attribution scores for how much each input gene
    drives the predicted expression of a target gene after intervention.

    Args:
        encoder: TranscriptomicEncoder.
        gnn: CausalGNN model with decode().
        x: [num_genes] input basal state tensor.
        intervention_indices: [max_interventions] intervention tensor.
        target_gene_idx: Output gene index to attribute towards.
        device: torch device.
        method: 'gradient_times_input' or 'zeros_baseline'.

    Returns:
        attributions: [num_genes] tensor of importance scores.
    """
    encoder.eval()
    gnn.eval()
    x = x.to(device)

    def forward_fn(x_batch):
        z0 = encoder(x_batch)  # [batch, genes, latent]
        z_int = apply_intervention(z0, intervention_indices.unsqueeze(0), value=0.0)
        decoded = gnn.decode(z_int)  # [batch, genes]
        return decoded

    attributions = compute_gradient_times_input(
        model_fn=forward_fn,
        input_tensor=x,
        target_dim=target_gene_idx,
    )
    return attributions


def compute_population_gene_importance(
    encoder,
    gnn,
    dataloader,
    target_gene_idx: int,
    device: torch.device,
    max_cells: int = 200,
    aggregation: str = "mean",
) -> torch.Tensor:
    """
    Compute population-level gene attribution by averaging across many cells.

    Args:
        encoder: TranscriptomicEncoder.
        gnn: CausalGNN.
        dataloader: DataLoader yielding (x, interventions, targets) batches.
        target_gene_idx: Output gene to attribute towards.
        device: torch device.
        max_cells: Max cells to compute over for speed.
        aggregation: 'mean', 'median', or 'sum'.

    Returns:
        [num_genes] population-level importance tensor.
    """
    all_attrs = []
    cell_count = 0

    for x_batch, int_batch, _ in dataloader:
        for i in range(x_batch.size(0)):
            if cell_count >= max_cells:
                break
            xi = x_batch[i].to(device)
            inti = int_batch[i].to(device)
            attr = compute_gene_importance(
                encoder, gnn, xi, inti, target_gene_idx, device
            )
            all_attrs.append(attr.cpu())
            cell_count += 1
        if cell_count >= max_cells:
            break

    agg = aggregate_attributions_over_cells(all_attrs, method=aggregation)
    return normalize_attributions(agg)


def report_top_driver_genes(
    attributions: torch.Tensor,
    gene_names: List[str],
    top_k: int = 20,
) -> None:
    """
    Print a formatted table of the top driver genes for a prediction.

    Args:
        attributions: [num_genes] attribution tensor.
        gene_names: Gene name list.
        top_k: Number of genes to report.
    """
    ranked = rank_genes_by_attribution(attributions, gene_names, top_k=top_k)
    print(f"\n{'Rank':<6} {'Gene':<20} {'Attribution':>12} {'|Attribution|':>14}")
    print("-" * 56)
    for entry in ranked:
        print(f"{entry['rank']:<6} {entry['gene']:<20} {entry['attribution']:>12.5f} {entry['abs_attribution']:>14.5f}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from models.encoders.transcriptomic_encoder import TranscriptomicEncoder
    from models.gnn.causal_gnn import CausalGNN
    from utils.data_loader import NormanDataset

    device = torch.device('cpu')
    dataset = NormanDataset(processed_dir="data/processed")
    gene_names = torch.load("data/processed/gene_names.pt", weights_only=False)
    num_genes = dataset.num_genes

    encoder = TranscriptomicEncoder(num_genes=num_genes, latent_dim=16).to(device)
    gnn = CausalGNN(in_channels=16, hidden_channels=16, out_channels=1, layers_count=3).to(device)

    ckpt = torch.load("models/checkpoints/epistemic_member_k0.pt", map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    gnn.load_state_dict(ckpt['gnn_state_dict'])

    x0, ints0, _ = dataset[0]
    attr = compute_gene_importance(encoder, gnn, x0, ints0, target_gene_idx=0, device=device)
    report_top_driver_genes(attr, gene_names, top_k=15)
