"""
root_cause.py

Epistemic root cause analysis for ATLAS.

When a cell has anomalously high uncertainty (U_traj), this module
identifies which input genes and GRN structural features are driving
that uncertainty spike — supporting downstream biological insight.
"""

import os
import sys
import torch
import numpy as np
from typing import List, Dict, Tuple
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.encoders.transcriptomic_encoder import TranscriptomicEncoder
from models.gnn.causal_gnn import CausalGNN
from models.ensemble import EnsembleWrapper
from causal.do_operator import apply_intervention
from utils.data_loader import NormanDataset, get_trrust_graph
from simulation.rollout_engine import RolloutEngine
from explainability.attribution_utils import (
    normalize_attributions,
    rank_genes_by_attribution,
    aggregate_attributions_over_cells,
)


def compute_uncertainty_per_cell(
    ensemble: EnsembleWrapper,
    simulator: RolloutEngine,
    dataloader: DataLoader,
    edge_index: torch.Tensor,
    device: torch.device,
    steps: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute epistemic uncertainty per cell across the dataset.

    Returns:
        u_all: [N_cells] array of U_traj values.
        mse_all: [N_cells] array of per-cell MSE values.
    """
    ensemble.eval()
    u_all, mse_all = [], []
    criterion = torch.nn.MSELoss(reduction="none")

    with torch.no_grad():
        for x, interventions, targets in dataloader:
            x = x.to(device)
            targets = targets.to(device)

            z0_k = ensemble.encode(x)  # [K, batch, genes, latent]
            z_int_k = torch.stack([
                apply_intervention(z0_k[k], interventions, value=0.0)
                for k in range(ensemble.k_models)
            ], dim=0)

            latent_preds_k, trajectory = simulator.rollout(
                z_int_k, edge_index, steps=steps, log_trajectory=True, enforce_stability=False
            )

            # Epistemic uncertainty
            u_traj, _ = ensemble.compute_epistemic_uncertainty(trajectory)  # [batch, genes]
            u_cell = u_traj.mean(dim=-1).cpu().numpy()  # [batch]

            # MSE
            preds_k = ensemble.decode(latent_preds_k)  # [K, batch, genes]
            pred_mean = preds_k.mean(dim=0)  # [batch, genes]
            mse_cell = criterion(pred_mean, targets).mean(dim=-1).cpu().numpy()

            u_all.extend(u_cell)
            mse_all.extend(mse_cell)

    return np.array(u_all), np.array(mse_all)


def identify_high_uncertainty_cells(
    u_all: np.ndarray,
    threshold_percentile: float = 95.0,
) -> np.ndarray:
    """
    Identify cell indices with U_traj above the given percentile.

    Args:
        u_all: [N] uncertainty array.
        threshold_percentile: Only keep cells with U above this percentile.

    Returns:
        Sorted array of high-uncertainty cell indices.
    """
    threshold = np.percentile(u_all, threshold_percentile)
    high_u_indices = np.where(u_all >= threshold)[0]
    print(f"High-uncertainty threshold ({threshold_percentile}th pctile): {threshold:.4f}")
    print(f"Identified {len(high_u_indices)} high-uncertainty cells.")
    return high_u_indices


def attribute_uncertainty_to_genes(
    encoder,
    ensemble: EnsembleWrapper,
    x: torch.Tensor,
    interventions: torch.Tensor,
    edge_index: torch.Tensor,
    device: torch.device,
    steps: int = 3,
) -> torch.Tensor:
    """
    Attribute a single cell's uncertainty to input genes via sensitivity analysis.

    Perturbs each input gene slightly and measures resulting uncertainty change.

    Args:
        encoder: Shared TranscriptomicEncoder.
        ensemble: EnsembleWrapper.
        x: [num_genes] input basal state.
        interventions: [max_interventions] intervention indices.
        edge_index: GRN edge_index.
        device: torch device.
        steps: Rollout steps.

    Returns:
        sensitivity: [num_genes] tensor — how much each input gene drives uncertainty.
    """
    from simulation.rollout_engine import RolloutEngine

    encoder.eval()
    ensemble.eval()
    x = x.to(device).unsqueeze(0)  # [1, num_genes]
    interventions = interventions.unsqueeze(0).to(device)

    simulator = RolloutEngine(core_model=ensemble, execution_mode="deterministic")

    def get_uncertainty(x_inp):
        with torch.no_grad():
            z0_k = ensemble.encode(x_inp)
            z_int_k = torch.stack([
                apply_intervention(z0_k[k], interventions, value=0.0)
                for k in range(ensemble.k_models)
            ], dim=0)
            _, trajectory = simulator.rollout(
                z_int_k, edge_index, steps=steps, log_trajectory=True, enforce_stability=False
            )
            u_traj, _ = ensemble.compute_epistemic_uncertainty(trajectory)
            return u_traj.mean().item()

    baseline_u = get_uncertainty(x)

    num_genes = x.size(-1)
    sensitivity = torch.zeros(num_genes)
    epsilon = 0.1

    for gene_idx in range(num_genes):
        x_perturbed = x.clone()
        x_perturbed[0, gene_idx] += epsilon
        perturbed_u = get_uncertainty(x_perturbed)
        sensitivity[gene_idx] = (perturbed_u - baseline_u) / epsilon

    return sensitivity


def run_root_cause_analysis(
    n_high_uncertainty_cells: int = 5,
    top_k_genes: int = 15,
):
    """
    Full pipeline: load ensemble, identify high-U cells, attribute uncertainty to genes.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Root Cause Analysis — ATLAS Epistemic Uncertainty\n")

    gene_names = torch.load("data/processed/gene_names.pt", weights_only=False)
    full_dataset = NormanDataset(processed_dir="data/processed")
    num_genes = full_dataset.num_genes

    edge_index = get_trrust_graph("data/processed").to(device)

    encoders, gnns = [], []
    for k in range(3):
        enc = TranscriptomicEncoder(num_genes=num_genes, latent_dim=16).to(device)
        gnn = CausalGNN(in_channels=16, hidden_channels=16, out_channels=1, layers_count=3).to(device)
        ckpt = torch.load(f"models/checkpoints/epistemic_member_k{k}.pt",
                          map_location=device, weights_only=True)
        enc.load_state_dict(ckpt['encoder_state_dict'])
        gnn.load_state_dict(ckpt['gnn_state_dict'])
        enc.eval(); gnn.eval()
        encoders.append(enc); gnns.append(gnn)

    ensemble = EnsembleWrapper(encoders, gnns)
    simulator = RolloutEngine(core_model=ensemble, execution_mode="deterministic")

    loader = DataLoader(full_dataset, batch_size=64, shuffle=False)
    u_all, mse_all = compute_uncertainty_per_cell(ensemble, simulator, loader, edge_index, device)

    high_u_idx = identify_high_uncertainty_cells(u_all, threshold_percentile=95.0)
    selected_idx = high_u_idx[:n_high_uncertainty_cells]

    print(f"\nAttributing uncertainty for top {len(selected_idx)} cells...\n")
    all_sensitivities = []

    for cell_i in selected_idx:
        x_i, int_i, _ = full_dataset[cell_i]
        sens = attribute_uncertainty_to_genes(
            encoders[0], ensemble, x_i, int_i, edge_index, device
        )
        all_sensitivities.append(sens)

        ranked = rank_genes_by_attribution(sens, gene_names, top_k=top_k_genes)
        print(f"[Cell {cell_i}] U={u_all[cell_i]:.4f} | MSE={mse_all[cell_i]:.4f}")
        print(f"  Top driver genes: {', '.join([r['gene'] for r in ranked[:5]])}\n")

    # Population level root cause
    pop_sensitivity = aggregate_attributions_over_cells(all_sensitivities, method="mean")
    pop_normalized = normalize_attributions(pop_sensitivity)

    print("\nPopulation-Level Root Cause Genes (High-U Cells):")
    pop_ranked = rank_genes_by_attribution(pop_normalized, gene_names, top_k=top_k_genes)
    print(f"{'Rank':<6} {'Gene':<20} {'Population Sensitivity':>22}")
    print("-" * 52)
    for r in pop_ranked:
        print(f"{r['rank']:<6} {r['gene']:<20} {r['abs_attribution']:>22.5f}")


if __name__ == "__main__":
    run_root_cause_analysis()
