"""
temporal_analysis.py

Analyzes how ATLAS predictions evolve across rollout timesteps T=1,2,...

Key outputs:
- Per-timestep mean gene expression change.
- Per-timestep pairwise ensemble disagreement (Δ_t).
- Identifies which timepoint drives the most epistemic divergence.
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.encoders.transcriptomic_encoder import TranscriptomicEncoder
from models.gnn.causal_gnn import CausalGNN
from models.ensemble import EnsembleWrapper
from causal.do_operator import apply_intervention
from utils.data_loader import NormanDataset, get_trrust_graph
from simulation.rollout_engine import RolloutEngine


def analyze_temporal_uncertainty(
    ensemble: EnsembleWrapper,
    simulator: RolloutEngine,
    dataloader: DataLoader,
    edge_index: torch.Tensor,
    device: torch.device,
    steps: int = 3,
    max_cells: int = 300,
) -> dict:
    """
    Collect per-timestep ensemble disagreement statistics over many cells.

    Returns:
        dict with keys:
        - 'd_t_mean': np.ndarray [T] — mean disagreement at each timestep.
        - 'd_t_std': np.ndarray [T] — std dev across cells.
        - 'expression_delta': np.ndarray [T, genes] — mean expression change per step.
    """
    ensemble.eval()
    d_t_all = []
    expr_deltas = [[] for _ in range(steps)]
    cell_count = 0

    with torch.no_grad():
        for x, interventions, _ in dataloader:
            if cell_count >= max_cells:
                break
            x = x.to(device)

            z0_k = ensemble.encode(x)  # [K, batch, genes, latent]
            z_int_k = torch.stack([
                apply_intervention(z0_k[k], interventions, value=0.0)
                for k in range(ensemble.k_models)
            ], dim=0)

            _, trajectory = simulator.rollout(
                z_int_k, edge_index, steps=steps,
                log_trajectory=True, enforce_stability=False
            )
            # trajectory: [T, K, batch, genes, latent]

            _, d_t = ensemble.compute_epistemic_uncertainty(trajectory)
            # d_t: [T] list of disagreement scalars

            d_t_all.append(d_t)

            # Expression delta at each timestep
            for t in range(steps):
                z_t = trajectory[t]  # [K, batch, genes, latent]
                preds_t = ensemble.decode(z_t)  # [K, batch, genes]
                mean_expr_t = preds_t.mean(dim=0)  # [batch, genes]
                delta = mean_expr_t.abs().mean(dim=0).cpu().numpy()  # [genes]
                expr_deltas[t].append(delta)

            cell_count += x.size(0)

    d_t_array = np.array(d_t_all)  # [batches, T]
    d_t_mean = d_t_array.mean(axis=0)
    d_t_std = d_t_array.std(axis=0)

    expr_delta_mean = np.array([
        np.stack(expr_deltas[t], axis=0).mean(axis=0)
        for t in range(steps)
    ])  # [T, genes]

    return {
        "d_t_mean": d_t_mean,
        "d_t_std": d_t_std,
        "expression_delta": expr_delta_mean,
    }


def plot_temporal_disagreement(
    results: dict,
    output_path: str = "experiments/results/temporal_disagreement.png",
) -> None:
    """
    Plot pairwise ensemble disagreement (Δ_t) across rollout timesteps.
    """
    d_t = results["d_t_mean"]
    d_std = results["d_t_std"]
    timesteps = np.arange(1, len(d_t) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, d_t, '-o', color='royalblue', linewidth=2, markersize=8,
             label='Mean Pairwise Disagreement ($\\Delta_t$)')
    plt.fill_between(timesteps, d_t - d_std, d_t + d_std,
                     alpha=0.2, color='royalblue', label='±1 Std Dev')

    plt.xlabel('Rollout Timestep (T)', fontsize=13)
    plt.ylabel('Ensemble Pairwise Disagreement ($\\Delta_t$)', fontsize=13)
    plt.title('Temporal Evolution of Epistemic Uncertainty', fontsize=15, fontweight='bold')
    plt.xticks(timesteps)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Temporal disagreement plot saved to {output_path}")


def plot_expression_trajectory(
    results: dict,
    gene_names: list,
    top_k: int = 10,
    output_path: str = "experiments/results/expression_trajectory.png",
) -> None:
    """
    Plot the mean expression trajectory of the top-k most dynamic genes.
    """
    expr_delta = results["expression_delta"]  # [T, genes]
    # Rank genes by total expression change across time
    total_change = expr_delta.sum(axis=0)
    top_gene_idx = np.argsort(total_change)[::-1][:top_k]

    timesteps = np.arange(1, expr_delta.shape[0] + 1)
    plt.figure(figsize=(10, 6))

    for gi in top_gene_idx:
        plt.plot(timesteps, expr_delta[:, gi], '-o', linewidth=1.5,
                 label=gene_names[gi] if gi < len(gene_names) else f"Gene {gi}")

    plt.xlabel('Rollout Timestep (T)', fontsize=13)
    plt.ylabel('Mean Predicted Expression Magnitude', fontsize=13)
    plt.title(f'Top {top_k} Dynamic Gene Expression Trajectories', fontsize=15, fontweight='bold')
    plt.xticks(timesteps)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=8, loc='upper left', ncol=2)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Expression trajectory plot saved to {output_path}")


def run_temporal_analysis():
    """Main entry point: run temporal analysis for the ATLAS ensemble."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Temporal Analysis — ATLAS Rollout Dynamics\n")

    gene_names = torch.load("data/processed/gene_names.pt", weights_only=False)
    dataset = NormanDataset(processed_dir="data/processed")
    num_genes = dataset.num_genes
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
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

    results = analyze_temporal_uncertainty(ensemble, simulator, loader, edge_index, device, steps=3)

    print(f"Mean Pairwise Disagreement per t: {[round(d, 4) for d in results['d_t_mean'].tolist()]}")
    plot_temporal_disagreement(results)
    plot_expression_trajectory(results, gene_names, top_k=10)


if __name__ == "__main__":
    run_temporal_analysis()
