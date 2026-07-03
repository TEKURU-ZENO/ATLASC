"""
evaluate.py

Full evaluation pipeline for a trained ATLAS ensemble.

Runs:
1. In-domain (Norman holdout) MSE and uncertainty.
2. Calibration: corr(U_traj, MSE), slope β.
3. Per-gene MSE report.
4. Structural generalization: seen vs unseen intervention genes.
5. Saves all results to experiments/results/.
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.encoders.transcriptomic_encoder import TranscriptomicEncoder
from models.gnn.causal_gnn import CausalGNN
from models.ensemble import EnsembleWrapper
from causal.do_operator import apply_intervention
from utils.data_loader import NormanDataset, get_trrust_graph
from simulation.rollout_engine import RolloutEngine
from evaluation.metrics import (
    compute_mse_per_cell, compute_pearson_correlation,
    compute_calibration_slope, print_metrics_summary
)
from evaluation.calibration import run_calibration_analysis
from evaluation.visualization_utils import plot_uncertainty_vs_mse, plot_mse_distribution


def load_ensemble(device, num_genes, latent_dim=16, k_models=3):
    encoders, gnns = [], []
    for k in range(k_models):
        enc = TranscriptomicEncoder(num_genes=num_genes, latent_dim=latent_dim).to(device)
        gnn = CausalGNN(in_channels=latent_dim, hidden_channels=latent_dim,
                         out_channels=1, layers_count=3).to(device)
        ckpt = torch.load(f"models/checkpoints/epistemic_member_k{k}.pt",
                          map_location=device, weights_only=True)
        enc.load_state_dict(ckpt['encoder_state_dict'])
        gnn.load_state_dict(ckpt['gnn_state_dict'])
        enc.eval(); gnn.eval()
        for p in enc.parameters(): p.requires_grad = False
        for p in gnn.parameters(): p.requires_grad = False
        encoders.append(enc); gnns.append(gnn)
    return EnsembleWrapper(encoders, gnns)


def run_split_evaluation(
    ensemble, simulator, dataloader, edge_index, device, split_name="Validation"
):
    """Evaluate a split and return (mse_array, u_array, pred_tensor, target_tensor)."""
    ensemble.eval()
    criterion = torch.nn.MSELoss(reduction='none')
    mse_list, u_list = [], []
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x, interventions, targets in dataloader:
            x = x.to(device)
            targets = targets.to(device)

            z0_k = ensemble.encode(x)
            z_int_k = torch.stack([
                apply_intervention(z0_k[k], interventions, value=0.0)
                for k in range(ensemble.k_models)
            ], dim=0)

            latent_k, trajectory = simulator.rollout(
                z_int_k, edge_index, steps=3,
                log_trajectory=True, enforce_stability=False
            )

            preds_k = ensemble.decode(latent_k)
            pred_mean = preds_k.mean(dim=0)

            mse = criterion(pred_mean, targets).mean(dim=-1).cpu().numpy()
            mse_list.extend(mse)

            u_traj, _ = ensemble.compute_epistemic_uncertainty(trajectory)
            u_cell = u_traj.mean(dim=-1).cpu().numpy()
            u_list.extend(u_cell)

            all_preds.append(pred_mean.cpu())
            all_targets.append(targets.cpu())

    if len(all_preds) == 0:
        print(f"\n[{split_name}]")
        print("  WARNING: No cells available in this split.")
        return np.array([]), np.array([]), torch.empty(0), torch.empty(0)
        
    mse_arr = np.array(mse_list)
    u_arr = np.array(u_list)
    preds_t = torch.cat(all_preds, dim=0)
    targets_t = torch.cat(all_targets, dim=0)
    
    print(f"\n[{split_name}]")
    print(f"  N cells: {len(mse_arr)}")
    print(f"  Mean MSE: {mse_arr.mean():.4f} +/- {mse_arr.std():.4f}")
    print(f"  Mean U_traj: {u_arr.mean():.4f}")

    r, p = compute_pearson_correlation(u_arr, mse_arr)
    beta, alpha = compute_calibration_slope(u_arr, mse_arr)
    print(f"  corr(U, MSE): {r:.4f} (p={p:.2e})")
    print(f"  Calibration slope beta: {beta:.4f}")

    return mse_arr, u_arr, preds_t, targets_t


def run_full_evaluation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("ATLAS Full Evaluation Pipeline\n" + "=" * 60)

    full_dataset = NormanDataset(processed_dir="data/processed")
    num_genes = full_dataset.num_genes
    gene_names = torch.load("data/processed/gene_names.pt", weights_only=False)

    # Structural holdout split (matching train.py seed)
    torch.manual_seed(42)
    all_interventions = full_dataset.interventions
    unique_genes = set()
    for b in range(all_interventions.size(0)):
        for idx in all_interventions[b]:
            if idx.item() != -1:
                unique_genes.add(idx.item())

    unique_genes_tensor = torch.tensor(list(unique_genes))
    shuffled = unique_genes_tensor[torch.randperm(len(unique_genes_tensor))]
    split_idx = int(0.9 * len(shuffled))
    val_vocab = set(shuffled[split_idx:].tolist())

    seen_idx, val_idx = [], []
    for i in range(len(full_dataset)):
        has_val = any(idx.item() in val_vocab for idx in all_interventions[i])
        (val_idx if has_val else seen_idx).append(i)

    torch.manual_seed(42)
    seen_sub = torch.utils.data.Subset(full_dataset,
                                        torch.tensor(seen_idx)[torch.randperm(len(seen_idx))][:500])
    val_sub = torch.utils.data.Subset(full_dataset,
                                       torch.tensor(val_idx)[torch.randperm(len(val_idx))][:200])

    seen_loader = DataLoader(seen_sub, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_sub, batch_size=64, shuffle=False)

    edge_index = get_trrust_graph("data/processed").to(device)
    ensemble = load_ensemble(device, num_genes)
    simulator = RolloutEngine(core_model=ensemble, execution_mode="deterministic")

    mse_seen, u_seen, _, _ = run_split_evaluation(ensemble, simulator, seen_loader, edge_index, device, "Seen")
    mse_val, u_val, _, _ = run_split_evaluation(ensemble, simulator, val_loader, edge_index, device, "Unseen (Holdout)")

    os.makedirs("experiments/results", exist_ok=True)
    print("\nGenerating visualization plots...")
    if len(u_val) > 0:
        plot_uncertainty_vs_mse(u_val, mse_val, "Unseen Holdout: U vs MSE",
                               "experiments/results/eval_u_vs_mse_holdout.png")
        plot_mse_distribution(mse_seen, mse_val, "experiments/results/eval_mse_distribution.png")

        run_calibration_analysis(u_val, mse_val, label="Unseen Holdout",
                                 output_prefix="experiments/results/eval_calibration")
    else:
        print("Skipping unseen holdout plots (no unseen cells).")
        plot_mse_distribution(mse_seen, mse_seen, "experiments/results/eval_mse_distribution_seen_only.png")

    print("\nEvaluation complete. Outputs saved to experiments/results/")


if __name__ == "__main__":
    run_full_evaluation()
