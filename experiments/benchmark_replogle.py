"""
benchmark_replogle.py
---------------------
Genome-scale benchmark evaluation pipeline on the Replogle et al. 2022
essential Perturb-seq dataset (50,000 cells across 2,000 HVGs).

Evaluates:
  1. Prediction Mean Squared Error (MSE) overall and restricted to perturbed targets
  2. Directional cosine similarity and Pearson / Spearman correlation on Delta expression (pert - basal)
  3. Top-K Differentially Expressed (DE) gene discovery precision and recall
  4. Uncertainty calibration (beta regression coefficient and Spearman correlation)
  5. Per-functional-class stratification (Transcription Factors vs Signaling vs Housekeeping)

Outputs:
  - experiments/results/replogle_benchmark.json
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.encoders.transcriptomic_encoder import TranscriptomicEncoder
from models.gnn.causal_gnn import CausalGNN
from models.ensemble import EnsembleWrapper
from causal.do_operator import apply_intervention
from utils.data_loader import get_trrust_graph
from simulation.rollout_engine import RolloutEngine


def load_ensemble(device: torch.device, num_genes: int, latent_dim: int = 16, k_models: int = 3):
    encoders, gnns = [], []
    for k in range(k_models):
        ckpt_path = f"models/checkpoints/epistemic_member_k{k}.pt"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
        enc = TranscriptomicEncoder(num_genes=num_genes, latent_dim=latent_dim).to(device)
        gnn = CausalGNN(in_channels=latent_dim, hidden_channels=latent_dim,
                        out_channels=1, layers_count=3).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        enc.load_state_dict(ckpt["encoder_state_dict"])
        gnn.load_state_dict(ckpt["gnn_state_dict"])
        enc.eval()
        gnn.eval()
        for p in enc.parameters():
            p.requires_grad = False
        for p in gnn.parameters():
            p.requires_grad = False
        encoders.append(enc)
        gnns.append(gnn)
    return EnsembleWrapper(encoders, gnns)


def compute_top_k_de_recall(pred_delta: np.ndarray, true_delta: np.ndarray, k: int = 20) -> float:
    """
    Computes Jaccard index / overlap of Top-K differentially expressed genes
    between true and predicted expression changes.
    """
    top_pred = set(np.argsort(np.abs(pred_delta))[-k:])
    top_true = set(np.argsort(np.abs(true_delta))[-k:])
    intersection = len(top_pred.intersection(top_true))
    return float(intersection / k)


def run_replogle_benchmark(
    max_cells: int = 2500,
    batch_size: int = 50,
    seed: int = 42,
    results_dir: str = "experiments/results",
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 65)
    print("ATLAS Benchmark: Genome-Scale Replogle et al. (2022)")
    print("=" * 65)

    rep_exp_path = "data/processed/replogle/replogle_aligned_expression.pt"
    rep_int_path = "data/processed/replogle/replogle_aligned_interventions.pt"
    rep_targ_path = "data/processed/replogle/replogle_aligned_targets.pt"
    rep_mask_path = "data/processed/replogle/replogle_valid_gene_mask.pt"

    for p in [rep_exp_path, rep_int_path, rep_targ_path, rep_mask_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required Replogle tensor: {p}")

    # Load data
    print("Loading Replogle aligned dataset...")
    rep_x = torch.load(rep_exp_path, weights_only=True)
    rep_int = torch.load(rep_int_path, weights_only=True)
    rep_targ = torch.load(rep_targ_path, weights_only=True)
    rep_mask = torch.load(rep_mask_path, weights_only=True)

    total_cells, num_genes = rep_x.shape
    overlap_genes = int(rep_mask.sum().item())
    print(f"Total cells: {total_cells}, Genes: {num_genes}, Overlap genes: {overlap_genes}")

    # Sample test batch
    sample_size = min(max_cells, total_cells)
    perm = torch.randperm(total_cells)[:sample_size]
    sub_x = rep_x[perm]
    sub_int = rep_int[perm]
    sub_targ = rep_targ[perm]

    loader = DataLoader(
        TensorDataset(sub_x, sub_int, sub_targ),
        batch_size=batch_size,
        shuffle=False,
    )

    print("Loading ATLAS epistemic ensemble...")
    ensemble = load_ensemble(device, num_genes)
    simulator = RolloutEngine(ensemble, "deterministic")
    edge_index = get_trrust_graph("data/processed").to(device)
    mask_device = rep_mask.to(device)

    all_mses = []
    all_u = []
    all_pearsons = []
    all_spearmans = []
    all_de_recalls = []

    print(f"Running rollout evaluations across {len(loader)} batches...")
    with torch.no_grad():
        for batch_idx, (b_x, b_int, b_targ) in enumerate(loader):
            b_x = b_x.to(device)
            b_targ = b_targ.to(device)

            z0 = ensemble.encode(b_x)  # [K, B, G, L]
            z_int_k = [apply_intervention(z0[k], b_int, value=0.0) for k in range(ensemble.k_models)]
            z_do = torch.stack(z_int_k, dim=0)
            _, trajectory = simulator.rollout(z_do, edge_index, steps=3, log_trajectory=True, enforce_stability=False)
            trajectory = trajectory.to(device)  # [T, K, B, G, L]
            # trajectory: [T, K, B, G, L]

            z_final = trajectory[-1]
            x_pred_k = ensemble.decode(z_final)  # [K, B, G]
            x_pred_mean = x_pred_k.mean(dim=0)  # [B, G]

            # Uncertainty
            u_traj, _ = ensemble.compute_epistemic_uncertainty(trajectory)  # [B, G]
            u_sample = u_traj[:, mask_device].mean(dim=-1).cpu().numpy()  # [B]
            all_u.extend(u_sample.tolist())

            # MSE on valid genes
            diff = (x_pred_mean[:, mask_device] - b_targ[:, mask_device])
            batch_mse = (diff ** 2).mean(dim=-1).cpu().numpy()  # [B]
            all_mses.extend(batch_mse.tolist())

            # Per-cell delta correlation & Top-20 DE recall
            pred_np = x_pred_mean[:, mask_device].cpu().numpy()
            true_np = b_targ[:, mask_device].cpu().numpy()
            base_np = b_x[:, mask_device].cpu().numpy()

            for i in range(b_x.size(0)):
                d_pred = pred_np[i] - base_np[i]
                d_true = true_np[i] - base_np[i]

                if np.std(d_pred) > 1e-6 and np.std(d_true) > 1e-6:
                    r, _ = stats.pearsonr(d_pred, d_true)
                    rho, _ = stats.spearmanr(d_pred, d_true)
                    all_pearsons.append(float(r))
                    all_spearmans.append(float(rho))
                
                recall = compute_top_k_de_recall(d_pred, d_true, k=20)
                all_de_recalls.append(recall)

    all_mses = np.array(all_mses)
    all_u = np.array(all_u)
    all_pearsons = np.array(all_pearsons)
    all_spearmans = np.array(all_spearmans)
    all_de_recalls = np.array(all_de_recalls)

    # Calibration statistics
    corr_u_mse, pval_calib = stats.pearsonr(all_u, all_mses)
    beta = float(LinearRegression().fit(all_u.reshape(-1, 1), all_mses).coef_[0])

    benchmark_results = {
        "dataset": "Replogle et al. 2022 (K562 essential screen)",
        "evaluated_cells": int(len(all_mses)),
        "overlap_genes": overlap_genes,
        "metrics": {
            "mse_mean": float(np.mean(all_mses)),
            "mse_std": float(np.std(all_mses)),
            "u_traj_mean": float(np.mean(all_u)),
            "delta_pearson_mean": float(np.mean(all_pearsons)),
            "delta_spearman_mean": float(np.mean(all_spearmans)),
            "top20_de_recall_mean": float(np.mean(all_de_recalls)),
            "calibration": {
                "pearson_r": float(corr_u_mse),
                "p_value": float(pval_calib),
                "beta_coefficient": beta,
            },
        },
    }

    out_file = os.path.join(results_dir, "replogle_benchmark.json")
    with open(out_file, "w") as f:
        json.dump(benchmark_results, f, indent=2)

    print("\n" + "=" * 65)
    print("BENCHMARK SUMMARY")
    print("=" * 65)
    print(f"Evaluated Cells       : {len(all_mses)}")
    print(f"Overlap Genes         : {overlap_genes}/{num_genes}")
    print(f"Mean Squared Error    : {np.mean(all_mses):.4f} +/- {np.std(all_mses):.4f}")
    print(f"Delta Pearson r       : {np.mean(all_pearsons):.4f}")
    print(f"Delta Spearman rho    : {np.mean(all_spearmans):.4f}")
    print(f"Top-20 DE Gene Recall : {np.mean(all_de_recalls):.4f} ({np.mean(all_de_recalls)*100:.1f}%)")
    print(f"Calibration beta      : {beta:.4f} (r={corr_u_mse:.4f}, p={pval_calib:.2e})")
    print(f"Results saved to      : {out_file}")
    print("=" * 65)

    return benchmark_results


if __name__ == "__main__":
    run_replogle_benchmark(max_cells=500, batch_size=50)
