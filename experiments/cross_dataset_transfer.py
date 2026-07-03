"""
cross_dataset_transfer.py  (Phase 6 + Phase 7 — FIXED)
-------------------------------------------------------
Validates ATLAS zero-shot generalisation across datasets:
  Norman (in-domain holdout)
  Dixit  (mild OOD, 1703 intersection genes)
  Replogle (large-scale OOD)

Critical fix (Phase 6):
  Both Norman AND Dixit MSE/U are now computed with the SAME
  dixit_valid_gene_mask so the numbers are genuinely comparable.

Phase 7 extension:
  Replogle is evaluated with its own intersection mask.

All metrics are saved to experiments/results/multi_dataset_metrics.json
for Phase 8 multi-dataset plotting.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.encoders.transcriptomic_encoder import TranscriptomicEncoder
from models.gnn.causal_gnn import CausalGNN
from models.ensemble import EnsembleWrapper
from causal.do_operator import apply_intervention
from utils.data_loader import get_trrust_graph
from simulation.rollout_engine import RolloutEngine


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_frozen_ensemble(device, num_genes, latent_dim=16, k_models=3):
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
        enc.eval(); gnn.eval()
        for p in enc.parameters(): p.requires_grad = False
        for p in gnn.parameters(): p.requires_grad = False
        encoders.append(enc); gnns.append(gnn)
    return EnsembleWrapper(encoders, gnns)


# ──────────────────────────────────────────────────────────────────────────────
# KL divergence helper
# ──────────────────────────────────────────────────────────────────────────────

def compute_kl_divergence(p_latents, q_latents):
    """Approximate KL(P||Q) between two empirical distributions via Gaussian fit."""
    mu_p = p_latents.mean(dim=0);  var_p = p_latents.var(dim=0, unbiased=True) + 1e-6
    mu_q = q_latents.mean(dim=0);  var_q = q_latents.var(dim=0, unbiased=True) + 1e-6
    kl = torch.log(torch.sqrt(var_q) / torch.sqrt(var_p)) \
         + (var_p + (mu_p - mu_q)**2) / (2 * var_q) - 0.5
    return kl.sum().item()


# ──────────────────────────────────────────────────────────────────────────────
# Core evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_zero_shot(dataset_name, dataloader, ensemble, simulator,
                       edge_index, device, valid_gene_mask=None,
                       max_batches=None):
    """
    Run zero-shot evaluation on a dataloader.

    Args:
        valid_gene_mask : boolean tensor [num_genes]; if provided MSE and U_traj
                          are ONLY computed on masked genes. MUST be the same mask
                          for every dataset being compared.
    """
    print(f"  Evaluating '{dataset_name}' …", flush=True)
    mse_list, u_list, latent_reps = [], [], []
    criterion = nn.MSELoss(reduction="none")

    with torch.no_grad():
        for batch_idx, (x, interventions, targets) in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
            x = x.to(device); targets = targets.to(device)

            # Encode: [K, batch, genes, latent]
            z0_k = ensemble.encode(x)

            # Collect global latent for KL (first model, avg over genes → [batch, latent])
            z_global = z0_k[0].mean(dim=1)
            latent_reps.append(z_global.cpu())

            # Apply per-model interventions
            z_int_k = [apply_intervention(z0_k[k], interventions, value=0.0)
                       for k in range(ensemble.k_models)]
            z_intervened = torch.stack(z_int_k, dim=0)

            # Rollout
            latent_preds_k, trajectory = simulator.rollout(
                z_intervened, edge_index, steps=3,
                log_trajectory=True, enforce_stability=False
            )

            # Decode and average
            preds_k = ensemble.decode(latent_preds_k)          # [K, batch, genes]
            ensemble_pred = preds_k.mean(dim=0)                # [batch, genes]

            # ── MSE (masked) ─────────────────────────────────────────────────
            if valid_gene_mask is not None:
                mse_per_cell = criterion(
                    ensemble_pred[:, valid_gene_mask],
                    targets[:, valid_gene_mask]
                ).mean(dim=1).cpu().numpy()
            else:
                mse_per_cell = criterion(ensemble_pred, targets).mean(dim=1).cpu().numpy()
            mse_list.extend(mse_per_cell)

            # ── U_traj (masked) ──────────────────────────────────────────────
            u_traj_genes, _ = ensemble.compute_epistemic_uncertainty(trajectory)  # [batch, genes]
            if valid_gene_mask is not None:
                u_cells = u_traj_genes[:, valid_gene_mask].mean(dim=1).cpu().numpy()
            else:
                u_cells = u_traj_genes.mean(dim=1).cpu().numpy()
            u_list.extend(u_cells)

    all_mse = np.array(mse_list)
    all_u   = np.array(u_list)
    all_lat = torch.cat(latent_reps, dim=0)
    print(f"    → {len(all_mse)} cells | MSE_mean={all_mse.mean():.4f} | U_mean={all_u.mean():.4f}")
    return all_mse, all_u, all_lat


# ──────────────────────────────────────────────────────────────────────────────
# Calibration helpers
# ──────────────────────────────────────────────────────────────────────────────

def calibration_stats(all_u, all_mse):
    corr, pval = stats.pearsonr(all_u, all_mse)
    X = all_u.reshape(-1, 1); y = all_mse
    beta = LinearRegression().fit(X, y).coef_[0]
    return float(corr), float(pval), float(beta)


def plot_calibration_scatter(all_u, all_mse, label, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(all_u, all_mse, alpha=0.25, s=8, color="#3b82f6", label=label)
    x_lin = np.linspace(all_u.min(), all_u.max(), 100)
    beta = LinearRegression().fit(all_u.reshape(-1,1), all_mse).coef_[0]
    intercept = LinearRegression().fit(all_u.reshape(-1,1), all_mse).intercept_
    ax.plot(x_lin, intercept + beta * x_lin, "r--", lw=2, label=f"β={beta:.4f}")
    ax.set_xlabel("U_traj (epistemic uncertainty)"); ax.set_ylabel("MSE")
    ax.set_title(f"Calibration – {label}")
    ax.legend(); fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  Calibration plot → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main validation
# ──────────────────────────────────────────────────────────────────────────────

def perform_transfer_validation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = "experiments/results"
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "="*60)
    print("ATLAS Cross-Dataset Transfer Validation  (Phase 6 + 7)")
    print("="*60)

    # ── Gene vocab ────────────────────────────────────────────────────────────
    norman_genes = torch.load("data/processed/gene_names.pt", weights_only=False)
    num_genes    = len(norman_genes)
    print(f"\nNorman vocab: {num_genes} genes")

    # ── Load shared infrastructure ────────────────────────────────────────────
    print("\nLoading frozen ensemble …")
    edge_index = get_trrust_graph("data/processed").to(device)
    ensemble   = load_frozen_ensemble(device, num_genes)
    simulator  = RolloutEngine(ensemble, "deterministic")

    # ── DIXIT intersection mask (Phase 6 fix) ─────────────────────────────────
    dixit_mask_path = "data/processed/dixit/dixit_valid_gene_mask.pt"
    assert os.path.exists(dixit_mask_path), f"Missing {dixit_mask_path}"
    dixit_mask = torch.load(dixit_mask_path, weights_only=True)
    n_overlap  = dixit_mask.sum().item()
    print(f"\nDixit intersection genes: {n_overlap}/{num_genes}")

    # ── Norman holdout (using the SAME Dixit mask → fair comparison) ──────────
    print("\n[Norman] Building holdout split …")
    from utils.data_loader import NormanDataset
    full_ds = NormanDataset(processed_dir="data/processed")
    all_int = full_ds.interventions
    unique_genes = set()
    for b in range(all_int.size(0)):
        for idx in all_int[b]:
            if idx.item() != -1:
                unique_genes.add(idx.item())
    unique_t = torch.tensor(list(unique_genes))
    torch.manual_seed(42)
    shuffled = unique_t[torch.randperm(len(unique_t))]
    val_vocab = set(shuffled[int(0.9 * len(shuffled)):].tolist())

    val_idx = []
    for i in range(len(full_ds)):
        if any(all_int[i][j].item() in val_vocab for j in range(all_int.size(1))):
            val_idx.append(i)
    torch.manual_seed(42)
    val_tens = torch.tensor(val_idx)
    val_idx  = val_tens[torch.randperm(len(val_tens))][:400].tolist()
    norman_loader = DataLoader(
        torch.utils.data.Subset(full_ds, val_idx), batch_size=50, shuffle=False
    )

    print(f"\n[Norman] Evaluating on {len(val_idx)} holdout cells (mask = Dixit intersection) …")
    mse_norman, u_norman, lat_norman = evaluate_zero_shot(
        "Norman holdout", norman_loader, ensemble, simulator,
        edge_index, device, valid_gene_mask=dixit_mask.to(device)
    )
    corr_n, pval_n, beta_n = calibration_stats(u_norman, mse_norman)

    # ── Dixit evaluation ──────────────────────────────────────────────────────
    print("\n[Dixit] Loading aligned tensors …")
    dixit_x    = torch.load("data/processed/dixit/dixit_aligned_expression.pt",    weights_only=True)
    dixit_int  = torch.load("data/processed/dixit/dixit_aligned_interventions.pt", weights_only=True)
    dixit_targ = torch.load("data/processed/dixit/dixit_aligned_targets.pt",       weights_only=True)

    # Sub-sample for speed (100 batches of 50 = 5000 cells)
    n_dixit = min(5000, dixit_x.size(0))
    idx_d   = torch.randperm(dixit_x.size(0))[:n_dixit]
    dixit_loader = DataLoader(
        TensorDataset(dixit_x[idx_d], dixit_int[idx_d], dixit_targ[idx_d]),
        batch_size=50
    )
    print(f"[Dixit] Evaluating on {n_dixit} cells (mask = same Dixit intersection) …")
    mse_dixit, u_dixit, lat_dixit = evaluate_zero_shot(
        "Dixit OOD", dixit_loader, ensemble, simulator,
        edge_index, device, valid_gene_mask=dixit_mask.to(device)
    )
    corr_d, pval_d, beta_d = calibration_stats(u_dixit, mse_dixit)
    kl_nd = compute_kl_divergence(lat_norman, lat_dixit)

    # ── Replogle evaluation (Phase 7) ─────────────────────────────────────────
    rep_mask_path = "data/processed/replogle/replogle_valid_gene_mask.pt"
    has_replogle = os.path.exists("data/processed/replogle/replogle_aligned_expression.pt")

    if has_replogle:
        print("\n[Replogle] Loading aligned tensors …")
        rep_mask  = torch.load(rep_mask_path, weights_only=True)
        rep_x     = torch.load("data/processed/replogle/replogle_aligned_expression.pt",    weights_only=True)
        rep_int   = torch.load("data/processed/replogle/replogle_aligned_interventions.pt", weights_only=True)
        rep_targ  = torch.load("data/processed/replogle/replogle_aligned_targets.pt",       weights_only=True)
        n_rep = min(5000, rep_x.size(0))
        idx_r = torch.randperm(rep_x.size(0))[:n_rep]
        rep_loader = DataLoader(
            TensorDataset(rep_x[idx_r], rep_int[idx_r], rep_targ[idx_r]),
            batch_size=50
        )
        print(f"[Replogle] Evaluating on {n_rep} cells ({rep_mask.sum().item()} overlap genes) …")
        mse_rep, u_rep, lat_rep = evaluate_zero_shot(
            "Replogle large-OOD", rep_loader, ensemble, simulator,
            edge_index, device, valid_gene_mask=rep_mask.to(device)
        )
        corr_r, pval_r, beta_r = calibration_stats(u_rep, mse_rep)
        kl_nr = compute_kl_divergence(lat_norman, lat_rep)
    else:
        print("\n[Replogle] Aligned tensors not found – run scripts/align_replogle_data.py first")
        mse_rep = u_rep = None

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    datasets = ["Norman", "Dixit"]
    mses     = [mse_norman.mean(), mse_dixit.mean()]
    us       = [u_norman.mean(),   u_dixit.mean()]

    print(f"\n{'Dataset':<12} {'MSE':>10} {'U_traj':>10}  corr(U,MSE)      p-val     beta")
    print("-"*72)
    print(f"{'Norman':<12} {mse_norman.mean():>10.4f} {u_norman.mean():>10.4f}  {corr_n:>+.4f}        {pval_n:.2e}   {beta_n:.4f}")
    print(f"{'Dixit':<12} {mse_dixit.mean():>10.4f} {u_dixit.mean():>10.4f}  {corr_d:>+.4f}        {pval_d:.2e}   {beta_d:.4f}")
    if has_replogle:
        print(f"{'Replogle':<12} {mse_rep.mean():>10.4f} {u_rep.mean():>10.4f}  {corr_r:>+.4f}        {pval_r:.2e}   {beta_r:.4f}")
        datasets.append("Replogle"); mses.append(mse_rep.mean()); us.append(u_rep.mean())

    delta_mse = mse_dixit.mean() - mse_norman.mean()
    delta_u   = u_dixit.mean()   - u_norman.mean()
    print(f"\n  ΔDomain shift (Dixit−Norman): ΔMSE={delta_mse:+.4f}  ΔU={delta_u:+.4f}")
    print(f"  KL(Norman‖Dixit) in latent space: {kl_nd:.2f}")

    # Interpretation
    print("\nInterpretation:")
    if delta_mse > 0 and delta_u > 0:
        print("  ✅ PASS – Error increases OOD and uncertainty responds correctly.")
    elif delta_mse > 0 and delta_u <= 0:
        print("  ❌ FAIL – Error increases OOD but uncertainty collapsed.")
    elif delta_mse <= 0:
        print("  ⚠️  WARNING – OOD MSE ≤ in-domain. Check zero-padding / data leakage.")

    if corr_d > 0.1 and pval_d < 0.05:
        print("  ✅ PASS – Calibration: U_traj predicts error on OOD data.")
    else:
        print(f"  ⚠️  WARNING – Calibration weak on Dixit (r={corr_d:.4f}, p={pval_d:.2e}).")

    # ── Save metrics JSON ──────────────────────────────────────────────────────
    metrics = {
        "norman":  {"mse_mean": float(mse_norman.mean()), "u_mean": float(u_norman.mean()),
                    "corr": corr_n, "pval": pval_n, "beta": beta_n},
        "dixit":   {"mse_mean": float(mse_dixit.mean()),  "u_mean": float(u_dixit.mean()),
                    "corr": corr_d, "pval": pval_d, "beta": beta_d,
                    "kl_from_norman": kl_nd,
                    "n_overlap_genes": int(n_overlap),
                    "delta_mse": float(delta_mse), "delta_u": float(delta_u)},
    }
    if has_replogle:
        metrics["replogle"] = {
            "mse_mean": float(mse_rep.mean()), "u_mean": float(u_rep.mean()),
            "corr": corr_r, "pval": pval_r, "beta": beta_r,
            "kl_from_norman": kl_nr,
            "n_overlap_genes": int(rep_mask.sum().item()),
        }

    json_path = f"{results_dir}/multi_dataset_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics JSON → {json_path}")

    # ── Calibration plots ──────────────────────────────────────────────────────
    plot_calibration_scatter(u_dixit, mse_dixit, "Dixit OOD",
                             f"{results_dir}/calibration_dixit_fixed.png")
    if has_replogle:
        plot_calibration_scatter(u_rep, mse_rep, "Replogle OOD",
                                 f"{results_dir}/calibration_replogle.png")

    print("\n✓ Phase 6+7 validation complete.\n")
    return metrics


if __name__ == "__main__":
    perform_transfer_validation()
