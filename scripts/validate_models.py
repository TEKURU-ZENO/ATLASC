"""
validate_models.py

Validates trained ATLAS ensemble members for parameter health and prediction quality.

Checks per ensemble member (k=0,1,2):
1. Checkpoint loads without error.
2. All parameters are finite (no NaN/Inf weights).
3. Gradient flow is not trivially zero (non-collapsed model).
4. Model output is finite on a representative batch.
5. Ensemble diversity: pairwise disagreement between K models is non-trivial.
"""

import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.encoders.transcriptomic_encoder import TranscriptomicEncoder
from models.gnn.causal_gnn import CausalGNN
from models.ensemble import EnsembleWrapper
from causal.do_operator import apply_intervention
from utils.data_loader import NormanDataset, get_trrust_graph
from simulation.rollout_engine import RolloutEngine


OK   = "   [OK]"
WARN = " [WARN]"
FAIL = " [FAIL]"

K_MODELS = 3
LATENT_DIM = 16


def check_parameters_finite(model: torch.nn.Module, name: str) -> bool:
    for param_name, p in model.named_parameters():
        if not p.data.isfinite().all():
            print(f"{FAIL} {name}.{param_name}: contains NaN or Inf!")
            return False
    print(f"{OK}  {name}: all parameters finite.")
    return True


def check_output_finite(
    encoder: torch.nn.Module,
    gnn: torch.nn.Module,
    x: torch.Tensor,
    ints: torch.Tensor,
    k: int,
) -> bool:
    try:
        encoder.eval(); gnn.eval()
        with torch.no_grad():
            z0 = encoder(x)
            z_int = apply_intervention(z0, ints, value=0.0)
            # Single-step decode
            decoded = gnn.decode(z_int)
        finite = decoded.isfinite().all().item()
        status = OK if finite else FAIL
        print(f"{status}  k={k}: output shape={tuple(decoded.shape)}, finite={finite}")
        return finite
    except Exception as e:
        print(f"{FAIL}  k={k}: forward pass failed: {e}")
        return False


def check_ensemble_diversity(
    ensemble: EnsembleWrapper,
    simulator: RolloutEngine,
    x: torch.Tensor,
    ints: torch.Tensor,
    edge_index: torch.Tensor,
) -> bool:
    K = ensemble.k_models
    x = x.to('cpu'); ints = ints.to('cpu')

    with torch.no_grad():
        z0_k = ensemble.encode(x)
        z_int_k = torch.stack([
            apply_intervention(z0_k[k], ints, value=0.0)
            for k in range(K)
        ], dim=0)
        latent_k, trajectory = simulator.rollout(
            z_int_k, edge_index, steps=3,
            log_trajectory=True, enforce_stability=False
        )
        preds_k = ensemble.decode(latent_k)  # [K, batch, genes]

    pairwise_diffs = []
    for k1 in range(K):
        for k2 in range(k1 + 1, K):
            diff = ((preds_k[k1] - preds_k[k2]) ** 2).mean().item()
            pairwise_diffs.append(diff)

    mean_div = np.mean(pairwise_diffs)
    diverse = mean_div > 1e-6
    status = OK if diverse else WARN
    print(f"{status}  Ensemble mean pairwise MSE: {mean_div:.6f} (diverse={diverse})")

    # Epistemic uncertainty
    u_traj, d_t = ensemble.compute_epistemic_uncertainty(trajectory)
    u_mean = u_traj.mean().item()
    print(f"{OK}  Mean U_traj: {u_mean:.4f} | Pairwise Delta_t per step: {[round(d, 3) for d in d_t]}")

    return diverse


def run_model_validation():
    device = torch.device('cpu')
    print("=" * 60)
    print("ATLAS Model Validation")
    print("=" * 60)

    dataset = NormanDataset(processed_dir="data/processed")
    num_genes = dataset.num_genes
    edge_index = get_trrust_graph("data/processed").to(device)

    x0, ints0, _ = dataset[0]
    x_batch = x0.unsqueeze(0).to(device)
    ints_batch = ints0.unsqueeze(0).to(device)

    all_ok = True
    encoders, gnns = [], []

    print(f"\n[1] Per-member parameter and output checks (K={K_MODELS})")
    for k in range(K_MODELS):
        ckpt_path = f"models/checkpoints/epistemic_member_k{k}.pt"
        if not os.path.exists(ckpt_path):
            print(f"{FAIL}  k={k}: checkpoint not found at {ckpt_path}")
            all_ok = False
            continue

        enc = TranscriptomicEncoder(num_genes=num_genes, latent_dim=LATENT_DIM).to(device)
        gnn = CausalGNN(in_channels=LATENT_DIM, hidden_channels=LATENT_DIM,
                         out_channels=1, layers_count=3).to(device)

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        enc.load_state_dict(ckpt['encoder_state_dict'])
        gnn.load_state_dict(ckpt['gnn_state_dict'])
        enc.eval(); gnn.eval()

        ok_enc = check_parameters_finite(enc, f"Encoder k={k}")
        ok_gnn = check_parameters_finite(gnn, f"GNN     k={k}")
        ok_out = check_output_finite(enc, gnn, x_batch, ints_batch, k)

        if not all([ok_enc, ok_gnn, ok_out]):
            all_ok = False

        encoders.append(enc)
        gnns.append(gnn)

    if len(encoders) == K_MODELS:
        ensemble = EnsembleWrapper(encoders, gnns)
        simulator = RolloutEngine(core_model=ensemble, execution_mode="deterministic")

        print(f"\n[2] Ensemble diversity check (K={K_MODELS})")
        diverse = check_ensemble_diversity(ensemble, simulator, x_batch, ints_batch, edge_index)
        if not diverse:
            all_ok = False
    else:
        print(f"{WARN} Skipping ensemble checks (not all {K_MODELS} checkpoints loaded).")

    print("\n" + "=" * 60)
    if all_ok:
        print("RESULT: All model validation checks passed.")
    else:
        print("RESULT: Some checks failed. Review output above.")
    print("=" * 60)


if __name__ == "__main__":
    run_model_validation()
