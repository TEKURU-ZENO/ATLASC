"""
ablation.py

Ablation study runner for ATLAS.

Systematically removes or disables architectural components and measures
the resulting degradation in prediction quality and epistemic calibration.

Ablation conditions:
1. Full Model (baseline)
2. No GRN (replace edge_index with an empty graph)
3. Single Ensemble Member (K=1, no epistemic uncertainty)
4. No Curriculum (fixed T=3 from epoch 0)
5. No Intervention (no-op apply_intervention)
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.encoders.transcriptomic_encoder import TranscriptomicEncoder
from models.gnn.causal_gnn import CausalGNN
from models.ensemble import EnsembleWrapper
from causal.do_operator import apply_intervention
from utils.data_loader import NormanDataset, get_trrust_graph
from simulation.rollout_engine import RolloutEngine


def evaluate_condition(
    ensemble: EnsembleWrapper,
    simulator: RolloutEngine,
    dataloader: DataLoader,
    edge_index: torch.Tensor,
    device: torch.device,
    steps: int = 3,
    no_intervention: bool = False,
) -> dict:
    """
    Evaluate a model configuration and return mean MSE and mean U_traj.
    """
    ensemble.eval()
    criterion = nn.MSELoss(reduction='none')
    mse_list, u_list = [], []

    with torch.no_grad():
        for x, interventions, targets in dataloader:
            x = x.to(device)
            targets = targets.to(device)

            if no_intervention:
                interventions = torch.full_like(interventions, -1)

            z0_k = ensemble.encode(x)
            z_int_k = torch.stack([
                apply_intervention(z0_k[k], interventions, value=0.0)
                for k in range(ensemble.k_models)
            ], dim=0)

            latent_k, trajectory = simulator.rollout(
                z_int_k, edge_index, steps=steps,
                log_trajectory=True, enforce_stability=False
            )

            preds = ensemble.decode(latent_k).mean(dim=0)
            mse = criterion(preds, targets).mean(dim=-1).cpu().numpy()
            mse_list.extend(mse)

            u_traj, _ = ensemble.compute_epistemic_uncertainty(trajectory)
            u_cell = u_traj.mean(dim=-1).cpu().numpy()
            u_list.extend(u_cell)

    return {
        "mean_mse": float(np.mean(mse_list)),
        "mean_u": float(np.mean(u_list)),
    }


def run_ablation_study():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("ATLAS Ablation Study\n" + "=" * 60)

    dataset = NormanDataset(processed_dir="data/processed")
    num_genes = dataset.num_genes

    torch.manual_seed(42)
    subset_size = 300
    indices = torch.randperm(len(dataset))[:subset_size]
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=64, shuffle=False)

    edge_index = get_trrust_graph("data/processed").to(device)
    empty_edge = torch.zeros((2, 0), dtype=torch.long).to(device)

    def load_ensemble(k_max=3):
        encoders, gnns = [], []
        for k in range(k_max):
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
        return ensemble, simulator

    conditions = [
        ("Full Model (K=3, GRN, Intervention)",
         lambda: load_ensemble(3),
         dict(edge_index=edge_index, no_intervention=False)),
        ("No GRN (empty graph)",
         lambda: load_ensemble(3),
         dict(edge_index=empty_edge, no_intervention=False)),
        ("Single Member (K=1)",
         lambda: load_ensemble(1),
         dict(edge_index=edge_index, no_intervention=False)),
        ("No Intervention (no-op)",
         lambda: load_ensemble(3),
         dict(edge_index=edge_index, no_intervention=True)),
    ]

    print(f"\n{'Condition':<45} {'Mean MSE':>10} {'Mean U':>10}")
    print("-" * 68)

    for name, loader_fn, kwargs in conditions:
        ens, sim = loader_fn()
        results = evaluate_condition(ens, sim, loader, **kwargs, device=device)
        print(f"{name:<45} {results['mean_mse']:>10.4f} {results['mean_u']:>10.4f}")

    print("\nAblation study complete.")


if __name__ == "__main__":
    run_ablation_study()
