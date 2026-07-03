"""
hyperparameter_search.py

Grid search over key ATLAS hyperparameters using the Norman holdout validation MSE.

Searched hyperparameters:
- latent_dim: [8, 16, 32]
- learning_rate: [5e-4, 1e-3, 5e-3]
- rollout_steps: [1, 2, 3]
- lambda_reg: [0.0, 0.01, 0.1]

Each configuration is trained quickly (2 epochs, small data subset) to rank configs.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from itertools import product
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.encoders.transcriptomic_encoder import TranscriptomicEncoder
from models.gnn.causal_gnn import CausalGNN
from causal.do_operator import apply_intervention
from utils.data_loader import NormanDataset, get_trrust_graph
from simulation.rollout_engine import RolloutEngine


SEARCH_SPACE = {
    "latent_dim":    [8, 16, 32],
    "lr":            [5e-4, 1e-3, 5e-3],
    "rollout_steps": [1, 2, 3],
    "lambda_reg":    [0.0, 0.01, 0.1],
}

FAST_EPOCHS = 2
TRAIN_SIZE = 400
VAL_SIZE = 100


def train_and_eval_config(
    config: dict,
    full_dataset: NormanDataset,
    edge_index: torch.Tensor,
    device: torch.device,
    val_vocab: set,
) -> float:
    """Train a single K=1 model with the given config and return validation MSE."""
    latent_dim = config["latent_dim"]
    lr = config["lr"]
    steps = config["rollout_steps"]
    lambda_reg = config["lambda_reg"]

    all_int = full_dataset.interventions
    seen_idx, val_idx = [], []
    for i in range(len(full_dataset)):
        has_val = any(idx.item() in val_vocab for idx in all_int[i])
        (val_idx if has_val else seen_idx).append(i)

    torch.manual_seed(99)
    train_sub = torch.utils.data.Subset(full_dataset,
        torch.tensor(seen_idx)[torch.randperm(len(seen_idx))][:TRAIN_SIZE])
    val_sub = torch.utils.data.Subset(full_dataset,
        torch.tensor(val_idx)[torch.randperm(len(val_idx))][:VAL_SIZE])

    train_loader = DataLoader(train_sub, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_sub,   batch_size=64, shuffle=False)

    num_genes = full_dataset.num_genes
    encoder = TranscriptomicEncoder(num_genes=num_genes, latent_dim=latent_dim).to(device)
    gnn = CausalGNN(in_channels=latent_dim, hidden_channels=latent_dim,
                    out_channels=1, layers_count=3).to(device)
    simulator = RolloutEngine(core_model=gnn, execution_mode="deterministic")

    optimizer = optim.Adam(list(encoder.parameters()) + list(gnn.parameters()), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(FAST_EPOCHS):
        encoder.train(); gnn.train()
        for x, interventions, target in train_loader:
            x = x.to(device); target = target.to(device)
            optimizer.zero_grad()
            z0 = encoder(x)
            z_int = apply_intervention(z0, interventions, value=0.0)
            latent_pred, traj = simulator.rollout(
                z_int, edge_index, steps=steps, log_trajectory=True, enforce_stability=True
            )
            pred = gnn.decode(latent_pred)
            loss = criterion(pred, target)
            if traj is not None and lambda_reg > 0:
                loss = loss + lambda_reg * torch.mean(torch.norm(traj, dim=-1))
            loss.backward()
            optimizer.step()

    # Validate
    encoder.eval(); gnn.eval()
    val_losses = []
    with torch.no_grad():
        for x, interventions, target in val_loader:
            x = x.to(device); target = target.to(device)
            z0 = encoder(x)
            z_int = apply_intervention(z0, interventions, value=0.0)
            latent_pred, _ = simulator.rollout(z_int, edge_index, steps=steps, log_trajectory=False)
            pred = gnn.decode(latent_pred)
            val_losses.append(criterion(pred, target).item())
    return float(np.mean(val_losses))


def run_hyperparameter_search():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("ATLAS Hyperparameter Search\n" + "=" * 60)

    dataset = NormanDataset(processed_dir="data/processed")
    edge_index = get_trrust_graph("data/processed").to(device)

    torch.manual_seed(42)
    all_int = dataset.interventions
    unique_genes = set()
    for b in range(all_int.size(0)):
        for idx in all_int[b]:
            if idx.item() != -1:
                unique_genes.add(idx.item())
    unique_genes_tensor = torch.tensor(list(unique_genes))
    shuffled = unique_genes_tensor[torch.randperm(len(unique_genes_tensor))]
    split_idx = int(0.9 * len(shuffled))
    val_vocab = set(shuffled[split_idx:].tolist())

    keys = list(SEARCH_SPACE.keys())
    values = list(SEARCH_SPACE.values())
    all_configs = [dict(zip(keys, combo)) for combo in product(*values)]

    print(f"Total configurations: {len(all_configs)}\n")

    results = []
    for i, config in enumerate(all_configs):
        print(f"[{i+1}/{len(all_configs)}] Config: {config}")
        try:
            val_mse = train_and_eval_config(config, dataset, edge_index, device, val_vocab)
            results.append((val_mse, config))
            print(f"  Val MSE: {val_mse:.5f}")
        except Exception as e:
            print(f"  FAILED: {e}")

    results.sort(key=lambda x: x[0])

    print("\n" + "=" * 60)
    print("Top 5 Configurations:")
    print(f"\n{'Rank':<6} {'Val MSE':>10}  Config")
    print("-" * 75)
    for rank, (mse, cfg) in enumerate(results[:5], start=1):
        print(f"{rank:<6} {mse:>10.5f}  {cfg}")

    print(f"\nBest Config: {results[0][1]}")
    print(f"Best Val MSE: {results[0][0]:.5f}")


if __name__ == "__main__":
    run_hyperparameter_search()
