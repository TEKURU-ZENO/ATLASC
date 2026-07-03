"""
generate_embeddings.py

Generate and save latent embeddings for all cells in the Norman dataset
using the trained ATLAS ensemble.

Outputs:
- experiments/results/embeddings_seen.pt    : [N_seen, latent_dim]
- experiments/results/embeddings_unseen.pt  : [N_unseen, latent_dim]
- experiments/results/embedding_labels.pt   : intervention gene indices per cell

These embeddings can be used for UMAP / downstream clustering analysis.
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


def extract_latent_embeddings(
    ensemble: EnsembleWrapper,
    simulator: RolloutEngine,
    dataloader: DataLoader,
    edge_index: torch.Tensor,
    device: torch.device,
    steps: int = 3,
) -> tuple:
    """
    Run forward pass and extract latent state at final rollout step.

    Returns:
        (embeddings, intervention_labels)
        embeddings: [N, latent_dim] tensor
        labels: [N, max_interventions] tensor
    """
    ensemble.eval()
    all_embeds, all_labels = [], []

    with torch.no_grad():
        for x, interventions, _ in dataloader:
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

            # Use first ensemble member's final latent, averaged across genes
            final_z = trajectory[-1][0]  # [batch, genes, latent]
            z_global = final_z.mean(dim=1)  # [batch, latent]

            all_embeds.append(z_global.cpu())
            all_labels.append(interventions.cpu())

    return torch.cat(all_embeds, dim=0), torch.cat(all_labels, dim=0)


def generate_and_save_embeddings():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Generating ATLAS Latent Embeddings\n" + "=" * 60)

    full_dataset = NormanDataset(processed_dir="data/processed")
    num_genes = full_dataset.num_genes

    # Split by intervention vocab (matching train.py)
    torch.manual_seed(42)
    all_int = full_dataset.interventions
    unique_genes = set()
    for b in range(all_int.size(0)):
        for idx in all_int[b]:
            if idx.item() != -1:
                unique_genes.add(idx.item())

    unique_genes_tensor = torch.tensor(list(unique_genes))
    shuffled = unique_genes_tensor[torch.randperm(len(unique_genes_tensor))]
    split_idx = int(0.9 * len(shuffled))
    val_vocab = set(shuffled[split_idx:].tolist())

    seen_idx, val_idx = [], []
    for i in range(len(full_dataset)):
        has_val = any(idx.item() in val_vocab for idx in all_int[i])
        (val_idx if has_val else seen_idx).append(i)

    seen_sub = torch.utils.data.Subset(full_dataset, seen_idx[:1000])
    val_sub  = torch.utils.data.Subset(full_dataset, val_idx[:300])

    seen_loader = DataLoader(seen_sub, batch_size=64, shuffle=False)
    val_loader  = DataLoader(val_sub,  batch_size=64, shuffle=False)

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

    print("Extracting seen-gene embeddings...")
    seen_embeds, seen_labels = extract_latent_embeddings(ensemble, simulator, seen_loader, edge_index, device)

    print("Extracting unseen-gene embeddings...")
    val_embeds, val_labels = extract_latent_embeddings(ensemble, simulator, val_loader, edge_index, device)

    os.makedirs("experiments/results", exist_ok=True)
    torch.save(seen_embeds, "experiments/results/embeddings_seen.pt")
    torch.save(val_embeds,  "experiments/results/embeddings_unseen.pt")
    torch.save(seen_labels, "experiments/results/embedding_labels_seen.pt")
    torch.save(val_labels,  "experiments/results/embedding_labels_unseen.pt")

    print(f"\nSeen embeddings shape:   {seen_embeds.shape}")
    print(f"Unseen embeddings shape: {val_embeds.shape}")
    print("Embeddings saved to experiments/results/")


if __name__ == "__main__":
    generate_and_save_embeddings()
