import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import argparse

# Add project root to sys path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.encoders.transcriptomic_encoder import TranscriptomicEncoder
from models.gnn.causal_gnn import CausalGNN
from models.ensemble import EnsembleWrapper
from causal.do_operator import apply_intervention
from utils.data_loader import NormanDataset, get_trrust_graph
from simulation.rollout_engine import RolloutEngine
from active_learning.simulate_candidate_pool import generate_candidate_pool
from active_learning.acquisition import compute_candidate_metrics, rank_and_select

def get_node_degree(gene_idx, edge_index):
    out_edges = (edge_index[0] == gene_idx).sum().item()
    in_edges = (edge_index[1] == gene_idx).sum().item()
    return out_edges + in_edges

def train_ensemble_in_memory(train_loader, edge_index, num_genes, num_epochs=2, latent_dim=16, device='cpu', verbose=False):
    K_models = 3
    encoders = []
    gnns = []
    
    for k in range(K_models):
        if verbose: print(f"  Training Model K={k+1}/{K_models}...")
        torch.manual_seed(100 + k)
        
        encoder = TranscriptomicEncoder(num_genes=num_genes, latent_dim=latent_dim).to(device)
        gnn = CausalGNN(in_channels=latent_dim, hidden_channels=latent_dim, out_channels=1, layers_count=3).to(device)
        simulator = RolloutEngine(core_model=gnn, execution_mode="deterministic")
        
        optimizer = optim.Adam(list(encoder.parameters()) + list(gnn.parameters()), lr=0.001)
        criterion = nn.MSELoss()
        
        encoder.train()
        gnn.train()
        
        for epoch in range(num_epochs):
            current_steps = 2
            
            for x, intervened_genes, target in train_loader:
                x = x.to(device)
                target = target.to(device)
                
                optimizer.zero_grad()
                z0 = encoder(x)
                z_intervened = apply_intervention(z0, intervened_genes, value=0.0)
                
                latent_pred, trajectory = simulator.rollout(
                    z_intervened, 
                    edge_index, 
                    steps=current_steps, 
                    log_trajectory=True, 
                    enforce_stability=True
                )
                
                pred = gnn.decode(latent_pred)
                recon_loss = criterion(pred, target)
                
                if trajectory is not None:
                    norm_penalty = torch.mean(torch.norm(trajectory, dim=-1))
                    loss = recon_loss + 1e-2 * norm_penalty
                else:
                    loss = recon_loss
                    
                loss.backward()
                optimizer.step()
                
        encoders.append(encoder)
        gnns.append(gnn)
        
    return EnsembleWrapper(encoders, gnns)

def evaluate_holdout(ensemble, holdout_loader, edge_index, simulator, device='cpu'):
    criterion = nn.MSELoss()
    total_loss = 0.0
    
    ensemble.eval()
    with torch.no_grad():
        for x, intervened_genes, target in holdout_loader:
            x = x.to(device)
            target = target.to(device)
            
            z0_k = ensemble.encode(x)
            
            preds_k = []
            for k in range(ensemble.k_models):
                z_k = apply_intervention(z0_k[k], intervened_genes, value=0.0)
                latent_pred, _ = RolloutEngine(ensemble.gnns[k], "deterministic").rollout(
                    z_k, edge_index, steps=2, log_trajectory=False
                )
                pred = ensemble.gnns[k].decode(latent_pred)
                preds_k.append(pred)
                
            pred_ensemble = torch.stack(preds_k, dim=0).mean(dim=0)
            total_loss += criterion(pred_ensemble, target).item()
            
    return total_loss / len(holdout_loader)

def extract_single_gene_interventions(dataset):
    """Filters base cells and strict single-gene interventions."""
    single_gene_vocab = set()
    cell_to_genes = {}
    valid_cells = []
    
    for i in range(len(dataset)):
        intervention = dataset.interventions[i]
        g1, g2 = intervention[0].item(), intervention[1].item()
        if g1 != -1 and (g2 == -1 or g1 == g2):
            single_gene_vocab.add(g1)
            cell_to_genes[i] = [g1]
            valid_cells.append(i)
        elif g1 == -1 and g2 == -1: # base cells
            cell_to_genes[i] = []
            valid_cells.append(i)
            
    return list(single_gene_vocab), cell_to_genes, valid_cells

def run_simulation(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Active Learning Simulation on {device}")
    
    # 1. Load Data
    full_dataset = NormanDataset(processed_dir="data/processed")
    edge_index = get_trrust_graph(processed_dir="data/processed").to(device)
    
    vocab, cell_targets, valid_cells = extract_single_gene_interventions(full_dataset)
    vocab = torch.tensor(vocab)
    
    # 2. Global Split (seed 42 for consistency)
    torch.manual_seed(42)
    shuffled_vocab = vocab[torch.randperm(len(vocab))].tolist()
    
    holdout_vocab = set(shuffled_vocab[:4])
    initial_train_vocab = set(shuffled_vocab[4:7])
    initial_pool_vocab = set(shuffled_vocab[7:])
    
    max_cells_per_gene = 80
    gene_to_cells = {}
    for g in vocab.tolist():
        c_list = [c for c in valid_cells if len(cell_targets[c]) > 0 and cell_targets[c][0] == g]
        rng = random.Random(42 + g)
        rng.shuffle(c_list)
        gene_to_cells[g] = c_list[:max_cells_per_gene]
    
    if args.dry_run:
        print("[DRY RUN] Subsampling data extensively...")
        initial_pool_vocab = set(list(initial_pool_vocab)[:5])
        held_cells = [c for g in holdout_vocab for c in gene_to_cells[g][:5]]
        K_queries = 2
        AL_steps = 2
        epochs = 1
    else:
        held_cells = [c for g in holdout_vocab for c in gene_to_cells[g]]
        K_queries = 2
        AL_steps = 4
        epochs = 2
        
    holdout_dataset = Subset(full_dataset, held_cells)
    holdout_loader = DataLoader(holdout_dataset, batch_size=128, shuffle=False)
    
    print(f"Absolute Holdout Set: {len(holdout_vocab)} genes, {len(held_cells)} cells")
    print(f"Initial Train Set: {len(initial_train_vocab)} genes, Pool Set: {len(initial_pool_vocab)} genes")
    
    # Get unperturbed base cells for computing metrics
    base_cells_indices = [c for c in valid_cells if len(cell_targets[c]) == 0][:50]
    base_cells_subset = Subset(full_dataset, base_cells_indices)
    base_cells_batch = next(iter(DataLoader(base_cells_subset, batch_size=50)))[0]
    
    strategies = ["random", "degree", "utraj_only", "atlas_full"]
    results = {s: [] for s in strategies}
    
    for strategy in strategies:
        print(f"\n=========================================")
        print(f"Starting Strategy: {strategy.upper()}")
        print(f"=========================================")
        
        current_train_vocab = set(initial_train_vocab)
        current_pool_vocab = list(initial_pool_vocab)
        
        for step in range(AL_steps + 1):
            train_cells = [c for g in current_train_vocab for c in gene_to_cells[g]]
            if args.dry_run: train_cells = train_cells[:20]
            
            train_loader = DataLoader(Subset(full_dataset, train_cells), batch_size=64, shuffle=True)
            
            print(f"\n--- [Step {step}] Training Ensemble (train_genes={len(current_train_vocab)}) ---")
            ensemble = train_ensemble_in_memory(
                train_loader, edge_index, full_dataset.num_genes, 
                num_epochs=epochs, device=device, verbose=(not args.dry_run)
            )
            
            simulator = RolloutEngine(ensemble, "deterministic")
            
            mse = evaluate_holdout(ensemble, holdout_loader, edge_index, simulator, device)
            results[strategy].append(mse)
            print(f"--- [Step {step}] Holdout MSE: {mse:.4f} ---")
            
            if step == AL_steps or len(current_pool_vocab) == 0:
                break
                
            print(f"--- [Step {step}] Acquiring Top-{K_queries} from pool of {len(current_pool_vocab)} ---")
            k_actual = min(K_queries, len(current_pool_vocab))
            
            if strategy == "random":
                rng = random.Random(42 + step * 100)
                selected = rng.sample(current_pool_vocab, k_actual)
            elif strategy == "degree":
                degrees = [(g, get_node_degree(g, edge_index)) for g in current_pool_vocab]
                degrees.sort(key=lambda x: x[1], reverse=True)
                selected = [g for g, d in degrees[:k_actual]]
            else:
                ablation = "utraj_only" if strategy == "utraj_only" else "full"
                u_traj_raw, z_mean_reps, hub_penalties = compute_candidate_metrics(
                    current_pool_vocab, ensemble, simulator, edge_index, 
                    base_cells_batch, apply_intervention, rollout_steps=2, device=device
                )
                selected, _ = rank_and_select(
                    current_pool_vocab, u_traj_raw, z_mean_reps, hub_penalties,
                    k=k_actual, lambda1=1.0, lambda2=1.0, ablation_mode=ablation
                )
                
            print(f"Selected: {selected}")
            
            for g in selected:
                current_pool_vocab.remove(g)
                current_train_vocab.add(g)
                
    print("\n=========================================")
    print("FINAL ACTIVE LEARNING RESULTS")
    print("=========================================")
    for strategy in strategies:
        errors = [f"{e:.4f}" for e in results[strategy]]
        print(f"Strategy: {strategy.ljust(12)} | Errors: {errors}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run with minimal data to test execution flow")
    args = parser.parse_args()
    
    run_simulation(args)
