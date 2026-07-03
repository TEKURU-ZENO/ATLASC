import sys
import os
import torch
from torch.utils.data import DataLoader
import numpy as np

# Add project root to sys path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.encoders.transcriptomic_encoder import TranscriptomicEncoder
from models.gnn.causal_gnn import CausalGNN
from models.ensemble import EnsembleWrapper
from causal.do_operator import apply_intervention
from utils.data_loader import NormanDataset, get_trrust_graph
from simulation.rollout_engine import RolloutEngine

def evaluate_uncertainty():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating Ensemble Uncertainty on device: {device}")
    
    latent_dim = 16
    rollout_steps = 3
    K_models = 3
    
    # 1. Load Dataset & Construct Seen/Unseen Splits
    full_dataset = NormanDataset(processed_dir="data/processed", use_cell_states=True)
    num_genes = full_dataset.num_genes
    
    # Same dataset seed as training to isolate unseen correctly
    torch.manual_seed(42) 
    
    # 1. Extract global intervention vocabulary
    all_interventions = full_dataset.interventions
    unique_genes = set()
    for b in range(all_interventions.size(0)):
        for idx in all_interventions[b]:
            if idx.item() != -1:
                unique_genes.add(idx.item())
                
    unique_genes = list(unique_genes)
    unique_genes_tensor = torch.tensor(unique_genes)
    shuffled_vocab = unique_genes_tensor[torch.randperm(len(unique_genes))]
    
    split_idx = int(0.9 * len(shuffled_vocab))
    train_vocab = set(shuffled_vocab[:split_idx].tolist())
    val_vocab = set(shuffled_vocab[split_idx:].tolist())
    
    train_indices_list = []
    val_indices_list = []
    
    for i in range(len(full_dataset)):
        has_val = False
        for idx in all_interventions[i]:
            if idx.item() in val_vocab:
                has_val = True
                break
        if has_val:
            val_indices_list.append(i)
        else:
            train_indices_list.append(i)
            
    train_tensor = torch.tensor(train_indices_list)
    val_tensor = torch.tensor(val_indices_list)
    
    # Evaluate on a large subset for statistical robustness
    seen_indices = train_tensor[torch.randperm(len(train_tensor))][:1500]
    unseen_indices = val_tensor[torch.randperm(len(val_tensor))][:500]
    
    seen_dataset = torch.utils.data.Subset(full_dataset, seen_indices)
    unseen_dataset = torch.utils.data.Subset(full_dataset, unseen_indices)
    
    seen_loader = DataLoader(seen_dataset, batch_size=100, shuffle=False)
    unseen_loader = DataLoader(unseen_dataset, batch_size=100, shuffle=False)
    
    edge_index = get_trrust_graph(processed_dir="data/processed").to(device)

    # 2. Load the mathematically distinct K models into the EnsembleWrapper
    encoders = []
    gnns = []
    for k in range(K_models):
        enc = TranscriptomicEncoder(num_genes=num_genes, latent_dim=latent_dim).to(device)
        gnn = CausalGNN(in_channels=latent_dim, hidden_channels=latent_dim, out_channels=1, layers_count=3).to(device)
        
        ckpt = torch.load(f"models/checkpoints/epistemic_member_k{k}.pt", map_location=device, weights_only=True)
        enc.load_state_dict(ckpt['encoder_state_dict'])
        gnn.load_state_dict(ckpt['gnn_state_dict'])
        
        # Keep models in train mode to activate MC Dropout for Uncertainty bounds
        enc.train()
        gnn.train()
        
        encoders.append(enc)
        gnns.append(gnn)
        
    ensemble = EnsembleWrapper(encoders, gnns)
    
    # 3. Polymorphic execution through the original RolloutEngine
    simulator = RolloutEngine(core_model=ensemble, execution_mode="deterministic")
    
    def evaluate_split(dataloader, split_name):
        u_traj_all = []
        d_t_all = []
        
        with torch.no_grad():
            for x, intervened_genes, target in dataloader:
                x = x.to(device)
                
                # z0_k: [K, batch, genes, latent]
                z0_k = ensemble.encode(x)
                
                # We need to apply interventions to EACH model's distinct latent space
                z_intervened_k = []
                for k in range(K_models):
                    z_k = apply_intervention(z0_k[k], intervened_genes, value=0.0)
                    z_intervened_k.append(z_k)
                z_intervened_k = torch.stack(z_intervened_k, dim=0)

                # Simulate robust trajectory 
                # trajectory: [T, K, batch, genes, latent]
                _, trajectory = simulator.rollout(
                    z_intervened_k, 
                    edge_index, 
                    steps=rollout_steps, 
                    log_trajectory=True,
                    enforce_stability=False
                )
                
                # IMPORTANT: The trajectory might be cloning identical pointers if not careful.
                # Compute Epistemic Metrics
                u_traj, d_t = ensemble.compute_epistemic_uncertainty(trajectory.clone())
                
                # u_traj shape: [batch, genes] -> compute mean across all genes for a single scalar
                u_cell = torch.mean(u_traj, dim=-1) # [batch]
                u_traj_all.append(u_cell.cpu().numpy())
                d_t_all.append(d_t)
                
        # Aggregate
        u_traj_all = np.concatenate(u_traj_all)
        d_t_mean = np.mean(d_t_all, axis=0)
        
        print(f"\n--- {split_name} Split Analysis ---")
        print(f"Mean Cell Trajectory Uncertainty (U_traj): {np.mean(u_traj_all):.4f} +/- {np.std(u_traj_all):.4f}")
        print(f"Mean Pairwise Disagreement (t=1,2,3...): {[round(d, 4) for d in d_t_mean]}")
        
        return u_traj_all
        
    print("\nRunning Forward Prediction on Ensemble...")
    u_seen = evaluate_split(seen_loader, "Seen (Training)")
    u_unseen = evaluate_split(unseen_loader, "Unseen (Validation)")
    
    # 4. Statistical Test
    from scipy import stats
    t_stat, p_val = stats.ttest_ind(u_unseen, u_seen)
    
    print("\n=============================================")
    print("🎓 Epistemic Generalization Verdict")
    print("=============================================")
    print(f"Difference (Unseen - Seen): {np.mean(u_unseen) - np.mean(u_seen):.4f}")
    print(f"Statistical Significance (p-value): {p_val:.4e}")
    if p_val < 0.05 and np.mean(u_unseen) > np.mean(u_seen):
        print("✅ SUCCESS: The ensemble correctly models epistemic uncertainty! Unseen perturbations have statistically higher variance.")
    else:
        print("❌ FAILED: The ensemble does not differentiate unseen perturbations from seen ones.")

if __name__ == "__main__":
    evaluate_uncertainty()
