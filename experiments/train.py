import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
from torch.utils.data import DataLoader

# Add project root to sys path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.encoders.transcriptomic_encoder import TranscriptomicEncoder
from models.gnn.causal_gnn import CausalGNN
from causal.do_operator import apply_intervention
from utils.data_loader import NormanDataset, get_trrust_graph
from simulation.rollout_engine import RolloutEngine

# Add project root to sys path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.encoders.transcriptomic_encoder import TranscriptomicEncoder
from models.gnn.causal_gnn import CausalGNN
from causal.do_operator import apply_intervention
from utils.data_loader import NormanDataset, get_trrust_graph
from simulation.rollout_engine import RolloutEngine

def run_training():
    num_epochs = 6
    batch_size = 128
    latent_dim = 16
    lr = 0.001
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Data (Fast PyTorch Version)
    full_dataset = NormanDataset(processed_dir="data/processed")
    
    # Structural Hold-Out Split (By Intervention Vocabulary, Not Cells)
    # Important: The dataset split must be the same for all K models.
    torch.manual_seed(42) # Dataset split seed
    
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
    
    # 90% Train Vocab, 10% Hold-out Vocab
    split_idx = int(0.9 * len(shuffled_vocab))
    train_vocab = set(shuffled_vocab[:split_idx].tolist())
    val_vocab = set(shuffled_vocab[split_idx:].tolist())
    
    print(f"Structural Split: {len(train_vocab)} Seen Genes | {len(val_vocab)} Held-Out Unseen Genes")
    
    # 2. Assign cells based on their structural payload
    train_indices_list = []
    val_indices_list = []
    
    for i in range(len(full_dataset)):
        # Check if cell has any hold-out genes
        has_val = False
        cell_interventions = all_interventions[i]
        for idx in cell_interventions:
            if idx.item() in val_vocab:
                has_val = True
                break
                
        if has_val:
            val_indices_list.append(i)
        else:
            train_indices_list.append(i)
            
    # Cap sizes for fast epoch iteration
    torch.manual_seed(42)
    train_tensor = torch.tensor(train_indices_list)
    val_tensor = torch.tensor(val_indices_list)
    
    train_indices = train_tensor[torch.randperm(len(train_tensor))][:800]
    val_indices = val_tensor[torch.randperm(len(val_tensor))][:200]
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    num_genes = full_dataset.num_genes
    
    # Graph Topology (True Biological GRN)
    edge_index = get_trrust_graph(processed_dir="data/processed").to(device)
    
    # Ensure checkpoints directory exists
    os.makedirs("models/checkpoints", exist_ok=True)
    
    K_models = 3
    print(f"Starting Independent Sequential Training of {K_models} models.")
    
    for k in range(K_models):
        print(f"\n======================================")
        print(f"Training Ensemble Member K={k+1}/{K_models}...")
        print(f"======================================")
        
        # Set distinct initialization seed
        torch.manual_seed(100 + k) 
        
        # Models & Simulation Engine (Independently Initialized)
        encoder = TranscriptomicEncoder(num_genes=num_genes, latent_dim=latent_dim).to(device)
        gnn = CausalGNN(in_channels=latent_dim, hidden_channels=latent_dim, out_channels=1, layers_count=3).to(device)
        
        # Wrap GNN in our robust rollout engine
        simulator = RolloutEngine(core_model=gnn, execution_mode="deterministic")
        
        optimizer = optim.Adam(list(encoder.parameters()) + list(gnn.parameters()), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(num_epochs):
            # Curriculum Schedule
            if epoch < 2:
                current_steps = 1
            elif epoch < 4:
                current_steps = 2
            else:
                current_steps = 3
                
            print(f"\n--- Epoch {epoch}/{num_epochs-1} | Rollout T={current_steps} ---")
            
            epoch_loss = 0.0
            encoder.train()
            gnn.train()
            
            for batch_idx, (x, intervened_genes, target) in enumerate(train_loader):
                x = x.to(device)
                target = target.to(device)
                
                optimizer.zero_grad()
                
                # Map basal expression to node states
                z0 = encoder(x)
                
                # Apply instant perturbation at t=0
                z_intervened = apply_intervention(z0, intervened_genes, value=0.0)
                
                # Rollout simulation forward T steps natively
                latent_pred, trajectory = simulator.rollout(
                    z_intervened, 
                    edge_index, 
                    steps=current_steps, 
                    log_trajectory=True, 
                    enforce_stability=True
                )
                
                # Map latents back to scalar gene expression
                pred = gnn.decode(latent_pred)
                
                # Reconstruction Loss
                recon_loss = criterion(pred, target)
                
                # Latent Norm Regularization Penalty
                if trajectory is not None:
                    norm_penalty = torch.mean(torch.norm(trajectory, dim=-1))
                    lambda_reg = 1e-2
                    loss = recon_loss + lambda_reg * norm_penalty
                    
                    with torch.no_grad():
                        norms_per_step = [torch.mean(torch.norm(trajectory[t], dim=-1)).item() for t in range(trajectory.size(0))]
                else:
                    loss = recon_loss
                    norms_per_step = []
                    
                loss.backward()
                
                # Gradient Norm tracking
                total_norm = 0.0
                for p in list(encoder.parameters()) + list(gnn.parameters()):
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                
                optimizer.step()
                
                epoch_loss += recon_loss.item()
                
                if batch_idx % 2 == 0:
                    print(f"Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Grad Norm: {total_norm:.2f} | Z-Norms: {[round(n, 2) for n in norms_per_step]}")
                
            avg_train_loss = epoch_loss / len(train_loader)
            
            # Validation Phase
            encoder.eval()
            gnn.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, intervened_genes, target in val_loader:
                    x = x.to(device)
                    target = target.to(device)
                    
                    z0 = encoder(x)
                    z_intervened = apply_intervention(z0, intervened_genes, value=0.0)
                    latent_pred, _ = simulator.rollout(z_intervened, edge_index, steps=current_steps, log_trajectory=False)
                    pred = gnn.decode(latent_pred)
                    
                    val_loss += criterion(pred, target).item()
                    
            avg_val_loss = val_loss / max(1, len(val_loader))
            print(f"Epoch {epoch} Summary - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
        # Save independent models after curriculum
        checkpoint_path = f"models/checkpoints/epistemic_member_k{k}.pt"
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'gnn_state_dict': gnn.state_dict()
        }, checkpoint_path)
        print(f"Model K={k+1} Checkpoint Saved -> {checkpoint_path}")
        
    print("Sequential Ensemble Generation Finished! K models successfully trained independently.")

if __name__ == "__main__":
    run_training()
