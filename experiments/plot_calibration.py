import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.encoders.transcriptomic_encoder import TranscriptomicEncoder
from models.gnn.causal_gnn import CausalGNN
from models.ensemble import EnsembleWrapper
from utils.data_loader import NormanDataset, get_trrust_graph
from simulation.rollout_engine import RolloutEngine
from causal.do_operator import apply_intervention

def plot_calibration():
    print("Generating Calibration Curve...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    K_models = 3
    
    # 1. Load Validation Dataset
    full_dataset = NormanDataset(processed_dir="data/processed", use_cell_states=True)
    num_genes = full_dataset.num_genes
    
    # Fast evaluation subset (1000 cells)
    torch.manual_seed(42)
    eval_indices = torch.randperm(len(full_dataset))[:1000]
    eval_dataset = torch.utils.data.Subset(full_dataset, eval_indices)
    eval_loader = DataLoader(eval_dataset, batch_size=100, shuffle=False)
    
    edge_index = get_trrust_graph(processed_dir="data/processed").to(device)

    # 2. Load Ensemble
    encoders = []
    gnns = []
    for k in range(K_models):
        enc = TranscriptomicEncoder(num_genes=num_genes, latent_dim=16).to(device)
        gnn = CausalGNN(in_channels=16, hidden_channels=16, layers_count=3).to(device)
        
        ckpt = torch.load(f"models/checkpoints/epistemic_member_k{k}.pt", map_location=device, weights_only=True)
        enc.load_state_dict(ckpt['encoder_state_dict'])
        gnn.load_state_dict(ckpt['gnn_state_dict'])
        
        # MC Dropout
        enc.train()
        gnn.train()
        
        encoders.append(enc)
        gnns.append(gnn)
        
    ensemble = EnsembleWrapper(encoders, gnns)
    rollout_engine = RolloutEngine(core_model=ensemble, execution_mode="deterministic")

    all_uncertainties = []
    all_mses = []

    with torch.no_grad():
        for batch_idx, (x_b, intervened_genes, targets) in enumerate(eval_loader):
            x_b = x_b.to(device)
            targets = targets.to(device)
            
            # z0_k: [K, batch, genes, latent]
            z0_k = ensemble.encode(x_b)
            # Apply intervention natively per-batch
            z_intervened_k = apply_intervention(z0_k, intervened_genes, value=0.0)
            
            # Simulate robust trajectory 
            # trajectory: [T, K, batch, genes, latent]
            _, trajectory = rollout_engine.rollout(
                z_intervened_k, 
                edge_index, 
                steps=3, 
                log_trajectory=True,
                enforce_stability=False
            )
            
            # Compute Epistemic Metrics
            u_traj_full, d_t = ensemble.compute_epistemic_uncertainty(trajectory.clone())
            
            # u_traj shape: [batch, genes] -> compute mean across all genes for a single scalar
            u_cell = torch.mean(u_traj_full, dim=-1) # [batch]
            
            # Compute empirical error
            # trajectory[-1] is [K, Batch, Genes, Latent]
            final_z_k = trajectory[-1]
            final_pred_k = ensemble.decode(final_z_k) # -> [K, Batch, Genes]
            mean_pred = final_pred_k.mean(dim=0)      # -> [Batch, Genes]
            
            # MSE per cell
            mse_per_cell = ((mean_pred - targets)**2).mean(dim=1)
            
            all_uncertainties.append(u_cell.cpu().numpy())
            all_mses.append(mse_per_cell.cpu().numpy())

    uncertainties = np.concatenate(all_uncertainties)
    mses = np.concatenate(all_mses)

    # 3. Binning into Deciles
    num_bins = 10
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(uncertainties, percentiles)
    
    bin_centers = []
    bin_mses = []
    bin_errors = []
    
    for i in range(num_bins):
        mask = (uncertainties >= bin_edges[i]) & (uncertainties <= bin_edges[i+1])
        if np.sum(mask) > 0:
            bin_centers.append(np.mean(uncertainties[mask]))
            bin_mses.append(np.mean(mses[mask]))
            # Standard error of the mean
            bin_errors.append(np.std(mses[mask]) / np.sqrt(np.sum(mask)))

    # 4. Plot
    plt.figure(figsize=(8, 6))
    plt.errorbar(bin_centers, bin_mses, yerr=bin_errors, fmt='-o', color='royalblue', ecolor='lightcoral', elinewidth=3, capsize=5, markerfacecolor='darkblue', markersize=8)
    
    plt.title('Phase 4: Epistemic Calibration Curve', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Predicted Trajectory Uncertainty ($U_{traj}$)', fontsize=14)
    plt.ylabel('Empirical Mean Squared Error (MSE)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add a trend line
    z = np.polyfit(bin_centers, bin_mses, 1)
    p = np.poly1d(z)
    plt.plot(bin_centers, p(bin_centers), "r--", alpha=0.8, linewidth=2, label=f"Trend (Slope: {z[0]:.4f})")
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    os.makedirs('experiments/results', exist_ok=True)
    out_path = 'experiments/results/calibration_curve.png'
    plt.savefig(out_path, dpi=300)
    print(f"Calibration curve saved mathematically perfectly to {out_path}.")

if __name__ == "__main__":
    plot_calibration()
