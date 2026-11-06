import torch
import torch.nn as nn
from itertools import combinations
from typing import Optional

class EnsembleWrapper(nn.Module):
    """
    Polymorphic wrapper for K independent models.
    It exposes the EXACT same forward and decode interface as a single CausalGNN,
    ensuring the RolloutEngine requires ZERO architectural changes. 
    """
    def __init__(self, encoders, gnns):
        super().__init__()
        assert len(encoders) == len(gnns), "Must have equal number of encoders and GNNs."
        self.k_models = len(encoders)
        
        # Register them as submodules so they move to target devices together
        self.encoders = nn.ModuleList(encoders)
        self.gnns = nn.ModuleList(gnns)

    def encode(self, x: torch.Tensor, x_epigenomic: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Maps [batch, genes] -> [K, batch, genes, latent]
        Supports optional x_epigenomic for multimodal encoders.
        """
        z_k = []
        for encoder in self.encoders:
            if x_epigenomic is not None:
                try:
                    z = encoder(x, x_epigenomic)
                except TypeError:
                    z = encoder(x)
            else:
                z = encoder(x)
            z_k.append(z)
        return torch.stack(z_k, dim=0)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps [K, batch, genes, latent] -> [K, batch, genes]
        """
        # z is shape [K, ...], so we decode each k independently
        out_k = []
        for k in range(self.k_models):
            out_k.append(self.gnns[k].decode(z[k]))
        return torch.stack(out_k, dim=0)

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Polymorphic transition function.
        Receives z: [K, batch, nodes, latent]
        Returns z_{t+1}: [K, batch, nodes, latent]
        """
        z_next_k = []
        for k in range(self.k_models):
            # Pass each ensemble member's state through its own GNN
            try:
                z_next_k.append(self.gnns[k](z[k], edge_index, edge_weight=edge_weight))
            except TypeError:
                z_next_k.append(self.gnns[k](z[k], edge_index))
        return torch.stack(z_next_k, dim=0)
        
    def compute_epistemic_uncertainty(self, trajectory):
        """
        Takes a full rollout trajectory of shape [T, K, batch, genes, latent]
        and computes the Time-Weighted Trajectory Variance & Pairwise Disagreement.
        
        Returns:
            U_traj: [batch, genes] scalar uncertainty score
            D_t: [T] array of mean pairwise disagreements at each timestep
        """
        T, K, B, G, L = trajectory.shape
        
        # 1. Time-Weighted Variance (U_traj)
        # Var[z_t] across the K models: [T, batch, genes, latent]
        var_z = torch.var(trajectory, dim=1, unbiased=True) 
        
        # Sum variance across latent dimension to get scalar per node: [T, batch, genes]
        var_node = torch.sum(var_z, dim=-1)
        
        # Compute time weights alpha_t = t / T
        # t is 1-indexed for the weight: (1/T, 2/T... T/T)
        device = trajectory.device
        alpha_t = torch.arange(1, T + 1, dtype=torch.float32, device=device) / T
        
        # Weight the variance: [T, batch, genes] * [T, 1, 1]
        weighted_var = var_node * alpha_t.view(-1, 1, 1)
        
        # Aggregate over time: [batch, genes]
        U_traj = torch.sum(weighted_var, dim=0)
        
        # 2. Pairwise Disagreement Metric (D_t)
        # D_t = 1 / (K*(K-1)) * sum_{i < j} ||z_t^(i) - z_t^(j)||
        D_t = []
        for t in range(T):
            disagreement_sum = 0.0
            pairs = 0
            
            # trajectory[t] shape: [K, batch, genes, latent]
            for i, j in combinations(range(K), 2):
                z_i = trajectory[t, i]
                z_j = trajectory[t, j]
                
                # Mean L2 norm difference across the batch & genes
                diff_norm = torch.norm(z_i - z_j, dim=-1).mean()
                disagreement_sum += diff_norm
                pairs += 1
                
            D_t.append((disagreement_sum / pairs).item() if pairs > 0 else 0)
            
        return U_traj, D_t
