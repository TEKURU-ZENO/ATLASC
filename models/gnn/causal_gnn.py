import torch
import torch.nn as nn
from .message_passing import MessagePassingLayer

class CausalGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, layers_count=3):
        super().__init__()
        self.layers = nn.ModuleList()
        
        current_in = in_channels
        for i in range(layers_count):
            self.layers.append(MessagePassingLayer(current_in, hidden_channels))
            current_in = hidden_channels
            
        # Predict gene expression from latent graph state
        # [batch_size, num_genes, hidden_channels] -> [batch_size, num_genes, out_channels]
        self.predictor = nn.Linear(hidden_channels, out_channels)

    def forward(self, z, edge_index):
        # Apply message passing layers with residual connection
        for layer in self.layers:
            # Residual: z_{t+1} = z_t + f(z_t)
            z = z + layer(z, edge_index)
            
        # Optional: For intermediate rollouts, we need the full latent state (3D).
        # For simplicity in rollout, GNN returns [batch, nodes, hidden]
        return z
        
    def decode(self, z):
        """Maps latent node states back to scalar expressions."""
        # [batch_size, num_genes, hidden_channels] -> [batch_size, num_genes]
        return self.predictor(z).squeeze(-1)
