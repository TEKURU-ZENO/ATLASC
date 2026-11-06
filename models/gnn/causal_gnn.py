import torch
import torch.nn as nn
from typing import Optional
from .message_passing import MessagePassingLayer

class CausalGNN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int = 1, layers_count: int = 3):
        super().__init__()
        self.layers = nn.ModuleList()
        
        current_in = in_channels
        for i in range(layers_count):
            self.layers.append(MessagePassingLayer(current_in, hidden_channels))
            current_in = hidden_channels
            
        # Predict gene expression from latent graph state
        # [batch_size, num_genes, hidden_channels] -> [batch_size, num_genes, out_channels]
        self.predictor = nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply causal message passing layers with residual connections and signed regulation.
        """
        for layer in self.layers:
            # Residual: z_{t+1} = z_t + f(z_t, G)
            z = z + layer(z, edge_index, edge_weight)
            
        return z
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Maps latent node states back to scalar expressions."""
        # [batch_size, num_genes, hidden_channels] -> [batch_size, num_genes]
        return self.predictor(z).squeeze(-1)
