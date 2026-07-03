"""
temporal_gnn.py
---------------
Temporal variant of the CausalGNN.

Extends CausalGNN with a GRU-based temporal memory so that latent node
states carry information from *previous* rollout steps.  This is useful
when modeling oscillatory or delayed regulatory effects.

Architecture per rollout step:
  z_prev  [batch, genes, latent]   (memory from last step – h_t-1 of GRU)
  input   [batch, genes, latent]   (current transcriptomic encoding)
  → MessagePassing layers → z_spatial
  → GRU cell update: h_t = GRU(z_spatial, h_t-1)
  → predictor head → expression delta
"""

import torch
import torch.nn as nn
from .message_passing import MessagePassingLayer


class TemporalGNN(nn.Module):
    """
    GRU-augmented causal GNN that retains temporal memory across rollout steps.

    Args:
        in_channels    : input latent dimension
        hidden_channels: GNN message-passing and GRU hidden size
        out_channels   : decoder output per gene (default 1 → scalar expression)
        layers_count   : number of MessagePassing layers per step
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int = 1,
        layers_count: int = 3,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels

        # Spatial message-passing stack
        self.mp_layers = nn.ModuleList()
        current_in = in_channels
        for _ in range(layers_count):
            self.mp_layers.append(MessagePassingLayer(current_in, hidden_channels))
            current_in = hidden_channels

        # GRU cell operates on flattened [batch*genes, hidden] vectors
        self.gru_cell = nn.GRUCell(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
        )

        # Expression predictor
        self.predictor = nn.Linear(hidden_channels, out_channels)

        # LayerNorm after GRU update
        self.norm = nn.LayerNorm(hidden_channels)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.predictor.weight, gain=0.3)
        nn.init.zeros_(self.predictor.bias)

    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Single rollout step.

        Args:
            z          : [batch, genes, in_channels]  current latent state
            edge_index : [2, E]
            h          : [batch, genes, hidden_channels]  GRU hidden (None = zeros)

        Returns:
            z_next     : [batch, genes, hidden_channels]  updated latent (new h)
        """
        B, G, _ = z.shape

        # 1. Spatial message passing with residual connections
        x = z
        for layer in self.mp_layers:
            x = x + layer(x, edge_index)

        # 2. GRU update (operate per-node)
        x_flat = x.reshape(B * G, self.hidden_channels)       # [B*G, H]

        if h is None:
            h_flat = torch.zeros(B * G, self.hidden_channels, device=z.device)
        else:
            h_flat = h.reshape(B * G, self.hidden_channels)

        h_next_flat = self.gru_cell(x_flat, h_flat)           # [B*G, H]
        h_next = h_next_flat.reshape(B, G, self.hidden_channels)

        # 3. Normalise
        z_next = self.norm(h_next)
        return z_next

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps latent → scalar gene expression.

        Args:
            z : [batch, genes, hidden_channels]  or  [K, batch, genes, hidden_channels]

        Returns:
            expr : [batch, genes]  or  [K, batch, genes]
        """
        return self.predictor(z).squeeze(-1)

    def init_hidden(self, batch_size: int, num_genes: int, device: torch.device) -> torch.Tensor:
        """Convenience method to create a zero-initialised hidden state."""
        return torch.zeros(batch_size, num_genes, self.hidden_channels, device=device)
