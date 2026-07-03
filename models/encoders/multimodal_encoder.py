"""
multimodal_encoder.py
---------------------
Fuses transcriptomic and epigenomic latent representations into a single
joint embedding that is passed to the GNN.

Fusion strategy:
  transcriptomic_z  [batch, genes, latent_dim]
  epigenomic_z      [batch, genes, latent_dim]
  → concat → [batch, genes, 2*latent_dim]
  → gated cross-modality attention
  → [batch, genes, latent_dim]

When only one modality is available, the missing modality is filled with
zeros so the module degrades gracefully without crashing.
"""

import torch
import torch.nn as nn
from .base_encoder import BaseEncoder
from .transcriptomic_encoder import TranscriptomicEncoder
from .epigenomic_encoder import EpigenomicEncoder


class CrossModalGate(nn.Module):
    """
    Gated fusion: learns which modality to trust per gene per sample.
    gate = sigmoid(W * [z_t || z_e])
    fused = gate * z_t + (1 - gate) * z_e
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim),
            nn.Sigmoid(),
        )

    def forward(
        self, z_t: torch.Tensor, z_e: torch.Tensor
    ) -> torch.Tensor:
        # z_t, z_e : [batch, genes, latent_dim]
        concat = torch.cat([z_t, z_e], dim=-1)   # [batch, genes, 2*latent]
        g = self.gate(concat)                     # [batch, genes, latent]
        return g * z_t + (1.0 - g) * z_e


class MultimodalEncoder(BaseEncoder):
    """
    Joint encoder for transcriptomic + epigenomic data.

    Args:
        num_genes         : number of genes (graph nodes)
        n_epigenomic_feats: epigenomic features per gene (default 1)
        latent_dim        : shared latent dimension
        hidden_dim        : MLP hidden width in sub-encoders
        dropout           : dropout applied inside sub-encoders
    """

    def __init__(
        self,
        num_genes: int,
        n_epigenomic_feats: int = 1,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.transcriptomic_enc = TranscriptomicEncoder(
            num_genes=num_genes,
            latent_dim=latent_dim,
        )
        self.epigenomic_enc = EpigenomicEncoder(
            num_genes=num_genes,
            n_features=n_epigenomic_feats,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.fusion = CrossModalGate(latent_dim)

        # Final projection to keep dimensionality consistent
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(
        self,
        x_transcriptomic: torch.Tensor,
        x_epigenomic: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x_transcriptomic : [batch, num_genes]
            x_epigenomic     : [batch, num_genes, n_epigenomic_feats]  (optional)

        Returns:
            z : [batch, num_genes, latent_dim]
        """
        # Transcriptomic branch: [batch, genes, latent]
        z_t = self.transcriptomic_enc(x_transcriptomic)

        # Epigenomic branch (zero-filled if absent)
        if x_epigenomic is None:
            z_e = torch.zeros_like(z_t)
        else:
            z_e = self.epigenomic_enc(x_epigenomic)

        # Gated fusion
        z_fused = self.fusion(z_t, z_e)

        # Final projection + layer-norm
        return self.proj(z_fused)

    # Compatibility alias so EnsembleWrapper.encode() works
    def encode(self, x_transcriptomic, x_epigenomic=None):
        return self.forward(x_transcriptomic, x_epigenomic)
