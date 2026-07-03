"""
epigenomic_encoder.py
---------------------
Encodes epigenomic features (e.g. ATAC-seq peak accessibility, histone marks,
or chromatin-state tracks) to per-gene latent vectors.

Input  : [batch, num_genes, n_epigenomic_features]
Output : [batch, num_genes, latent_dim]

If n_epigenomic_features == 1 (a single accessibility score per gene) the
module degrades gracefully to a linear projection, making it backward-
compatible with datasets that only expose one epigenomic track.
"""

import torch
import torch.nn as nn
from .base_encoder import BaseEncoder


class EpigenomicEncoder(BaseEncoder):
    def __init__(
        self,
        num_genes: int,
        n_features: int,
        latent_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        """
        Args:
            num_genes      : total number of genes (graph nodes)
            n_features     : number of epigenomic features per gene
            latent_dim     : output dimensionality per gene
            hidden_dim     : width of intermediate MLP layer
            dropout        : dropout probability
        """
        super().__init__()
        self.num_genes = num_genes
        self.latent_dim = latent_dim

        # Shared (across genes) feature MLP
        self.feature_mlp = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Gene-specific identity embedding (learnable offset)
        self.gene_embeddings = nn.Parameter(
            torch.randn(1, num_genes, latent_dim) * 0.05
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.feature_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.3)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [batch, num_genes, n_epigenomic_features]
        Returns:
            z : [batch, num_genes, latent_dim]
        """
        # [batch, genes, n_features] -> [batch, genes, latent_dim]
        feat = self.feature_mlp(x)

        # Add gene-specific identity signal
        z = feat + self.gene_embeddings.expand(x.size(0), -1, -1)
        return z
