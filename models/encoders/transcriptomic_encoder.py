import torch
import torch.nn as nn
from .base_encoder import BaseEncoder

class TranscriptomicEncoder(BaseEncoder):
    def __init__(self, num_genes, latent_dim):
        """
        Maps (batch_size, num_genes) -> (batch_size, num_genes, latent_dim)
        Each gene gets its own latent representation for GNN processing.
        """
        super().__init__()
        self.num_genes = num_genes
        self.latent_dim = latent_dim
        
        # Gene-specific embeddings initialized small to prevent early divergence
        self.gene_embeddings = nn.Parameter(torch.randn(1, num_genes, latent_dim) * 0.1)
        
        # Shared MLP to process scalar expression
        self.expression_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, latent_dim)
        )

    def forward(self, x):
        # x: [batch_size, num_genes]
        batch_size = x.size(0)
        
        # [batch_size, num_genes, 1]
        x_expanded = x.unsqueeze(-1)
        
        # [batch_size, num_genes, latent_dim]
        expr_features = self.expression_mlp(x_expanded)
        
        # Combine expression features with gene identity
        z = expr_features + self.gene_embeddings.expand(batch_size, -1, -1)
        
        return z
