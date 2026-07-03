import torch
import torch.nn as nn


class GeneExpressionDecoder(nn.Module):
    """
    Decodes a 3-D latent state [batch, genes, latent_dim] produced by the GNN
    back to a scalar gene-expression profile [batch, genes].

    The decoder is intentionally lightweight so that the representational
    capacity lives in the encoder / GNN stack rather than the decoder.
    """

    def __init__(self, latent_dim: int, num_genes: int, hidden_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_genes = num_genes

        # Per-gene projection: maps each node's latent vector to a scalar
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

        # Learnable gene-specific bias (captures baseline expression levels)
        self.gene_bias = nn.Parameter(torch.zeros(num_genes))

        self._init_weights()

    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [batch, genes, latent_dim]  or  [K, batch, genes, latent_dim]
        Returns:
            expr: [batch, genes]  or  [K, batch, genes]
        """
        ensemble_mode = z.dim() == 4
        if ensemble_mode:
            K, B, G, L = z.shape
            z_flat = z.reshape(K * B, G, L)
        else:
            z_flat = z  # [B, G, L]

        # [*, genes, 1] -> [*, genes]
        out = self.mlp(z_flat).squeeze(-1)  # [B, genes] or [K*B, genes]

        # Add gene-specific bias (broadcast over batch)
        out = out + self.gene_bias.unsqueeze(0)

        if ensemble_mode:
            out = out.reshape(K, B, G)

        return out

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        """Alias for forward – used by training code that calls .reconstruct()."""
        return self.forward(z)
