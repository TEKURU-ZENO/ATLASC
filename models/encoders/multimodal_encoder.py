"""
multimodal_encoder.py
---------------------
Fuses transcriptomic (scRNA) and epigenomic (scATAC chromatin accessibility)
latent representations into a single joint biological embedding passed to the GNN.

Fusion strategies supported:
  1. "cross_attention": Multi-head cross-attention where RNA queries attend to ATAC
     chromatin accessibility keys/values with residual connection and feed-forward refinement.
  2. "gate": Gated linear fusion learning per-gene modality confidence weights.
  3. "hybrid": Cross-modal attention followed by adaptive confidence gating.

Gracefully handles cases when chromatin accessibility data is absent by zero-filling,
guaranteeing single-modality backward compatibility.
"""

from typing import Optional
import torch
import torch.nn as nn
from .base_encoder import BaseEncoder
from .transcriptomic_encoder import TranscriptomicEncoder
from .epigenomic_encoder import EpigenomicEncoder


class CrossModalAttention(nn.Module):
    """
    Multi-head cross-modality attention mechanism.
    Allows transcriptomic embeddings (Query) to attend dynamically to
    chromatin accessibility / epigenetic peak embeddings (Key, Value),
    capturing promoter/enhancer openness constraints on transcription.
    """

    def __init__(self, latent_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # Ensure num_heads divides latent_dim; fallback to 1 if latent_dim is small
        if latent_dim % num_heads != 0:
            num_heads = 1
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def forward(self, z_rna: torch.Tensor, z_atac: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_rna : [batch, genes, latent_dim]
            z_atac: [batch, genes, latent_dim]
        Returns:
            fused : [batch, genes, latent_dim]
        """
        # Cross-attention: Query = RNA, Key/Value = ATAC
        attn_out, _ = self.cross_attn(query=z_rna, key=z_atac, value=z_atac)
        x = self.norm1(z_rna + attn_out)
        out = self.norm2(x + self.ffn(x))
        return out


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
    Joint encoder for transcriptomic (scRNA-seq) + epigenomic (scATAC-seq) data.

    Args:
        num_genes         : number of genes (graph nodes)
        n_epigenomic_feats: epigenomic features per gene (e.g. peak accessibility, default 1)
        latent_dim        : shared latent dimension
        hidden_dim        : MLP hidden width in sub-encoders
        dropout           : dropout applied inside sub-encoders
        fusion_strategy   : "cross_attention", "gate", or "hybrid"
        num_heads         : number of attention heads for cross-attention
    """

    def __init__(
        self,
        num_genes: int,
        n_epigenomic_feats: int = 1,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        fusion_strategy: str = "cross_attention",
        num_heads: int = 4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.fusion_strategy = fusion_strategy

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

        if fusion_strategy in ("cross_attention", "hybrid"):
            self.cross_attn = CrossModalAttention(latent_dim, num_heads=num_heads, dropout=dropout)
        if fusion_strategy in ("gate", "hybrid"):
            self.gate = CrossModalGate(latent_dim)

        # Final projection to keep dimensionality consistent
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(
        self,
        x_transcriptomic: torch.Tensor,
        x_epigenomic: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x_transcriptomic : [batch, num_genes]
            x_epigenomic     : [batch, num_genes, n_epigenomic_feats] (optional)

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

        # Apply selected fusion strategy
        if self.fusion_strategy == "cross_attention":
            z_fused = self.cross_attn(z_t, z_e)
        elif self.fusion_strategy == "gate":
            z_fused = self.gate(z_t, z_e)
        elif self.fusion_strategy == "hybrid":
            z_attn = self.cross_attn(z_t, z_e)
            z_fused = self.gate(z_t, z_attn)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

        # Final projection + layer-norm
        return self.proj(z_fused)

    # Compatibility alias so EnsembleWrapper.encode() works
    def encode(self, x_transcriptomic, x_epigenomic=None):
        return self.forward(x_transcriptomic, x_epigenomic)
