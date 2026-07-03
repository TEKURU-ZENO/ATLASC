"""
model_loader.py
---------------
Centralised checkpoint load / save utilities for ATLAS models.

Supports:
  - Single-model checkpoints  (encoder + GNN)
  - Ensemble checkpoints      (K members)
  - Partial / forward-compatible loading with key-mismatch warnings
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from models.encoders.transcriptomic_encoder import TranscriptomicEncoder
from models.encoders.multimodal_encoder import MultimodalEncoder
from models.gnn.causal_gnn import CausalGNN
from models.gnn.temporal_gnn import TemporalGNN
from models.ensemble import EnsembleWrapper


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _load_state(path: str, device: torch.device) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"[model_loader] Checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=True)


def _apply_state(module: nn.Module, state_dict: dict, name: str = "") -> None:
    """Load state dict with informative warnings on key mismatches."""
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[model_loader] {name} – missing keys: {missing}")
    if unexpected:
        print(f"[model_loader] {name} – unexpected keys: {unexpected}")


# ---------------------------------------------------------------------------
# Single-model checkpoint I/O
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    encoder: nn.Module,
    gnn: nn.Module,
    epoch: int,
    metrics: Optional[dict] = None,
) -> None:
    """Save a single (encoder, GNN) pair to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "encoder_state_dict": encoder.state_dict(),
            "gnn_state_dict": gnn.state_dict(),
            "metrics": metrics or {},
        },
        path,
    )
    print(f"[model_loader] Saved checkpoint → {path}  (epoch {epoch})")


def load_checkpoint(
    path: str,
    encoder: nn.Module,
    gnn: nn.Module,
    device: torch.device | str = "cpu",
) -> dict:
    """
    Load weights into an existing (encoder, GNN) pair.

    Returns:
        The full checkpoint dict (contains 'epoch', 'metrics', …).
    """
    device = torch.device(device) if isinstance(device, str) else device
    ckpt = _load_state(path, device)
    _apply_state(encoder, ckpt["encoder_state_dict"], name=f"encoder@{path}")
    _apply_state(gnn,     ckpt["gnn_state_dict"],     name=f"gnn@{path}")
    encoder.to(device).eval()
    gnn.to(device).eval()
    print(f"[model_loader] Loaded checkpoint ← {path}  (epoch {ckpt.get('epoch', '?')})")
    return ckpt


# ---------------------------------------------------------------------------
# Ensemble checkpoint I/O
# ---------------------------------------------------------------------------

def save_ensemble(
    base_dir: str,
    ensemble: EnsembleWrapper,
    epoch: int,
    metrics: Optional[dict] = None,
) -> None:
    """Save all K members of an ensemble."""
    os.makedirs(base_dir, exist_ok=True)
    for k in range(ensemble.k_models):
        path = os.path.join(base_dir, f"epistemic_member_k{k}.pt")
        save_checkpoint(
            path,
            encoder=ensemble.encoders[k],
            gnn=ensemble.gnns[k],
            epoch=epoch,
            metrics=metrics,
        )


def load_ensemble(
    base_dir: str,
    num_genes: int,
    latent_dim: int = 16,
    k_models: int = 3,
    gnn_layers: int = 3,
    use_temporal: bool = False,
    device: torch.device | str = "cpu",
) -> EnsembleWrapper:
    """
    Re-instantiate and reload an ensemble from individual member checkpoints.

    Args:
        base_dir    : directory containing  epistemic_member_k{k}.pt  files
        num_genes   : gene vocabulary size (must match what was trained)
        latent_dim  : latent dimension
        k_models    : number of ensemble members
        gnn_layers  : number of message-passing layers
        use_temporal: if True, use TemporalGNN instead of CausalGNN
        device      : target device

    Returns:
        EnsembleWrapper (all k models loaded and in eval mode)
    """
    device = torch.device(device) if isinstance(device, str) else device
    encoders: List[nn.Module] = []
    gnns:     List[nn.Module] = []

    GNNClass = TemporalGNN if use_temporal else CausalGNN

    for k in range(k_models):
        path = os.path.join(base_dir, f"epistemic_member_k{k}.pt")
        enc = TranscriptomicEncoder(num_genes=num_genes, latent_dim=latent_dim)
        gnn = GNNClass(
            in_channels=latent_dim,
            hidden_channels=latent_dim,
            out_channels=1,
            layers_count=gnn_layers,
        )
        load_checkpoint(path, enc, gnn, device=device)
        encoders.append(enc)
        gnns.append(gnn)

    return EnsembleWrapper(encoders, gnns).to(device)


# ---------------------------------------------------------------------------
# Convenience factory: build fresh (untrained) ensemble
# ---------------------------------------------------------------------------

def build_ensemble(
    num_genes: int,
    latent_dim: int = 16,
    k_models: int = 3,
    gnn_layers: int = 3,
    use_temporal: bool = False,
    device: torch.device | str = "cpu",
) -> EnsembleWrapper:
    """Build a freshly-initialised EnsembleWrapper (for training)."""
    device = torch.device(device) if isinstance(device, str) else device
    GNNClass = TemporalGNN if use_temporal else CausalGNN

    encoders = [
        TranscriptomicEncoder(num_genes=num_genes, latent_dim=latent_dim).to(device)
        for _ in range(k_models)
    ]
    gnns = [
        GNNClass(
            in_channels=latent_dim,
            hidden_channels=latent_dim,
            out_channels=1,
            layers_count=gnn_layers,
        ).to(device)
        for _ in range(k_models)
    ]
    return EnsembleWrapper(encoders, gnns).to(device)
