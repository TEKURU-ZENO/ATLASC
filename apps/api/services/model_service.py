"""
model_service.py
----------------
Singleton service that loads and caches the trained ATLAS ensemble.
Called once at startup; subsequent requests reuse the cached models.
"""

from __future__ import annotations

import os
import torch
from typing import Optional

from models.encoders.transcriptomic_encoder import TranscriptomicEncoder
from models.gnn.causal_gnn import CausalGNN
from models.ensemble import EnsembleWrapper

# Project root is two levels up from this file (apps/api/services/model_service.py)
_HERE = os.path.abspath(os.path.dirname(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "../../.."))

# ---------------------------------------------------------------------------
# Internal singleton state
# ---------------------------------------------------------------------------

_LOADED: dict = {}


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_ensemble(
    checkpoint_dir: str = None,
    k_models: int = 3,
    latent_dim: int = 16,
    gnn_layers: int = 3,
    num_genes: Optional[int] = None,
) -> EnsembleWrapper:
    """
    Return the cached ensemble, building and loading it on first call.

    Args:
        checkpoint_dir : directory containing epistemic_member_k{k}.pt files.
                         Defaults to <project_root>/models/checkpoints.
        k_models       : number of ensemble members
        latent_dim     : latent dimension used during training
        gnn_layers     : number of GNN message-passing layers
        num_genes      : gene vocab size; inferred from gene_names.pt if None

    Returns:
        EnsembleWrapper in eval mode on the appropriate device.
    """
    global _LOADED
    if _LOADED:
        return _LOADED["ensemble"]

    # Resolve checkpoint dir relative to project root (not CWD)
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(_PROJECT_ROOT, "models", "checkpoints")

    device = _default_device()
    print(f"[model_service] Loading ATLAS ensemble on {device} from {checkpoint_dir} ...")

    # Infer num_genes from gene_names.pt (always absolute)
    if num_genes is None:
        names_path = os.path.join(_PROJECT_ROOT, "data", "processed", "gene_names.pt")
        if os.path.exists(names_path):
            names = torch.load(names_path, map_location="cpu", weights_only=False)
            num_genes = len(names)
        else:
            num_genes = 18_080  # Norman dataset default

    encoders, gnns = [], []
    for k in range(k_models):
        ckpt_path = os.path.join(checkpoint_dir, f"epistemic_member_k{k}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[model_service] WARNING: checkpoint missing – {ckpt_path}")
            continue

        enc = TranscriptomicEncoder(num_genes=num_genes, latent_dim=latent_dim).to(device)
        gnn = CausalGNN(
            in_channels=latent_dim,
            hidden_channels=latent_dim,
            out_channels=1,
            layers_count=gnn_layers,
        ).to(device)

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        enc.load_state_dict(ckpt["encoder_state_dict"])
        gnn.load_state_dict(ckpt["gnn_state_dict"])
        enc.eval(); gnn.eval()

        encoders.append(enc); gnns.append(gnn)

    if not encoders:
        raise RuntimeError(
            "[model_service] No checkpoints found. Train the model first."
        )

    ensemble = EnsembleWrapper(encoders, gnns).to(device)
    ensemble.eval()

    _LOADED["ensemble"] = ensemble
    _LOADED["device"]   = device
    _LOADED["num_genes"] = num_genes
    print(f"[model_service] Loaded {len(encoders)}-member ensemble ({num_genes} genes).")
    return ensemble


def get_device() -> torch.device:
    """Return the device the ensemble is loaded on (triggers load if needed)."""
    get_ensemble()
    return _LOADED.get("device", _default_device())


def get_num_genes() -> int:
    """Return num_genes inferred during model loading."""
    get_ensemble()
    return _LOADED.get("num_genes", 18_080)


def unload() -> None:
    """Clear the cache (useful for testing)."""
    global _LOADED
    _LOADED = {}
