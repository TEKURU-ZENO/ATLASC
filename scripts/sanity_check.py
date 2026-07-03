"""
sanity_check.py

Lightweight sanity check script for the ATLAS project.

Verifies:
1. All required processed data files exist and are valid tensors.
2. Gene names and expression matrix dimensions are consistent.
3. Basal state shape matches gene count.
4. Intervention indices are within valid range.
5. Trained model checkpoints exist and can be loaded.
6. A single forward pass through the full pipeline produces finite outputs.

Run this after preprocessing and after training to confirm environment integrity.
"""

import os
import sys
import torch
import numpy as np
from typing import Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


REQUIRED_DATA_FILES = [
    "data/processed/expression_matrix.pt",
    "data/processed/gene_names.pt",
    "data/processed/basal_state.pt",
    "data/processed/interventions.pt",
    "data/processed/trrust_edge_index.pt",
]

REQUIRED_CHECKPOINT_FILES = [
    "models/checkpoints/epistemic_member_k0.pt",
    "models/checkpoints/epistemic_member_k1.pt",
    "models/checkpoints/epistemic_member_k2.pt",
]

OK = "   [OK]"
WARN = " [WARN]"
FAIL = " [FAIL]"


def check_file_exists(path: str) -> bool:
    exists = os.path.exists(path)
    status = OK if exists else FAIL
    print(f"{status} {path}")
    return exists


def check_tensor_shape(path: str, expected_dims: Optional[int] = None) -> bool:
    try:
        is_list = path.endswith("gene_names.pt")
        t = torch.load(path, weights_only=not is_list)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        shape_ok = (expected_dims is None or t.dim() == expected_dims)
        status = OK if shape_ok else WARN
        print(f"{status} {os.path.basename(path)}: shape={tuple(t.shape)}, dtype={t.dtype}")
        return shape_ok
    except Exception as e:
        print(f"{FAIL} {os.path.basename(path)}: {e}")
        return False


def check_dimension_consistency(
    expression_path: str,
    gene_names_path: str,
    basal_path: str,
    interventions_path: str,
) -> bool:
    try:
        expr   = torch.load(expression_path, weights_only=True)
        genes  = torch.load(gene_names_path, weights_only=False)
        basal  = torch.load(basal_path, weights_only=True)
        ints   = torch.load(interventions_path, weights_only=True)

        num_cells, num_genes_expr = expr.shape
        num_genes_vocab = len(genes)
        basal_genes = basal.shape[0]
        num_cells_int = ints.shape[0]

        ok = all([
            num_genes_expr == num_genes_vocab,
            num_genes_vocab == basal_genes,
            num_cells == num_cells_int,
        ])
        status = OK if ok else FAIL
        print(f"{status} Dimension consistency: {num_cells} cells x {num_genes_vocab} genes.")
        if not ok:
            print(f"       expr={expr.shape}, vocab={num_genes_vocab}, basal={basal.shape}, ints={ints.shape}")
        return ok
    except Exception as e:
        print(f"{FAIL} Dimension check failed: {e}")
        return False


def check_intervention_range(interventions_path: str, num_genes: int) -> bool:
    try:
        ints = torch.load(interventions_path, weights_only=True)
        valid = ints[(ints != -1)]
        in_range = (valid >= 0).all() and (valid < num_genes).all()
        status = OK if in_range else FAIL
        max_idx = valid.max().item() if valid.numel() > 0 else -1
        print(f"{status} Intervention indices: valid range [0, {num_genes-1}], max_seen={max_idx}")
        return bool(in_range)
    except Exception as e:
        print(f"{FAIL} Intervention range check: {e}")
        return False


def check_forward_pass() -> bool:
    """Run one mini forward pass through the full ATLAS pipeline."""
    try:
        from models.encoders.transcriptomic_encoder import TranscriptomicEncoder
        from models.gnn.causal_gnn import CausalGNN
        from models.ensemble import EnsembleWrapper
        from causal.do_operator import apply_intervention
        from utils.data_loader import NormanDataset, get_trrust_graph
        from simulation.rollout_engine import RolloutEngine

        device = torch.device('cpu')
        dataset = NormanDataset(processed_dir="data/processed")
        num_genes = dataset.num_genes
        edge_index = get_trrust_graph("data/processed").to(device)

        encoders, gnns = [], []
        for k in range(3):
            enc = TranscriptomicEncoder(num_genes=num_genes, latent_dim=16).to(device)
            gnn = CausalGNN(in_channels=16, hidden_channels=16, out_channels=1, layers_count=3).to(device)
            ckpt = torch.load(f"models/checkpoints/epistemic_member_k{k}.pt",
                              map_location=device, weights_only=True)
            enc.load_state_dict(ckpt['encoder_state_dict'])
            gnn.load_state_dict(ckpt['gnn_state_dict'])
            enc.eval(); gnn.eval()
            encoders.append(enc); gnns.append(gnn)

        ensemble = EnsembleWrapper(encoders, gnns)
        simulator = RolloutEngine(core_model=ensemble, execution_mode="deterministic")

        x, ints, target = dataset[0]
        x = x.unsqueeze(0).to(device)
        ints = ints.unsqueeze(0).to(device)

        with torch.no_grad():
            z0_k = ensemble.encode(x)
            z_int_k = torch.stack([
                apply_intervention(z0_k[k], ints, value=0.0)
                for k in range(ensemble.k_models)
            ], dim=0)
            latent_k, trajectory = simulator.rollout(
                z_int_k, edge_index, steps=3, log_trajectory=True, enforce_stability=False
            )
            preds = ensemble.decode(latent_k)

        is_finite = preds.isfinite().all().item()
        status = OK if is_finite else FAIL
        print(f"{status} Forward pass: output shape={tuple(preds.shape)}, finite={is_finite}")
        return is_finite

    except Exception as e:
        print(f"{FAIL} Forward pass: {e}")
        return False


def run_sanity_check():
    print("=" * 60)
    print("ATLAS Sanity Check")
    print("=" * 60)

    all_ok = True

    print("\n[1] Required data files:")
    for f in REQUIRED_DATA_FILES:
        if not check_file_exists(f):
            all_ok = False

    print("\n[2] Tensor shapes:")
    for f in REQUIRED_DATA_FILES:
        if os.path.exists(f):
            check_tensor_shape(f)

    print("\n[3] Dimension consistency:")
    first_four = [f for i, f in enumerate(REQUIRED_DATA_FILES) if i < 4]
    if all(os.path.exists(f) for f in first_four):
        if not check_dimension_consistency(
            "data/processed/expression_matrix.pt",
            "data/processed/gene_names.pt",
            "data/processed/basal_state.pt",
            "data/processed/interventions.pt",
        ):
            all_ok = False

        genes = torch.load("data/processed/gene_names.pt", weights_only=False)
        if not check_intervention_range("data/processed/interventions.pt", len(genes)):
            all_ok = False

    print("\n[4] Model checkpoints:")
    for ckpt in REQUIRED_CHECKPOINT_FILES:
        if not check_file_exists(ckpt):
            all_ok = False

    print("\n[5] Forward pass:")
    if all(os.path.exists(c) for c in REQUIRED_CHECKPOINT_FILES):
        if not check_forward_pass():
            all_ok = False
    else:
        print(f"{WARN} Skipping forward pass (checkpoints missing).")

    print("\n" + "=" * 60)
    if all_ok:
        print("RESULT: All checks passed. ATLAS environment is healthy.")
    else:
        print("RESULT: Some checks failed. Review errors above before training.")
    print("=" * 60)


if __name__ == "__main__":
    run_sanity_check()
