"""
align_replogle_data.py
----------------------
Preprocesses and aligns the Replogle (2022) large-scale Perturb-seq dataset
to the Norman HVG vocabulary used to train ATLAS.

Replogle obs columns relevant here:
  - 'gene'            : perturbed gene name (category); controls == 'non-targeting'
  - 'perturbation'    : 'gene' | 'non-targeting' | 'targeting' (broader type)
  - 'perturbation_type': 'gene' | 'control'

Output tensors (saved to data/processed/replogle/):
  replogle_aligned_expression.pt   – [N_cells, len(Norman_genes)]
  replogle_aligned_interventions.pt – [N_cells, 2]  (gene index or -1)
  replogle_aligned_targets.pt      – [N_cells, len(Norman_genes)]
  replogle_valid_gene_mask.pt      – [len(Norman_genes)]  bool
"""

import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def align_replogle_to_norman(raw_matrix, replogle_genes, norman_genes):
    """
    Aligns Replogle expression to the Norman HVG gene order.
    Missing genes are zero-padded (same semantic as align_dixit_data.py).
    """
    print(f"Aligning {len(replogle_genes)} Replogle genes -> {len(norman_genes)} Norman HVGs …")
    num_cells = raw_matrix.shape[0]
    num_norman = len(norman_genes)

    aligned = torch.zeros((num_cells, num_norman), dtype=torch.float32)
    valid_mask = torch.zeros(num_norman, dtype=torch.bool)

    rep_gene_to_idx = {g: i for i, g in enumerate(replogle_genes)}
    common = 0
    for nor_idx, nor_gene in enumerate(norman_genes):
        if nor_gene in rep_gene_to_idx:
            rep_idx = rep_gene_to_idx[nor_gene]
            col = raw_matrix[:, rep_idx]
            if torch.is_tensor(col):
                aligned[:, nor_idx] = col
            else:
                aligned[:, nor_idx] = torch.tensor(col, dtype=torch.float32)
            valid_mask[nor_idx] = True
            common += 1

    missing = num_norman - common
    print(f"  [OK] {common} overlapping genes | {missing} genes zero-padded")
    return aligned, valid_mask


def process_replogle(replogle_h5ad_path, norman_genes_path, output_dir, max_cells=50_000):
    """
    Full preprocessing pipeline:
    1. Read h5ad
    2. Normalise + log1p
    3. Align to Norman HVG space
    4. Build intervention tensor using the 'gene' obs column
    5. Save aligned tensors
    """
    import scanpy as sc

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading Norman genes from {norman_genes_path} …")
    norman_genes = torch.load(norman_genes_path, weights_only=False)
    norman_gene_to_idx = {g: i for i, g in enumerate(norman_genes)}

    print(f"Loading Replogle h5ad from {replogle_h5ad_path} …")
    adata = sc.read_h5ad(replogle_h5ad_path)
    print(f"  Raw shape: {adata.shape}")

    # ── Preprocessing ──────────────────────────────────────────────────────────
    # Normalise to 10k counts per cell then log1p (same pipeline as Norman)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print("  [OK] normalize_total + log1p done")

    # subsample to keep memory manageable for evaluation
    if adata.n_obs > max_cells:
        np.random.seed(42)
        idx = np.random.choice(adata.n_obs, max_cells, replace=False)
        adata = adata[idx, :]
        print(f"  Subsampled to {adata.n_obs} cells")

    # ── Gene alignment ─────────────────────────────────────────────────────────
    replogle_genes = adata.var_names.tolist()
    X_dense = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
    aligned_expr, valid_mask = align_replogle_to_norman(X_dense, replogle_genes, norman_genes)

    # ── Intervention tensors ───────────────────────────────────────────────────
    print("Parsing interventions from 'gene' obs column …")
    # Controls: adata.obs['gene'] == 'non-targeting'  (or similar strings)
    pert_col = adata.obs["gene"].values.astype(str)
    control_labels = {"non-targeting", "safe-targeting", "control", "nan"}

    interventions = []
    skipped_ood = 0
    for pert in pert_col:
        if pert in control_labels or pert == "nan":
            interventions.append([-1, -1])
        else:
            # Replogle is single-gene (no doubles), gene string is the gene symbol
            gene = pert.strip()
            if gene in norman_gene_to_idx:
                interventions.append([norman_gene_to_idx[gene], -1])
            else:
                # Gene not in Norman vocab → treat as unknown control
                skipped_ood += 1
                interventions.append([-1, -1])

    print(f"  [OK] Interventions parsed | {skipped_ood} cells with vocab-OOD targets (treated as control)")

    interventions_t = torch.tensor(interventions, dtype=torch.long)
    target_expr = aligned_expr.clone()

    # Build basal state from control cells
    control_idx = [i for i, p in enumerate(pert_col) if p in control_labels]
    if control_idx:
        basal_state = aligned_expr[control_idx].mean(dim=0)
        print(f"  Basal state computed from {len(control_idx)} control cells")
    else:
        basal_state = aligned_expr.mean(dim=0)
        print("  WARNING: no control cells found; using global mean as basal state")

    # Input X = basal state repeated for every cell (same as Dixit)
    replogle_x = basal_state.unsqueeze(0).repeat(aligned_expr.size(0), 1)

    # ── Save ───────────────────────────────────────────────────────────────────
    print(f"Saving to {output_dir} …")
    torch.save(replogle_x,           os.path.join(output_dir, "replogle_aligned_expression.pt"))
    torch.save(interventions_t,      os.path.join(output_dir, "replogle_aligned_interventions.pt"))
    torch.save(target_expr,          os.path.join(output_dir, "replogle_aligned_targets.pt"))
    torch.save(valid_mask,           os.path.join(output_dir, "replogle_valid_gene_mask.pt"))

    n_inter = (interventions_t[:, 0] != -1).sum().item()
    print(f"[DONE] Replogle alignment complete | {adata.n_obs} cells | {valid_mask.sum().item()} overlap genes | {n_inter} cells with known interventions")


if __name__ == "__main__":
    process_replogle(
        replogle_h5ad_path="data/external/repogle.h5ad",
        norman_genes_path="data/processed/gene_names.pt",
        output_dir="data/processed/replogle/",
        max_cells=50_000,
    )
