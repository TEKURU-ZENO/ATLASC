"""
preprocess_data.py

Preprocesses the raw Norman 2019 Perturb-seq dataset for ATLAS training.

Pipeline:
1. Load raw .h5ad file.
2. Filter cells (min counts, max counts, non-control).
3. Normalize total counts per cell (sc.pp.normalize_total).
4. Log1p transform.
5. Select highly variable genes (HVGs) using scanpy.
6. Save processed expression matrix, gene names, and basal_state.

Expected input:  data/raw/Norman_2019.h5ad
Expected output: data/processed/{expression_matrix.pt, gene_names.pt, basal_state.pt, interventions.pt}
"""

import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def preprocess_norman(
    raw_h5ad_path: str = "data/external/norman_2019_adata.h5ad",
    output_dir: str = "data/processed",
    n_hvgs: int = 2000,
    min_counts: int = 200,
    max_counts: int = 25000,
) -> None:
    """
    Full preprocessing pipeline for the Norman 2019 dataset.

    Args:
        raw_h5ad_path: Path to the raw .h5ad file.
        output_dir: Directory to write processed tensors.
        n_hvgs: Number of highly variable genes to select.
        min_counts: Min total count filter per cell.
        max_counts: Max total count filter per cell.
    """
    try:
        import scanpy as sc
    except ImportError:
        print("ERROR: scanpy not installed. Run: pip install scanpy")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading raw dataset: {raw_h5ad_path}")
    adata = sc.read_h5ad(raw_h5ad_path)
    print(f"Loaded: {adata.n_obs} cells x {adata.n_vars} genes")

    # Basic quality filtering
    print("Filtering cells by count...")
    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.filter_cells(adata, max_counts=max_counts)
    print(f"After filtering: {adata.n_obs} cells")

    # Normalize
    print("Normalizing...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # HVG selection
    print(f"Selecting top {n_hvgs} highly variable genes...")
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvgs, flavor='seurat')
    adata = adata[:, adata.var.highly_variable]
    print(f"HVG subset: {adata.n_obs} cells x {adata.n_vars} genes")

    # Extract dense matrix
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Use real gene symbols from var['gene_name'], NOT numeric var_names
    if 'gene_name' in adata.var.columns:
        gene_names = adata.var['gene_name'].tolist()
        print(f"Using adata.var['gene_name'] for gene symbols (sample: {gene_names[:5]})")
    else:
        gene_names = adata.var_names.tolist()
        print(f"WARNING: 'gene_name' column not found, falling back to var_names")

    # Basal state: mean of control cells
    # Norman 2019 uses 'guide_identity' column; controls have NegCtrl in both slots
    if 'guide_identity' in adata.obs.columns:
        gi_values = adata.obs['guide_identity'].values.astype(str)
        ctrl_mask = np.array([
            all(p.startswith('NegCtrl') for p in gi.split('__')[0].split('_'))
            for gi in gi_values
        ])
        print(f"Control detection via guide_identity (NegCtrl pattern): {np.sum(ctrl_mask)} control cells")
    elif 'perturbation_name' in adata.obs.columns:
        ctrl_mask = adata.obs['perturbation_name'].values == 'control'
    else:
        ctrl_mask = np.ones(adata.n_obs, dtype=bool)
        print("WARNING: no control label found — using global mean as basal_state.")

    ctrl_X = X_tensor[torch.tensor(ctrl_mask)]
    basal_state = ctrl_X.mean(dim=0)

    # Parse interventions from guide_identity metadata
    # Format: GENE1_GENE2__GENE1_GENE2 (double underscore separates repeated pair)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    max_interventions = 2
    interventions = []
    mapped_count = 0
    unmapped_genes = set()

    if 'guide_identity' in adata.obs.columns:
        gi_values = adata.obs['guide_identity'].values.astype(str)
        for i in range(adata.n_obs):
            indices = [-1] * max_interventions
            gi = gi_values[i]
            # Take first half before __ (the two guide gene names)
            first_half = gi.split('__')[0]
            parts = first_half.split('_')
            slot = 0
            for p in parts:
                if p.startswith('NegCtrl'):
                    continue
                if slot < max_interventions:
                    if p in gene_to_idx:
                        indices[slot] = gene_to_idx[p]
                        slot += 1
                        mapped_count += 1
                    else:
                        unmapped_genes.add(p)
            interventions.append(indices)
    else:
        # Fallback: try perturbation_name with + separator
        pert_col = 'perturbation_name' if 'perturbation_name' in adata.obs.columns else None
        for i in range(adata.n_obs):
            indices = [-1] * max_interventions
            if pert_col:
                pert = str(adata.obs[pert_col].iloc[i])
                if pert != 'control' and pert != 'nan':
                    genes = pert.split('+')
                    for slot, g in enumerate(genes):
                        g = g.strip()
                        if slot < max_interventions and g in gene_to_idx:
                            indices[slot] = int(gene_to_idx[g])
            interventions.append(indices)

    print(f"Intervention mapping: {mapped_count} gene slots mapped across {adata.n_obs} cells")
    if unmapped_genes:
        print(f"  {len(unmapped_genes)} perturbation genes not in HVG vocab: {sorted(unmapped_genes)}")

    interventions_tensor = torch.tensor(interventions, dtype=torch.long)

    # Save
    torch.save(X_tensor,            os.path.join(output_dir, "expression_matrix.pt"))
    torch.save(gene_names,          os.path.join(output_dir, "gene_names.pt"))
    torch.save(basal_state,         os.path.join(output_dir, "basal_state.pt"))
    torch.save(interventions_tensor, os.path.join(output_dir, "interventions.pt"))

    print(f"\nSaved preprocessed tensors to {output_dir}/")
    print(f"  expression_matrix.pt: {X_tensor.shape}")
    print(f"  gene_names.pt:        {len(gene_names)} genes (sample: {gene_names[:3]})")
    print(f"  basal_state.pt:       {basal_state.shape}")
    print(f"  interventions.pt:     {interventions_tensor.shape}")
    ctrl_count = int(np.sum(ctrl_mask))
    valid_interventions = (interventions_tensor != -1).any(dim=-1).sum().item()
    print(f"  Control cells:        {ctrl_count}")
    print(f"  Cells with valid interventions: {valid_interventions}/{adata.n_obs}")


if __name__ == "__main__":
    preprocess_norman()
