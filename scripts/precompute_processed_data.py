import os
import torch
import scanpy as sc
import numpy as np

def precompute_data(h5ad_path="data/external/norman_2019_adata.h5ad", output_dir="data/processed", max_interventions=2):
    """
    Loads h5ad once, extracts required tensors, and saves them for fast loading.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)
    
    num_samples = adata.n_obs
    num_genes = adata.n_vars
    
    gene_names = adata.var['gene_name'].values.astype(str)
    gene_to_idx = {name: idx for idx, name in enumerate(gene_names)}
    
    # Save gene names
    torch.save(gene_names, os.path.join(output_dir, "gene_names.pt"))
    
    print("Parsing interventions...")
    interventions = []
    guide_identities = adata.obs['guide_identity'].values.astype(str)
    
    control_indices = []
    
    for i, guide in enumerate(guide_identities):
        first_part = guide.split('__')[0]
        genes = first_part.split('_')
        
        indices = [-1] * max_interventions
        idx_count = 0
        is_control = True
        
        for g in genes:
            if 'NegCtrl' not in g:
                is_control = False
                if g in gene_to_idx and idx_count < max_interventions:
                    indices[idx_count] = gene_to_idx[g]
                    idx_count += 1
                    
        if is_control:
            control_indices.append(i)
            
        interventions.append(indices)
        
    interventions_tensor = torch.tensor(interventions, dtype=torch.long)
    torch.save(interventions_tensor, os.path.join(output_dir, "interventions.pt"))
    
    print(f"Dataset has {len(control_indices)} control cells out of {num_samples} total.")
    
    # Compute basal state
    if len(control_indices) > 0:
        control_X = adata.X[control_indices]
        basal_state = torch.tensor(np.array(control_X.mean(axis=0)).flatten(), dtype=torch.float32)
    else:
        print("Warning: No control cells found. Setting basal state to zeros.")
        basal_state = torch.zeros(num_genes, dtype=torch.float32)
        
    torch.save(basal_state, os.path.join(output_dir, "basal_state.pt"))
    
    print("Extracting full expression matrix...")
    # Convert sparse to dense incrementally or save via sparse tensor if memory limited.
    # Norman is 27k x 3k, ~320MB dense float32. Fits in memory easily.
    X_dense = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    X_tensor = torch.tensor(X_dense, dtype=torch.float32)
    torch.save(X_tensor, os.path.join(output_dir, "expression_matrix.pt"))
    
    print(f"Precomputation complete! Tensors saved to {output_dir}/")

if __name__ == "__main__":
    precompute_data()
