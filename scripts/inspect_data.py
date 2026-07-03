import scanpy as sc

def inspect():
    path = "data/external/norman_2019_adata.h5ad"
    print(f"Loading {path}...")
    adata = sc.read_h5ad(path)
    
    print("\n--- AnnData Info ---")
    print(adata)
    
    print("\n--- OBS (Cells / Interventions) HEAD ---")
    print(adata.obs.head())
    
    print("\n--- VAR (Genes) HEAD ---")
    print(adata.var.head())
    
    print("\n--- Available OBS Columns ---")
    print(adata.obs.columns.tolist())
    
    if 'condition' in adata.obs.columns:
        print("\n--- Unique Conditions (Sample) ---")
        print(adata.obs['condition'].unique()[:10])
        print(f"Total Unique Conditions: {adata.obs['condition'].nunique()}")
        
    if 'guide_identity' in adata.obs.columns:
        print("\n--- Unique Guide Identities (Sample) ---")
        print(adata.obs['guide_identity'].unique()[:10])

if __name__ == "__main__":
    inspect()
