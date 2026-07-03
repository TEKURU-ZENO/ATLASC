import scanpy as sc
import json

adata = sc.read_h5ad('data/external/Dixit_2016.h5ad')

result = {
    'shape': [adata.n_obs, adata.n_vars],
    'obs_columns': list(adata.obs.columns),
    'var_columns': list(adata.var.columns),
    'var_names_sample': adata.var_names[:5].tolist(),
}

# Check for gene_name column
if 'gene_name' in adata.var.columns:
    result['var_gene_name_sample'] = adata.var['gene_name'][:5].tolist()

# Check perturbation column
if 'perturbation_name' in adata.obs.columns:
    vals = adata.obs['perturbation_name'].astype(str).unique().tolist()
    result['perturbation_name_unique_count'] = len(vals)
    result['perturbation_name_samples'] = vals[:15]

if 'guide_identity' in adata.obs.columns:
    vals = adata.obs['guide_identity'].astype(str).unique().tolist()
    result['guide_identity_unique_count'] = len(vals)
    result['guide_identity_samples'] = vals[:15]

with open('scripts/_dixit_info.json', 'w') as f:
    json.dump(result, f, indent=2, default=str)

print('Done. Output in scripts/_dixit_info.json')
