import scanpy as sc
import json

# Norman
adata = sc.read_h5ad('data/external/norman_2019_adata.h5ad')

real_gene_names = adata.var['gene_name'].tolist()
gene_set = set(real_gene_names)

# Extract perturbation genes from guide_identity
pert_genes = set()
gi_vals = adata.obs['guide_identity'].unique().tolist()
for gi in gi_vals:
    first_half = gi.split('__')[0]
    parts = first_half.split('_')
    for p in parts:
        if not p.startswith('NegCtrl'):
            pert_genes.add(p)

overlap = pert_genes & gene_set
missing = pert_genes - gene_set

result = {
    'norman_shape': [adata.n_obs, adata.n_vars],
    'num_real_gene_names': len(gene_set),
    'real_gene_names_sample': real_gene_names[:10],
    'num_pert_genes': len(pert_genes),
    'pert_genes': sorted(list(pert_genes)),
    'pert_in_vocab': len(overlap),
    'pert_overlap': sorted(list(overlap)),
    'pert_missing': sorted(list(missing)),
}

with open('scripts/_norman_info.json', 'w') as f:
    json.dump(result, f, indent=2)

print("Done. Written to scripts/_norman_info.json")
