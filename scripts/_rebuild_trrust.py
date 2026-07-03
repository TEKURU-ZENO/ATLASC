"""
Rebuild TRRUST edge index from local TSV file using corrected gene_names.pt
"""
import os
import pandas as pd
import torch

processed_dir = "data/processed"
tsv_path = "data/external/trrust_rawdata.human.tsv"
out_graph_path = os.path.join(processed_dir, "trrust_edge_index.pt")

# Load corrected gene names
norman_genes = torch.load(os.path.join(processed_dir, "gene_names.pt"), weights_only=False)
gene_to_idx = {name.upper(): idx for idx, name in enumerate(norman_genes)}
print(f"Loaded {len(norman_genes)} genes (sample: {norman_genes[:5]})")

# Parse TRRUST
df = pd.read_csv(tsv_path, sep='\t', header=None, names=['TF', 'Target', 'Mode', 'PMID'])
print(f"TRRUST total interactions: {len(df)}")

edges_src = []
edges_dst = []
mapped = 0
for _, row in df.iterrows():
    tf = str(row['TF']).upper()
    target = str(row['Target']).upper()
    if tf in gene_to_idx and target in gene_to_idx:
        edges_src.append(gene_to_idx[tf])
        edges_dst.append(gene_to_idx[target])
        mapped += 1

print(f"Mapped {mapped} biological directed edges")

# Add self-loops
for i in range(len(norman_genes)):
    edges_src.append(i)
    edges_dst.append(i)

edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
torch.save(edge_index, out_graph_path)
print(f"Saved TRRUST graph: {edge_index.shape} (edges: {mapped}, self-loops: {len(norman_genes)})")
