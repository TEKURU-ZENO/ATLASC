import os
import urllib.request
import pandas as pd
import torch

def download_and_process_trrust(processed_dir="data/processed"):
    os.makedirs(processed_dir, exist_ok=True)
    
    trrust_url = "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv"
    tsv_path = os.path.join(processed_dir, "trrust_human.tsv")
    out_graph_path = os.path.join(processed_dir, "trrust_edge_index.pt")
    
    # 1. Download
    print(f"Downloading TRRUST database from {trrust_url}...")
    urllib.request.urlretrieve(trrust_url, tsv_path)
    print("Download complete.")
    
    # 2. Load Norman Genes
    gene_names_path = os.path.join(processed_dir, "gene_names.pt")
    if not os.path.exists(gene_names_path):
        raise FileNotFoundError(f"Missing {gene_names_path}. Cannot align TRRUST edges.")
        
    norman_genes = torch.load(gene_names_path, weights_only=False)
    # Create lookup map (Gene Name -> Int Index in Norman dataset)
    # TRRUST usually has uppercase gene symbols
    # Norman might have mixed. Uppercase everything for strict intersection.
    gene_to_idx = {name.upper(): idx for idx, name in enumerate(norman_genes)}
    
    # 3. Parse TRRUST (TSV columns: TF, Target, Mode of Regulation, PubMed ID)
    edges_src = []
    edges_dst = []
    
    df = pd.read_csv(tsv_path, sep='\t', header=None, names=['TF', 'Target', 'Mode', 'PMID'])
    
    mapped_edges = 0
    for _, row in df.iterrows():
        tf = str(row['TF']).upper()
        target = str(row['Target']).upper()
        
        # Only add edge if BOTH genes exist in our 3214 Norman dimension
        if tf in gene_to_idx and target in gene_to_idx:
            edges_src.append(gene_to_idx[tf])
            edges_dst.append(gene_to_idx[target])
            mapped_edges += 1
            
    print(f"Mapped {mapped_edges} true biological directed edges out of {len(df)} total TRRUST interactions.")
    
    # 4. Construct Tensor & Adding Self-Loops for numerical stability in GNN
    # A gene always biologically depends on its previous state.
    for i in range(len(norman_genes)):
        edges_src.append(i)
        edges_dst.append(i)
        
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    
    # 5. Save
    torch.save(edge_index, out_graph_path)
    print(f"Saved true directed GRN to {out_graph_path}. Shape: {edge_index.shape}")
    
    # Clean up TSV
    os.remove(tsv_path)

if __name__ == "__main__":
    download_and_process_trrust()
