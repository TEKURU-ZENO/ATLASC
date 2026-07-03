"""
build_graphs.py

Constructs and saves biological gene regulatory network (GRN) graph structures
from processed TRRUST data or other sources.

Outputs:
- data/processed/trrust_edge_index.pt: COO edge index [2, E]
- data/processed/trrust_edge_weights.pt (optional, future)
"""

import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_loader import get_trrust_graph


def verify_graph_integrity(edge_index: torch.Tensor, num_genes: int) -> dict:
    """
    Verify the loaded GRN graph has valid structure.

    Returns:
        dict with 'num_edges', 'max_node', 'is_valid'.
    """
    if edge_index.size(0) != 2:
        return {"is_valid": False, "error": "edge_index must be [2, E]"}

    max_node = int(edge_index.max().item()) if edge_index.numel() > 0 else 0
    num_edges = edge_index.size(1)

    issues = []
    if max_node >= num_genes:
        issues.append(f"max node index {max_node} >= num_genes {num_genes}")
    if num_edges == 0:
        issues.append("graph has zero edges")

    return {
        "num_edges": num_edges,
        "num_nodes_referenced": max_node + 1,
        "max_node_index": max_node,
        "is_valid": len(issues) == 0,
        "issues": issues,
    }


def summarize_graph(edge_index: torch.Tensor) -> None:
    """Print graph statistics."""
    num_edges = edge_index.size(1)
    unique_src = edge_index[0].unique().numel()
    unique_dst = edge_index[1].unique().numel()
    self_loops = (edge_index[0] == edge_index[1]).sum().item()

    print(f"\nGRN Graph Summary:")
    print(f"  Edges:           {num_edges}")
    print(f"  Unique sources:  {unique_src}")
    print(f"  Unique targets:  {unique_dst}")
    print(f"  Self-loops:      {self_loops}")


def build_and_save_graph(processed_dir: str = "data/processed") -> None:
    """
    Load or rebuild the TRRUST GRN edge_index and verify its integrity.
    """
    edge_index_path = os.path.join(processed_dir, "trrust_edge_index.pt")
    gene_names_path = os.path.join(processed_dir, "gene_names.pt")

    if not os.path.exists(gene_names_path):
        print(f"ERROR: gene_names.pt not found at {gene_names_path}")
        print("Run precompute_processed_data.py first.")
        return

    gene_names = torch.load(gene_names_path, weights_only=False)
    num_genes = len(gene_names)
    print(f"Loaded {num_genes} Norman HVGs.")

    print("Building TRRUST GRN edge_index...")
    edge_index = get_trrust_graph(processed_dir)

    # Verify integrity
    info = verify_graph_integrity(edge_index, num_genes)
    if not info["is_valid"]:
        print(f"WARNING: Graph integrity issues: {info['issues']}")
    else:
        print(f"Graph integrity: OK")

    summarize_graph(edge_index)

    # Save
    os.makedirs(processed_dir, exist_ok=True)
    torch.save(edge_index, edge_index_path)
    print(f"\nSaved: {edge_index_path}")


if __name__ == "__main__":
    build_and_save_graph()
