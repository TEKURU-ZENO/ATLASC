"""
gene_regulatory_graph.py
------------------------
Biological Gene Regulatory Network (GRN) graph builder with signed edge attributes.
Supports interaction directionality, mode of regulation (Activation vs Repression),
and conversion to PyTorch Geometric edge tensors.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Set, Tuple
import networkx as nx
import torch


# Mode of regulation mappings
REGULATION_SIGNS = {
    "ACTIVATION": 1.0,
    "ACTIVATOR": 1.0,
    "UPREGULATES": 1.0,
    "POSITIVE": 1.0,
    "REPRESSION": -1.0,
    "REPRESSOR": -1.0,
    "DOWNREGULATES": -1.0,
    "NEGATIVE": -1.0,
    "UNKNOWN": 0.5,
}


class GeneRegulatoryGraph:
    """
    Biological Directed GRN with signed edge weights for causal intervention propagation.
    """
    def __init__(self, gene_names: List[str]):
        self.gene_names = [g.upper() for g in gene_names]
        self.gene_to_idx = {g: idx for idx, g in enumerate(self.gene_names)}
        self.num_genes = len(self.gene_names)
        self.graph = nx.DiGraph()
        for idx, g in enumerate(self.gene_names):
            self.graph.add_node(idx, gene_name=g)

    def add_interaction(
        self,
        source_gene: str,
        target_gene: str,
        mode: str = "Activation",
        weight: Optional[float] = None,
    ) -> bool:
        """
        Add a directed regulatory interaction between two genes.
        """
        src_clean = source_gene.strip().upper()
        tgt_clean = target_gene.strip().upper()

        if src_clean in self.gene_to_idx and tgt_clean in self.gene_to_idx:
            src_idx = self.gene_to_idx[src_clean]
            tgt_idx = self.gene_to_idx[tgt_clean]
            if weight is None:
                mode_clean = mode.strip().upper()
                sign = REGULATION_SIGNS.get(mode_clean, 0.5)
            else:
                sign = weight
            self.graph.add_edge(src_idx, tgt_idx, mode=mode, sign=sign)
            return True
        return False

    def load_trrust_tsv(self, tsv_path: str) -> int:
        """
        Loads interactions from TRRUST TSV (TF, Target, Mode, PMID).
        """
        if not os.path.exists(tsv_path):
            raise FileNotFoundError(f"Missing {tsv_path}")

        count = 0
        with open(tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    tf, target, mode = parts[0], parts[1], parts[2]
                    if self.add_interaction(tf, target, mode=mode):
                        count += 1
        return count

    def to_pytorch(self, add_self_loops: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Exports graph as (edge_index, edge_weight) PyTorch long and float tensors.
        """
        src_list, dst_list, weight_list = [], [], []

        for u, v, data in self.graph.edges(data=True):
            src_list.append(u)
            dst_list.append(v)
            weight_list.append(data.get("sign", 1.0))

        if add_self_loops:
            for i in range(self.num_genes):
                src_list.append(i)
                dst_list.append(i)
                weight_list.append(1.0)  # Self-loop retention

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_weight = torch.tensor(weight_list, dtype=torch.float32)
        return edge_index, edge_weight

    def get_centrality_metrics(self) -> Dict[str, Dict[int, float]]:
        """Returns in-degree, out-degree, and betweenness centrality."""
        in_deg = dict(self.graph.in_degree())
        out_deg = dict(self.graph.out_degree())
        between = nx.betweenness_centrality(self.graph)
        return {"in_degree": in_deg, "out_degree": out_deg, "betweenness": between}


def build_gene_graph(gene_list: List[str], interaction_db: List[Tuple[str, str, str]]) -> GeneRegulatoryGraph:
    """Convenience constructor from list of (TF, Target, Mode) tuples."""
    grn = GeneRegulatoryGraph(gene_list)
    for row in interaction_db:
        if len(row) == 2:
            grn.add_interaction(row[0], row[1], mode="Activation")
        else:
            grn.add_interaction(row[0], row[1], mode=row[2])
    return grn
