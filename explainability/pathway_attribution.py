"""
pathway_attribution.py

Maps gene-level attribution scores onto biological pathways and gene sets.
Aggregates ATLAS gene importance signals to pathway-level interpretations.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


# Minimal built-in pathway map (KEGG hallmark gene signatures, curated subset)
BUILTIN_PATHWAY_MAP: Dict[str, List[str]] = {
    "Cell Cycle": ["CDK2", "CDK4", "CDK6", "CCND1", "CCNE1", "E2F1", "RB1", "CDKN1A", "CDKN2A"],
    "Apoptosis": ["BCL2", "BAX", "CASP3", "CASP8", "CASP9", "TP53", "PUMA", "NOXA"],
    "MAPK Signaling": ["KRAS", "BRAF", "MAP2K1", "MAPK1", "MAPK3", "RAF1", "EGFR"],
    "PI3K-AKT": ["PIK3CA", "AKT1", "AKT2", "PTEN", "MTOR", "TSC1", "TSC2"],
    "Wnt Signaling": ["CTNNB1", "APC", "AXIN1", "FZD1", "LRP5", "DVL1", "TCF7L2"],
    "DNA Repair": ["BRCA1", "BRCA2", "ATM", "ATR", "CHEK1", "CHEK2", "RAD51", "MLH1"],
    "Transcription Factors": ["MYC", "TP53", "E2F1", "SP1", "JUN", "FOS", "STAT3", "NFKB1"],
    "Immune Response": ["TNF", "IL6", "IFNG", "CD8A", "PDCD1", "CD274", "IRF3", "IRF7"],
    "Metabolism": ["LDHA", "MDH2", "IDH1", "IDH2", "G6PD", "PFKM", "PKM", "HK2"],
    "Epigenetics": ["DNMT1", "DNMT3A", "EZH2", "KDM6A", "HDAC1", "HDAC2", "SIRT1"],
}


def compute_pathway_importance(
    gene_importance: torch.Tensor,
    gene_names: List[str],
    pathway_map: Optional[Dict[str, List[str]]] = None,
    aggregation: str = "mean",
) -> Dict[str, float]:
    """
    Aggregate gene-level attribution scores to pathway-level importance.

    Args:
        gene_importance: [num_genes] tensor of per-gene attribution scores.
        gene_names: List of gene name strings, aligned with gene_importance.
        pathway_map: Dict mapping pathway names to lists of member gene names.
                     If None, uses BUILTIN_PATHWAY_MAP.
        aggregation: How to aggregate within a pathway: 'mean', 'sum', 'max'.

    Returns:
        Dict mapping pathway name -> aggregated attribution score.
    """
    if pathway_map is None:
        pathway_map = BUILTIN_PATHWAY_MAP

    gene_to_attr = {gene: gene_importance[i].item()
                    for i, gene in enumerate(gene_names)}

    pathway_scores = {}
    for pathway, members in pathway_map.items():
        member_scores = [gene_to_attr[g] for g in members if g in gene_to_attr]
        if len(member_scores) == 0:
            pathway_scores[pathway] = 0.0
            continue
        arr = np.array(member_scores)
        if aggregation == "mean":
            pathway_scores[pathway] = float(arr.mean())
        elif aggregation == "sum":
            pathway_scores[pathway] = float(arr.sum())
        elif aggregation == "max":
            pathway_scores[pathway] = float(np.abs(arr).max())
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    return pathway_scores


def rank_pathways(
    pathway_scores: Dict[str, float],
    use_abs: bool = True,
) -> List[Tuple[str, float]]:
    """
    Sort pathway scores by magnitude.

    Args:
        pathway_scores: Dict of pathway -> score.
        use_abs: If True, rank by absolute value.

    Returns:
        Sorted list of (pathway_name, score) tuples.
    """
    if use_abs:
        return sorted(pathway_scores.items(), key=lambda x: abs(x[1]), reverse=True)
    return sorted(pathway_scores.items(), key=lambda x: x[1], reverse=True)


def report_pathway_attribution(
    gene_importance: torch.Tensor,
    gene_names: List[str],
    pathway_map: Optional[Dict[str, List[str]]] = None,
    top_k: int = 10,
) -> None:
    """
    Print a human-readable table of top biological pathways by attribution.

    Args:
        gene_importance: [num_genes] attribution scores.
        gene_names: Gene name list.
        pathway_map: Optional custom pathway map.
        top_k: Number of top pathways to display.
    """
    scores = compute_pathway_importance(gene_importance, gene_names, pathway_map)
    ranked = rank_pathways(scores)
    ranked_top = [item for i, item in enumerate(ranked) if i < top_k]

    print(f"\n{'Rank':<6} {'Pathway':<35} {'Attribution Score':>18}")
    print("-" * 62)
    for i, (pathway, score) in enumerate(ranked_top):
        print(f"{i+1:<6} {pathway:<35} {score:>18.5f}")


def get_pathway_gene_coverage(
    gene_names: List[str],
    pathway_map: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, dict]:
    """
    Report what fraction of each pathway's genes are present in our vocabulary.

    Args:
        gene_names: Gene vocabulary list.
        pathway_map: Optional custom pathway dict.

    Returns:
        Dict with pathway name -> {covered, total, fraction}.
    """
    if pathway_map is None:
        pathway_map = BUILTIN_PATHWAY_MAP

    gene_set = set(gene_names)
    coverage = {}
    for pathway, members in pathway_map.items():
        covered = [g for g in members if g in gene_set]
        coverage[pathway] = {
            "covered": len(covered),
            "total": len(members),
            "fraction": len(covered) / len(members) if members else 0.0,
            "covered_genes": covered,
        }
    return coverage


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    gene_names = torch.load("data/processed/gene_names.pt", weights_only=False)
    print("Pathway Coverage in Norman Gene Vocabulary:")
    cov = get_pathway_gene_coverage(gene_names)
    for pathway, info in cov.items():
        print(f"  {pathway}: {info['covered']}/{info['total']} genes ({info['fraction']*100:.0f}%)")

    # Simulate random attribution for demo
    num_genes = len(gene_names)
    fake_attr = torch.randn(num_genes)
    report_pathway_attribution(fake_attr, gene_names, top_k=5)
