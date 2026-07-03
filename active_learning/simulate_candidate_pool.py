import torch
import numpy as np

def generate_candidate_pool(all_genes_vocab, train_interventions):
    """
    Generates a pool of candidate UNSEEN SINGLE-GENE perturbations.
    
    Args:
        all_genes_vocab (list or set): All possible genes in the vocabulary.
        train_interventions (torch.Tensor): Tensor of shape [N_train, 2] indicating
            interventions already in the training set.
            Format assumes [gene_idx, -1] for single-gene perturbations.
            
    Returns:
        list of int: Candidate gene indices that are not in the training set.
    """
    # Extract unique single genes already seen in training
    seen_single_genes = set()
    for i in range(train_interventions.size(0)):
        # Check if it's a single gene intervention (e.g., [target, -1])
        intervention = train_interventions[i]
        # In this dataset, if the second element is -1 or matches the first, it's single
        gene1, gene2 = intervention[0].item(), intervention[1].item()
        if gene1 != -1 and (gene2 == -1 or gene1 == gene2):
            seen_single_genes.add(gene1)
            
    # Generate unseen single-gene candidates
    candidates = []
    for gene in all_genes_vocab:
        if gene not in seen_single_genes:
            candidates.append(gene)
            
    return candidates
