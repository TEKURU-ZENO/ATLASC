import torch

def compute_hub_penalty(candidate_gene_idx, edge_index, epsilon=1.0):
    """
    Computes the Hub Penalty to avoid repeatedly sampling central TFs.
    Uses dampening to smooth out heavy-tailed degree distributions.
    
    HubPenalty = 1 / (sqrt(degree(gene)) + epsilon)
    
    Args:
        candidate_gene_idx (int): The index of the candidate gene.
        edge_index (torch.Tensor): The GRN edge index [2, num_edges].
        epsilon (float): Smoothing parameter to prevent division by zero 
                         and overly favoring extremely low-degree nodes.
        
    Returns:
        float: The raw hub penalty score.
    """
    # Calculate degree (in + out edges) to capture total connectivity
    out_edges = (edge_index[0] == candidate_gene_idx).sum().item()
    in_edges = (edge_index[1] == candidate_gene_idx).sum().item()
    total_degree = out_edges + in_edges
    
    penalty = 1.0 / (torch.sqrt(torch.tensor(total_degree, dtype=torch.float32)) + epsilon)
    return penalty.item()
