import torch

def compute_raw_diversity(candidate_z, selected_z_list):
    """
    Computes the raw diversity score as the mean L2 distance between the candidate's
    mean latent representation and the representations of the already selected candidates.
    
    Args:
        candidate_z (torch.Tensor): Mean latent representation of the candidate.
        selected_z_list (list of torch.Tensor): Latent representations of selected candidates.
            
    Returns:
        float: The raw diversity distance.
    """
    if not selected_z_list:
        return 0.0 # No diversity penalty for the first selection
        
    distances = []
    for selected_z in selected_z_list:
        # Compute mean L2 distance across the batch/gene dimensions
        dist = torch.norm(candidate_z - selected_z, p=2, dim=-1).mean().item()
        distances.append(dist)
        
    return sum(distances) / len(distances)

def normalize_diversity(raw_diversity_dict):
    """
    Normalizes diversity distances across the candidate pool.
    NormalizedDistance = ||z_i - z_j|| / mean_distance_across_pool
    
    Args:
        raw_diversity_dict (dict): Mapping candidate -> raw_diversity_score
        
    Returns:
        dict: Mapping candidate -> normalized_diversity_score
    """
    # If there are no raw scores or they are all 0 (first selection), return as is
    values = list(raw_diversity_dict.values())
    if not values or sum(values) == 0:
        return {k: 0.0 for k in raw_diversity_dict}
        
    mean_distance_across_pool = sum(values) / len(values)
    
    # Precaution against division by zero in edge cases
    if mean_distance_across_pool == 0:
        return {k: 0.0 for k in raw_diversity_dict}
        
    return {k: v / mean_distance_across_pool for k, v in raw_diversity_dict.items()}
