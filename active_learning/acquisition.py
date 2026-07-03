import torch
import numpy as np
from tqdm import tqdm
from .diversity import compute_raw_diversity, normalize_diversity
from .hub_regularizer import compute_hub_penalty

def normalize_scores(score_dict):
    """
    Min-Max normalization of scores across candidates so components are on the same scale.
    
    Args:
        score_dict (dict): Mapping candidate -> raw score
        
    Returns:
        dict: Mapping candidate -> normalized score [0, 1]
    """
    if not score_dict:
        return {}
        
    values = list(score_dict.values())
    max_val = max(values)
    min_val = min(values)
    
    if max_val == min_val:
        return {k: 0.0 for k in score_dict}
        
    return {k: (v - min_val) / (max_val - min_val) for k, v in score_dict.items()}

def compute_candidate_metrics(
    candidates, 
    ensemble, 
    simulator, 
    edge_index, 
    base_cells, 
    apply_intervention_fn,
    rollout_steps=3,
    device='cpu'
):
    """
    Computes Utraj and mean latent representations for all candidates.
    
    Args:
        candidates (list): List of candidate gene indices.
        ensemble (EnsembleWrapper): The trained models.
        simulator (RolloutEngine): Simulator engine.
        edge_index (torch.Tensor): GRN topology.
        base_cells (torch.Tensor): A batch of unperturbed cells [batch, genes]
        apply_intervention_fn (callable): Function to apply intervention `apply_intervention(z, target, value)`
        
    Returns:
        dict: candidate -> raw Utraj score
        dict: candidate -> mean latent representation [batch, genes, latent]
        dict: candidate -> raw Hub Penalty
    """
    u_traj_raw = {}
    z_mean_reps = {}
    hub_penalties = {}
    
    ensemble.eval()
    
    # Pre-encode base cells (since intervention is on latent)
    with torch.no_grad():
        x = base_cells.to(device)
        z0_k = ensemble.encode(x) # [K, batch, genes, latent]
        
        for candidate_gene in tqdm(candidates, desc="Computing Candidate Pool Metrics"):
            # 1. Apply Intervention to all K models
            # interventions shape: [batch, 2]. We format it for single gene: [candidate_gene, -1]
            interventions = torch.tensor([[candidate_gene, -1]] * x.size(0), device=device)
            
            z_intervened_k = []
            for k in range(ensemble.k_models):
                z_k = apply_intervention_fn(z0_k[k], interventions, value=0.0)
                z_intervened_k.append(z_k)
            z_intervened_k = torch.stack(z_intervened_k, dim=0)
            
            # 2. Rollout
            _, trajectory = simulator.rollout(
                z_intervened_k,
                edge_index,
                steps=rollout_steps,
                log_trajectory=True,
                enforce_stability=False
            ) # trajectory: [T, K, batch, genes, latent]
            
            # 3. Compute Epistemic Uncertainty
            u_traj, _ = ensemble.compute_epistemic_uncertainty(trajectory) # [batch, genes]
            # Average across cells and genes
            u_scalar = u_traj.mean().item()
            u_traj_raw[candidate_gene] = u_scalar
            
            # 4. Mean Representation (average across time T and models K)
            z_mean = trajectory.mean(dim=0).mean(dim=0) # [batch, genes, latent]
            z_mean_reps[candidate_gene] = z_mean
            
            # 5. Hub Penalty
            hub_penalties[candidate_gene] = compute_hub_penalty(candidate_gene, edge_index)
            
    return u_traj_raw, z_mean_reps, hub_penalties

def rank_and_select(
    candidates, 
    u_traj_raw, 
    z_mean_reps, 
    hub_penalties, 
    k=10, 
    lambda1=1.0, 
    lambda2=1.0, 
    ablation_mode="full"
):
    """
    Greedy acquisition of top-K subset.
    
    ablation_mode: 
        "full" -> Utraj + lambda1*Diversity + lambda2*HubPenalty
        "utraj_only" -> Utraj only
    """
    selected_candidates = []
    selected_z_list = []
    
    # Pre-normalize static components
    u_traj_norm = normalize_scores(u_traj_raw)
    hub_norm = normalize_scores(hub_penalties)
    
    pool = set(candidates)
    
    # Store scores for API returning
    final_explanations = {}
    
    for step in range(k):
        if not pool:
            break
            
        best_candidate = None
        best_score = -float('inf')
        step_best_explanation = {}
        
        # Diversity requires computing distance against *currently* selected candidates
        raw_diversities = {}
        if ablation_mode == "full":
            for cand in pool:
                raw_diversities[cand] = compute_raw_diversity(z_mean_reps[cand], selected_z_list)
            div_norm = normalize_diversity(raw_diversities)
        else:
            div_norm = {cand: 0.0 for cand in pool}
            
        for cand in pool:
            u_val = u_traj_norm[cand]
            
            if ablation_mode == "full":
                div_val = div_norm[cand]
                hub_val = hub_norm[cand]
                score = u_val + (lambda1 * div_val) + (lambda2 * hub_val)
            else:
                div_val = 0.0
                hub_val = 0.0
                score = u_val
                
            if score > best_score:
                best_score = score
                best_candidate = cand
                step_best_explanation = {
                    "rank": step + 1,
                    "Total_Score": score,
                    "Utraj_Norm": u_val,
                    "Diversity_Norm": div_val,
                    "Hub_Norm": hub_val,
                    "Utraj_Raw": u_traj_raw[cand]
                }
                
        # Commit selection
        selected_candidates.append(best_candidate)
        selected_z_list.append(z_mean_reps[best_candidate])
        pool.remove(best_candidate)
        final_explanations[best_candidate] = step_best_explanation
        
    return selected_candidates, final_explanations
