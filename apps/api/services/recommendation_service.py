"""
recommendation_service.py
-------------------------
Business logic for the /recommend endpoint.

Wraps the active-learning acquisition pipeline and exploration strategy
into a service callable by the route handler.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from services.model_service import get_ensemble, get_device, get_num_genes
from utils.data_loader import get_trrust_graph, NormanDataset
from simulation.rollout_engine import RolloutEngine
from causal.do_operator import apply_intervention
from active_learning.acquisition import compute_candidate_metrics, rank_and_select
from active_learning.candidate_generator import combined_candidates
from active_learning.budget_constraints import BudgetManager
from active_learning.exploration_strategy import build_strategy


def recommend_perturbations(
    candidate_genes: Optional[List[int]] = None,
    k_to_select: int = 5,
    rollout_horizon: int = 3,
    lambda_diversity: float = 1.0,
    lambda_hub: float = 1.0,
    budget: float = 10.0,
    strategy: str = "ucb",
    ablation_mode: str = "full",
) -> dict:
    """
    Run the full active-learning recommendation pipeline.

    Args:
        candidate_genes   : gene indices to score; if None, uses combined_candidates()
        k_to_select       : how many experiments to recommend
        rollout_horizon   : rollout steps for uncertainty estimation
        lambda_diversity  : weight for diversity component
        lambda_hub        : weight for hub-penalty component
        budget            : total experiment budget
        strategy          : 'ucb' | 'epsilon_greedy' | 'greedy'
        ablation_mode     : 'full' | 'utraj_only'

    Returns:
        Serialisable dict with selected genes and explanation scores.
    """
    ensemble   = get_ensemble()
    device     = get_device()
    num_genes  = get_num_genes()

    edge_index = get_trrust_graph(processed_dir="data/processed").to(device)

    # Build candidate pool if not provided
    if candidate_genes is None:
        candidate_genes = combined_candidates(
            num_genes=num_genes,
            edge_index=edge_index.cpu(),
            top_k_hub=50,
            random_k=30,
            seed=42,
        )

    # Load base (unperturbed) cells
    base_cells = _load_base_cells(num_genes, device)

    simulator = RolloutEngine(ensemble, execution_mode="deterministic")

    # Compute acquisition metrics for all candidates
    u_traj_raw, z_mean_reps, hub_penalties = compute_candidate_metrics(
        candidates=candidate_genes,
        ensemble=ensemble,
        simulator=simulator,
        edge_index=edge_index,
        base_cells=base_cells,
        apply_intervention_fn=apply_intervention,
        rollout_steps=rollout_horizon,
        device=device,
    )

    # Budget-constrained selection via exploration strategy
    manager  = BudgetManager(total_budget=budget)
    explorer = build_strategy(strategy)

    # Get ranked scores dict for strategy
    scores = {g: u_traj_raw.get(g, 0.0) for g in candidate_genes}
    strategy_selected = explorer.select(candidate_genes, scores, k=k_to_select)

    # Greedy fine-grained ranking over strategy pre-selection
    k = min(k_to_select, len(strategy_selected))
    selected, explanations = rank_and_select(
        candidates=strategy_selected,
        u_traj_raw={g: u_traj_raw.get(g, 0.0) for g in strategy_selected},
        z_mean_reps={g: z_mean_reps.get(g) for g in strategy_selected if g in z_mean_reps},
        hub_penalties={g: hub_penalties.get(g, 0.0) for g in strategy_selected},
        k=k,
        lambda1=lambda_diversity,
        lambda2=lambda_hub,
        ablation_mode=ablation_mode,
    )

    return {
        "selected_genes": selected,
        "explanations":   explanations,
        "strategy":       strategy,
        "budget_summary": manager.summary(),
        "candidate_pool_size": len(candidate_genes),
    }


def _load_base_cells(num_genes: int, device: torch.device) -> torch.Tensor:
    """Load a small batch of unperturbed reference cells."""
    import os
    if os.path.exists("data/processed/cells_data.pt"):
        try:
            ds = NormanDataset(processed_dir="data/processed")
            valid = []
            for i in range(len(ds)):
                if ds.interventions[i][0].item() == -1:
                    valid.append(i)
                if len(valid) >= 10:
                    break
            if valid:
                return torch.stack([ds[v][0] for v in valid]).to(device)
        except Exception:
            pass
    return torch.zeros(10, num_genes, device=device)
