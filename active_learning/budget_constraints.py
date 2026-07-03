"""
budget_constraints.py
---------------------
Enforce experimental budget limits on active-learning acquisition.

Provides:
  - BudgetManager: tracks remaining experiment budget and enforces caps
  - CostModel: estimates cost of a perturbation experiment
  - filter_by_budget: removes candidates that exceed remaining budget
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

@dataclass
class ExperimentCost:
    """
    Estimated cost of running a perturbation experiment.

    Fields:
        base_cost    : default per-experiment cost unit (e.g. 1.0)
        double_ko_multiplier: multiplier for double-KO experiments
        sequencing_depth_factor: additional cost for high-depth sequencing
    """
    base_cost: float = 1.0
    double_ko_multiplier: float = 1.5
    sequencing_depth_factor: float = 1.0

    def estimate(self, n_genes_perturbed: int) -> float:
        """Return estimated cost for perturbing n genes simultaneously."""
        if n_genes_perturbed == 1:
            return self.base_cost * self.sequencing_depth_factor
        elif n_genes_perturbed == 2:
            return self.base_cost * self.double_ko_multiplier * self.sequencing_depth_factor
        else:
            # Higher-order perturbations scale super-linearly
            return self.base_cost * (self.double_ko_multiplier ** (n_genes_perturbed - 1)) * self.sequencing_depth_factor


# ---------------------------------------------------------------------------
# Budget manager
# ---------------------------------------------------------------------------

class BudgetManager:
    """
    Tracks consumed experimental budget and enforces a hard cap.

    Args:
        total_budget: total budget units available
        cost_model  : ExperimentCost instance (uses default if None)
    """

    def __init__(
        self,
        total_budget: float,
        cost_model: Optional[ExperimentCost] = None,
    ):
        self.total_budget  = total_budget
        self.remaining     = total_budget
        self.cost_model    = cost_model or ExperimentCost()
        self._history: List[dict] = []

    @property
    def spent(self) -> float:
        return self.total_budget - self.remaining

    def can_afford(self, n_genes: int = 1) -> bool:
        """Check whether a new experiment would fit within the remaining budget."""
        return self.remaining >= self.cost_model.estimate(n_genes)

    def consume(self, gene_indices: List[int], label: str = "") -> float:
        """
        Record a consumed experiment.

        Args:
            gene_indices: genes perturbed in this experiment
            label       : optional human-readable label

        Returns:
            cost: the budget units consumed

        Raises:
            BudgetExhaustedError if the cost exceeds remaining budget
        """
        cost = self.cost_model.estimate(len(gene_indices))
        if cost > self.remaining:
            raise BudgetExhaustedError(
                f"Experiment cost {cost:.2f} exceeds remaining budget {self.remaining:.2f}."
            )
        self.remaining -= cost
        self._history.append({"genes": gene_indices, "cost": cost, "label": label})
        return cost

    def reset(self) -> None:
        self.remaining = self.total_budget
        self._history.clear()

    def summary(self) -> dict:
        return {
            "total_budget": self.total_budget,
            "spent": self.spent,
            "remaining": self.remaining,
            "n_experiments": len(self._history),
        }

    def __repr__(self) -> str:
        return (
            f"BudgetManager(total={self.total_budget}, "
            f"spent={self.spent:.2f}, remaining={self.remaining:.2f})"
        )


class BudgetExhaustedError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

def filter_by_budget(
    candidates: List[int],
    manager: BudgetManager,
    n_genes_per_experiment: int = 1,
) -> List[int]:
    """
    Return only those candidates that fit within the remaining budget.

    Args:
        candidates             : list of gene indices
        manager                : BudgetManager instance
        n_genes_per_experiment : how many genes are perturbed per experiment

    Returns:
        Filtered list of affordable candidates.
    """
    cost_per = manager.cost_model.estimate(n_genes_per_experiment)
    affordable = [g for g in candidates if manager.remaining >= cost_per]
    if len(affordable) < len(candidates):
        dropped = len(candidates) - len(affordable)
        print(
            f"[budget_constraints] Dropped {dropped} candidates due to budget limit "
            f"(remaining={manager.remaining:.2f}, cost_per={cost_per:.2f})."
        )
    return affordable


def greedy_budget_selection(
    ranked_candidates: List[int],
    manager: BudgetManager,
    n_genes_per_experiment: int = 1,
) -> List[int]:
    """
    Select as many top-ranked candidates as the budget allows,
    consuming budget as we go.

    Args:
        ranked_candidates: candidates in priority order (best first)
        manager          : BudgetManager instance
        n_genes_per_experiment

    Returns:
        List of selected gene indices (subset of ranked_candidates).
    """
    selected = []
    for gene in ranked_candidates:
        if not manager.can_afford(n_genes_per_experiment):
            print("[budget_constraints] Budget exhausted – stopping selection.")
            break
        manager.consume([gene])
        selected.append(gene)
    return selected
