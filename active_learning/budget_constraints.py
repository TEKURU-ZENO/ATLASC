"""
budget_constraints.py
---------------------
Enforce experimental budget limits and microplate layouts (96-well / 384-well)
on active-learning perturbation acquisitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd


@dataclass
class ExperimentCost:
    """
    Estimated cost of running a perturbation experiment.
    """
    base_cost: float = 1.0
    double_ko_multiplier: float = 1.5
    sequencing_depth_factor: float = 1.0

    def estimate(self, n_genes_perturbed: int) -> float:
        if n_genes_perturbed <= 1:
            return self.base_cost * self.sequencing_depth_factor
        elif n_genes_perturbed == 2:
            return self.base_cost * self.double_ko_multiplier * self.sequencing_depth_factor
        else:
            return self.base_cost * (self.double_ko_multiplier ** (n_genes_perturbed - 1)) * self.sequencing_depth_factor


class BudgetManager:
    """
    Tracks consumed experimental budget and enforces budget caps.
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
        return self.remaining >= self.cost_model.estimate(n_genes)

    def consume(self, gene_indices: List[int], label: str = "") -> float:
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


class BudgetExhaustedError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Plate Layout Designer (96-Well / 384-Well Wet-Lab Automation)
# ---------------------------------------------------------------------------

class PlateLayout:
    """
    Wet-lab automated plate mapping for active-learning candidate batches.
    Supports 96-well (8x12) and 384-well (16x24) standard microplates.
    """
    def __init__(
        self,
        format_type: str = "96",
        control_wells: int = 8,
        blank_wells: int = 4,
        replicates_per_perturbation: int = 2,
    ):
        self.format_type = format_type
        if format_type == "96":
            self.rows = [chr(65 + i) for i in range(8)]   # A-H
            self.cols = list(range(1, 13))                 # 1-12
        elif format_type == "384":
            self.rows = [chr(65 + i) for i in range(16)]  # A-P
            self.cols = list(range(1, 25))                 # 1-24
        else:
            raise ValueError(f"Unsupported plate format: {format_type}")

        self.total_wells = len(self.rows) * len(self.cols)
        self.control_wells = control_wells
        self.blank_wells = blank_wells
        self.replicates = replicates_per_perturbation
        self.available_perturbation_wells = self.total_wells - (control_wells + blank_wells)
        self.max_perturbations = self.available_perturbation_wells // self.replicates

    def allocate_plate(
        self,
        selected_genes: List[int],
        gene_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Maps selected active learning gene perturbations to physical microplate wells.
        
        Returns:
            DataFrame with columns: [Well, Row, Column, Sample_Type, Gene_Idx, Gene_Symbol, Replicate]
        """
        wells = [f"{r}{c:02d}" for r in self.rows for c in self.cols]
        records = []
        well_idx = 0

        # 1. Allocate Controls (NegCtrl)
        for c in range(self.control_wells):
            w = wells[well_idx]
            r, col = w[0], int(w[1:])
            records.append({
                "Well": w, "Row": r, "Column": col,
                "Sample_Type": "Control_NegCtrl",
                "Gene_Idx": -1, "Gene_Symbol": "NegCtrl", "Replicate": c + 1
            })
            well_idx += 1

        # 2. Allocate Blanks
        for b in range(self.blank_wells):
            w = wells[well_idx]
            r, col = w[0], int(w[1:])
            records.append({
                "Well": w, "Row": r, "Column": col,
                "Sample_Type": "Blank_Media",
                "Gene_Idx": -1, "Gene_Symbol": "Blank", "Replicate": b + 1
            })
            well_idx += 1

        # 3. Allocate Perturbations
        capped_genes = selected_genes[:self.max_perturbations]
        for g_idx in capped_genes:
            symbol = gene_names[g_idx] if (gene_names and g_idx < len(gene_names)) else f"GENE_{g_idx}"
            for rep in range(self.replicates):
                if well_idx >= self.total_wells:
                    break
                w = wells[well_idx]
                r, col = w[0], int(w[1:])
                records.append({
                    "Well": w, "Row": r, "Column": col,
                    "Sample_Type": "Perturbation",
                    "Gene_Idx": g_idx, "Gene_Symbol": symbol, "Replicate": rep + 1
                })
                well_idx += 1

        # 4. Fill remaining empty wells
        while well_idx < self.total_wells:
            w = wells[well_idx]
            r, col = w[0], int(w[1:])
            records.append({
                "Well": w, "Row": r, "Column": col,
                "Sample_Type": "Unused_Empty",
                "Gene_Idx": -1, "Gene_Symbol": "None", "Replicate": 0
            })
            well_idx += 1

        return pd.DataFrame(records)


def filter_by_budget(
    candidates: List[int],
    manager: BudgetManager,
    n_genes_per_experiment: int = 1,
) -> List[int]:
    cost_per = manager.cost_model.estimate(n_genes_per_experiment)
    return [g for g in candidates if manager.remaining >= cost_per]


def greedy_budget_selection(
    ranked_candidates: List[int],
    manager: BudgetManager,
    n_genes_per_experiment: int = 1,
) -> List[int]:
    selected = []
    for gene in ranked_candidates:
        if not manager.can_afford(n_genes_per_experiment):
            break
        manager.consume([gene])
        selected.append(gene)
    return selected
