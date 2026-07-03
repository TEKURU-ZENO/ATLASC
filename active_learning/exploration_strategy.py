"""
exploration_strategy.py
-----------------------
High-level exploration strategy wrappers for ATLAS active learning.

These classes orchestrate multiple active-learning iterations, each of
which calls the acquisition pipeline, respects a budget, and updates the
set of tested genes.

Strategies:
  - EpsilonGreedyExploration : with probability ε select a random gene,
                               otherwise select the highest-scoring one
  - UCBExploration           : Upper-Confidence-Bound balancing
                               exploitation (mean Utraj) and exploration
                               (uncertainty / visit count)
  - PureGreedyExploration    : always select the top-ranked candidate
                               (baseline)
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Set, Tuple

import torch


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseExplorationStrategy:
    """Abstract base for exploration strategies."""

    def __init__(self, seed: Optional[int] = 42):
        self.rng = random.Random(seed)
        self._tested: Set[int] = set()
        self._visit_counts: Dict[int, int] = {}
        self._score_history: Dict[int, List[float]] = {}
        self._iteration: int = 0

    def select(
        self,
        candidates: List[int],
        scores: Dict[int, float],
        k: int = 1,
    ) -> List[int]:
        raise NotImplementedError

    def mark_selected(self, genes: List[int]) -> None:
        """Record that these genes have been selected / tested."""
        for g in genes:
            self._tested.add(g)
            self._visit_counts[g] = self._visit_counts.get(g, 0) + 1
        self._iteration += 1

    @property
    def tested_genes(self) -> Set[int]:
        return self._tested

    def reset(self) -> None:
        self._tested.clear()
        self._visit_counts.clear()
        self._score_history.clear()
        self._iteration = 0


# ---------------------------------------------------------------------------
# Pure Greedy (baseline)
# ---------------------------------------------------------------------------

class PureGreedyExploration(BaseExplorationStrategy):
    """
    Always selects the top-k scoring candidates.
    No exploration component – pure exploitation.
    """

    def select(
        self,
        candidates: List[int],
        scores: Dict[int, float],
        k: int = 1,
    ) -> List[int]:
        available = [g for g in candidates if g not in self._tested]
        if not available:
            return []
        ranked = sorted(available, key=lambda g: scores.get(g, 0.0), reverse=True)
        selected = ranked[:k]
        self.mark_selected(selected)
        return selected


# ---------------------------------------------------------------------------
# Epsilon-Greedy
# ---------------------------------------------------------------------------

class EpsilonGreedyExploration(BaseExplorationStrategy):
    """
    With probability ε draw a random gene; otherwise pick the best.

    Args:
        epsilon     : exploration probability in [0, 1]
        epsilon_decay: multiplicative decay per iteration (set to 1.0 to disable)
        min_epsilon : floor for ε after decay
    """

    def __init__(
        self,
        epsilon: float = 0.15,
        epsilon_decay: float = 0.99,
        min_epsilon: float = 0.05,
        seed: Optional[int] = 42,
    ):
        super().__init__(seed=seed)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def select(
        self,
        candidates: List[int],
        scores: Dict[int, float],
        k: int = 1,
    ) -> List[int]:
        available = [g for g in candidates if g not in self._tested]
        if not available:
            return []

        selected = []
        remaining = list(available)

        for _ in range(min(k, len(remaining))):
            if not remaining:
                break
            eps = max(self.epsilon, self.min_epsilon)
            if self.rng.random() < eps:
                # Explore: random pick
                gene = self.rng.choice(remaining)
            else:
                # Exploit: highest score
                gene = max(remaining, key=lambda g: scores.get(g, 0.0))

            selected.append(gene)
            remaining.remove(gene)

        # Decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
        self.mark_selected(selected)
        return selected


# ---------------------------------------------------------------------------
# UCB (Upper Confidence Bound)
# ---------------------------------------------------------------------------

class UCBExploration(BaseExplorationStrategy):
    """
    UCB1-style selection balancing:
      score(g) = mean_utraj(g) + c * sqrt(log(t+1) / (n_visits(g) + 1))

    Args:
        exploration_c : controls exploration-exploitation trade-off
    """

    def __init__(self, exploration_c: float = 1.0, seed: Optional[int] = 42):
        super().__init__(seed=seed)
        self.c = exploration_c

    def ucb_score(
        self,
        gene: int,
        base_score: float,
        t: int,
    ) -> float:
        n = self._visit_counts.get(gene, 0)
        bonus = self.c * math.sqrt(math.log(t + 1) / (n + 1))
        return base_score + bonus

    def select(
        self,
        candidates: List[int],
        scores: Dict[int, float],
        k: int = 1,
    ) -> List[int]:
        available = [g for g in candidates if g not in self._tested]
        if not available:
            return []

        t = self._iteration + 1
        ucb_scores = {g: self.ucb_score(g, scores.get(g, 0.0), t) for g in available}
        ranked = sorted(available, key=lambda g: ucb_scores[g], reverse=True)

        selected = ranked[:k]
        self.mark_selected(selected)
        return selected


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY = {
    "greedy":        PureGreedyExploration,
    "epsilon_greedy": EpsilonGreedyExploration,
    "ucb":           UCBExploration,
}


def build_strategy(
    name: str = "ucb",
    **kwargs,
) -> BaseExplorationStrategy:
    """
    Instantiate an exploration strategy by name.

    Args:
        name   : one of 'greedy', 'epsilon_greedy', 'ucb'
        kwargs : forwarded to strategy constructor

    Returns:
        Instantiated strategy.
    """
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{name}'. Choose from: {list(STRATEGY_REGISTRY.keys())}"
        )
    return STRATEGY_REGISTRY[name](**kwargs)
