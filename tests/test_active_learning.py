import unittest
import torch
import numpy as np
from active_learning.budget_constraints import PlateLayout, BudgetManager
from active_learning.experiment_ranker import UnifiedExperimentRanker
from active_learning.bayesian_acquisition import BatchBayesianAcquisition


class TestActiveLearning(unittest.TestCase):
    def test_plate_layout_96_and_384(self):
        # 96-well plate test
        plate96 = PlateLayout(format_type="96", control_wells=8, blank_wells=4, replicates_per_perturbation=2)
        df96 = plate96.allocate_plate([10, 20, 30])
        self.assertEqual(len(df96), 96)
        self.assertEqual(df96["Sample_Type"].value_counts()["Control_NegCtrl"], 8)
        self.assertEqual(df96["Sample_Type"].value_counts()["Blank_Media"], 4)
        self.assertEqual(df96["Sample_Type"].value_counts()["Perturbation"], 6)

        # 384-well plate test
        plate384 = PlateLayout(format_type="384", control_wells=16, blank_wells=8, replicates_per_perturbation=2)
        df384 = plate384.allocate_plate([1, 2, 3, 4, 5])
        self.assertEqual(len(df384), 384)

    def test_unified_experiment_ranker(self):
        ranker = UnifiedExperimentRanker()
        candidates = [101, 102, 103]
        u_traj = {101: 0.8, 102: 0.5, 103: 0.9}
        diversity = {101: 1.0, 102: 0.8, 103: 0.5}
        hub = {101: 0.1, 102: 0.2, 103: 0.05}

        # Discovery mode
        ranked_disc = ranker.rank_discovery(candidates, u_traj, diversity, hub)
        self.assertEqual(len(ranked_disc), 3)

        # Synthetic lethality mode
        pred_deltas = {
            101: torch.tensor([0.1, -0.2, 0.0]),
            102: torch.tensor([-1.5, -2.0, -0.5]),
            103: torch.tensor([0.0, 0.1, 0.0]),
        }
        essential_genes = [0, 1]
        ranked_sl = ranker.rank_synthetic_lethality(
            candidates, pred_deltas, essential_genes, u_traj, lethality_weight=2.0
        )
        self.assertEqual(len(ranked_sl), 3)
        self.assertEqual(ranked_sl[0][0], 102)  # Strongest knock-down of essential genes

    def test_bayesian_acquisition(self):
        acq = BatchBayesianAcquisition(mc_samples=64, seed=42)
        candidate_preds = {
            101: torch.randn(3, 20),
            102: torch.randn(3, 20) * 2.0,
            103: torch.randn(3, 20) * 0.5,
        }

        q_ei = acq.compute_q_expected_improvement(candidate_preds, best_observed_value=1.0)
        self.assertEqual(len(q_ei), 3)
        self.assertTrue(all(v >= 0.0 for v in q_ei.values()))

        selected_ts = acq.thompson_sampling_batch(
            candidates=[101, 102, 103],
            candidate_predictions=candidate_preds,
            batch_size=2,
        )
        self.assertEqual(len(selected_ts), 2)


if __name__ == "__main__":
    unittest.main()
