import unittest
import torch
import torch.nn as nn
from models.gnn.causal_gnn import CausalGNN
from models.gnn.neural_ode_gnn import CausalGraphODEFunc, ContinuousCausalRollout
from causal.counterfactuals import compute_combinatorial_synergy
from simulation.rollout_engine import RolloutEngine
from models.ensemble import EnsembleWrapper


class MockEncoder(nn.Module):
    def __init__(self, num_genes=10, latent_dim=8):
        super().__init__()
        self.proj = nn.Linear(num_genes, latent_dim)

    def forward(self, x):
        # [B, G] -> [B, G, L]
        return self.proj(x).unsqueeze(1).repeat(1, x.size(1), 1)


class TestGNNAndDynamics(unittest.TestCase):
    def test_signed_causal_gnn(self):
        gnn = CausalGNN(in_channels=16, hidden_channels=16, out_channels=1, layers_count=2)
        z = torch.randn(2, 20, 16)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        edge_weight = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float32)

        # Forward with signed edge weights
        z_next = gnn(z, edge_index, edge_weight=edge_weight)
        self.assertEqual(z_next.shape, (2, 20, 16))

        # Decode
        x_hat = gnn.decode(z_next)
        self.assertEqual(x_hat.shape, (2, 20))

    def test_neural_ode_continuous_rollout(self):
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        edge_weight = torch.tensor([1.0, -1.0], dtype=torch.float32)
        ode_func = CausalGraphODEFunc(hidden_channels=16, edge_index=edge_index, edge_weight=edge_weight)
        rollout = ContinuousCausalRollout(ode_func, solver="rk4")

        z0 = torch.randn(2, 20, 16)
        eval_times = [0.0, 0.25, 0.5, 0.75, 1.0]

        z_final, trajectory = rollout.integrate(z0, eval_times=eval_times, substeps=2)
        self.assertEqual(trajectory.shape, (5, 2, 20, 16))
        self.assertEqual(z_final.shape, (2, 20, 16))

    def test_combinatorial_synergy(self):
        num_genes = 10
        latent_dim = 8
        encoders = [MockEncoder(num_genes, latent_dim), MockEncoder(num_genes, latent_dim)]
        gnns = [
            CausalGNN(in_channels=latent_dim, hidden_channels=latent_dim, out_channels=1, layers_count=1),
            CausalGNN(in_channels=latent_dim, hidden_channels=latent_dim, out_channels=1, layers_count=1),
        ]
        ensemble = EnsembleWrapper(encoders, gnns)
        simulator = RolloutEngine(ensemble, execution_mode="deterministic")

        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        base_cells = torch.randn(2, num_genes)

        res = compute_combinatorial_synergy(
            ensemble=ensemble,
            simulator=simulator,
            edge_index=edge_index,
            base_cells=base_cells,
            gene_combination=[0, 1],
            rollout_steps=1,
            device="cpu",
        )
        self.assertIn("epistasis_vector", res)
        self.assertIn("epistasis_magnitude", res)
        self.assertIn("is_synergistic", res)


if __name__ == "__main__":
    unittest.main()
