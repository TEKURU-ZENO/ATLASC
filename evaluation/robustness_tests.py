"""
robustness_tests.py

Systematic robustness tests for ATLAS predictions.

Tests include:
1. Input noise robustness: does MSE degrade gracefully under input perturbation?
2. Ensemble consistency: are K model predictions stable?
3. Intervention sensitivity: does applying a no-op intervention change MSE?
4. OOD stress test: simulate extrapolation beyond training gene value range.
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, List
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ensemble import EnsembleWrapper
from causal.do_operator import apply_intervention
from simulation.rollout_engine import RolloutEngine


def test_input_noise_robustness(
    ensemble: EnsembleWrapper,
    simulator: RolloutEngine,
    dataloader: DataLoader,
    edge_index: torch.Tensor,
    device: torch.device,
    noise_levels: List[float] = [0.0, 0.05, 0.1, 0.2, 0.5],
    steps: int = 3,
) -> Dict[float, float]:
    """
    Measure how MSE degrades as additive Gaussian noise is added to input x.

    Returns:
        Dict mapping noise_std -> mean MSE.
    """
    ensemble.eval()
    criterion = torch.nn.MSELoss(reduction='none')
    results = {}

    for noise_std in noise_levels:
        mse_list = []
        with torch.no_grad():
            for x, interventions, targets in dataloader:
                x = x.to(device)
                targets = targets.to(device)
                if noise_std > 0:
                    x = x + torch.randn_like(x) * noise_std

                z0_k = ensemble.encode(x)
                z_int_k = torch.stack([
                    apply_intervention(z0_k[k], interventions, value=0.0)
                    for k in range(ensemble.k_models)
                ], dim=0)
                latent_k, _ = simulator.rollout(
                    z_int_k, edge_index, steps=steps, log_trajectory=False
                )
                preds = ensemble.decode(latent_k).mean(dim=0)
                mse = criterion(preds, targets).mean(dim=-1).cpu().numpy()
                mse_list.extend(mse)
        results[noise_std] = float(np.mean(mse_list))
        print(f"Noise std={noise_std:.2f} -> Mean MSE: {results[noise_std]:.4f}")

    return results


def test_ensemble_consistency(
    ensemble: EnsembleWrapper,
    simulator: RolloutEngine,
    dataloader: DataLoader,
    edge_index: torch.Tensor,
    device: torch.device,
    steps: int = 3,
) -> Dict[str, float]:
    """
    Measure pairwise prediction disagreement between all K ensemble members.

    A well-trained diverse ensemble should have moderate (not zero, not extreme)
    disagreement.

    Returns:
        Dict with 'mean_pairwise_mse' and 'max_pairwise_mse'.
    """
    ensemble.eval()
    K = ensemble.k_models
    pairwise_mses = []

    with torch.no_grad():
        for x, interventions, _ in dataloader:
            x = x.to(device)
            z0_k = ensemble.encode(x)
            z_int_k = torch.stack([
                apply_intervention(z0_k[k], interventions, value=0.0)
                for k in range(K)
            ], dim=0)
            latent_k, _ = simulator.rollout(
                z_int_k, edge_index, steps=steps, log_trajectory=False
            )
            preds_k = ensemble.decode(latent_k)  # [K, batch, genes]
            for k1 in range(K):
                for k2 in range(k1 + 1, K):
                    diff = ((preds_k[k1] - preds_k[k2]) ** 2).mean().item()
                    pairwise_mses.append(diff)

    return {
        "mean_pairwise_mse": float(np.mean(pairwise_mses)),
        "max_pairwise_mse": float(np.max(pairwise_mses)),
    }


def test_noop_intervention(
    ensemble: EnsembleWrapper,
    simulator: RolloutEngine,
    dataloader: DataLoader,
    edge_index: torch.Tensor,
    device: torch.device,
    steps: int = 3,
) -> Dict[str, float]:
    """
    Compare predictions with and without a no-op intervention
    (all intervention indices set to -1).

    A no-op should produce predictions matching undisturbed rollout.

    Returns:
        Dict with 'mse_with_intervention', 'mse_no_intervention', 'delta'.
    """
    ensemble.eval()
    criterion = torch.nn.MSELoss(reduction='none')
    mse_with, mse_without = [], []

    with torch.no_grad():
        for x, interventions, targets in dataloader:
            x = x.to(device)
            targets = targets.to(device)
            noop_interventions = torch.full_like(interventions, -1)

            for int_tensor, mse_container in [
                (interventions, mse_with),
                (noop_interventions, mse_without),
            ]:
                z0_k = ensemble.encode(x)
                z_int_k = torch.stack([
                    apply_intervention(z0_k[k], int_tensor, value=0.0)
                    for k in range(ensemble.k_models)
                ], dim=0)
                latent_k, _ = simulator.rollout(
                    z_int_k, edge_index, steps=steps, log_trajectory=False
                )
                preds = ensemble.decode(latent_k).mean(dim=0)
                mse = criterion(preds, targets).mean(dim=-1).cpu().numpy()
                mse_container.extend(mse)

    m_with = float(np.mean(mse_with))
    m_without = float(np.mean(mse_without))
    return {
        "mse_with_intervention": m_with,
        "mse_no_intervention": m_without,
        "delta": m_with - m_without,
    }


def test_ood_stress(
    ensemble: EnsembleWrapper,
    simulator: RolloutEngine,
    x_sample: torch.Tensor,
    interventions: torch.Tensor,
    edge_index: torch.Tensor,
    device: torch.device,
    scale_factors: List[float] = [1.0, 2.0, 5.0, 10.0],
    steps: int = 3,
) -> Dict[float, float]:
    """
    Scale input expression to simulate OOD extrapolation. Measure resulting uncertainty.

    Returns:
        Dict mapping scale_factor -> mean U_traj.
    """
    ensemble.eval()
    results = {}

    with torch.no_grad():
        for sf in scale_factors:
            x_scaled = x_sample.to(device) * sf
            z0_k = ensemble.encode(x_scaled.unsqueeze(0))
            z_int_k = torch.stack([
                apply_intervention(z0_k[k], interventions.unsqueeze(0), value=0.0)
                for k in range(ensemble.k_models)
            ], dim=0)
            _, trajectory = simulator.rollout(
                z_int_k, edge_index, steps=steps, log_trajectory=True, enforce_stability=False
            )
            u_traj, _ = ensemble.compute_epistemic_uncertainty(trajectory)
            u_mean = u_traj.mean().item()
            results[sf] = u_mean
            print(f"Scale x{sf:.1f} -> Mean U_traj: {u_mean:.4f}")

    return results


def run_all_robustness_tests(
    ensemble: EnsembleWrapper,
    simulator: RolloutEngine,
    dataloader: DataLoader,
    edge_index: torch.Tensor,
    device: torch.device,
) -> None:
    """Run all robustness tests and print a summary table."""
    print("\n" + "=" * 60)
    print("ATLAS Robustness Test Suite")
    print("=" * 60)

    print("\n[1] Input Noise Robustness")
    noise_results = test_input_noise_robustness(ensemble, simulator, dataloader, edge_index, device)

    print("\n[2] Ensemble Consistency")
    consistency = test_ensemble_consistency(ensemble, simulator, dataloader, edge_index, device)
    print(f"  Mean pairwise disagreement: {consistency['mean_pairwise_mse']:.5f}")
    print(f"  Max pairwise disagreement:  {consistency['max_pairwise_mse']:.5f}")

    print("\n[3] No-Op Intervention Sanity Check")
    noop = test_noop_intervention(ensemble, simulator, dataloader, edge_index, device)
    print(f"  MSE with intervention:    {noop['mse_with_intervention']:.4f}")
    print(f"  MSE without intervention: {noop['mse_no_intervention']:.4f}")
    print(f"  Delta:                    {noop['delta']:+.4f}")

    print("\n[DONE] Robustness tests complete.")
