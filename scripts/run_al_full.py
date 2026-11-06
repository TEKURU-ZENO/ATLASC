"""
run_al_full.py
--------------
Executes the full Active Learning experiment across 10 to 20 random seeds
with real multi-round acquisitions and rigorous statistical significance testing:
  - Paired Student's t-test (two-sided and one-sided)
  - Wilcoxon signed-rank test
  - 10,000-sample non-parametric bootstrap 95% confidence intervals
  - Cohen's d effect sizes

Saves per-seed results, aggregated metrics, and statistical significance analysis to:
  experiments/results/al_results.json
"""

import os
import sys
import json
import random
import copy
import argparse
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from scipy import stats

from models.encoders.transcriptomic_encoder import TranscriptomicEncoder
from models.gnn.causal_gnn import CausalGNN
from models.ensemble import EnsembleWrapper
from causal.do_operator import apply_intervention
from utils.data_loader import NormanDataset, get_trrust_graph
from simulation.rollout_engine import RolloutEngine
from active_learning.acquisition import compute_candidate_metrics, rank_and_select


# ---------------------------------------------------------------------------
# Graph & Model Helpers
# ---------------------------------------------------------------------------

def get_node_degree(gene_idx: int, edge_index: torch.Tensor) -> int:
    return int(((edge_index[0] == gene_idx).sum() + (edge_index[1] == gene_idx).sum()).item())


def train_ensemble(
    train_loader: DataLoader,
    edge_index: torch.Tensor,
    num_genes: int,
    num_epochs: int = 2,
    latent_dim: int = 16,
    device: str = "cpu",
    seed_offset: int = 0,
) -> EnsembleWrapper:
    K = 3
    encoders, gnns = [], []
    for k in range(K):
        torch.manual_seed(100 + k + seed_offset)
        enc = TranscriptomicEncoder(num_genes=num_genes, latent_dim=latent_dim).to(device)
        gnn = CausalGNN(
            in_channels=latent_dim, hidden_channels=latent_dim,
            out_channels=1, layers_count=3
        ).to(device)
        sim = RolloutEngine(core_model=gnn, execution_mode="deterministic")
        opt = optim.Adam(list(enc.parameters()) + list(gnn.parameters()), lr=1e-3)
        crit = nn.MSELoss()
        enc.train()
        gnn.train()

        for _ in range(num_epochs):
            for x, ints, target in train_loader:
                x, target = x.to(device), target.to(device)
                opt.zero_grad()
                z0 = enc(x)
                z_int = apply_intervention(z0, ints, value=0.0)
                lat, traj = sim.rollout(
                    z_int, edge_index, steps=2,
                    log_trajectory=True, enforce_stability=True
                )
                pred = gnn.decode(lat)
                loss = crit(pred, target)
                if traj is not None:
                    loss = loss + 1e-2 * torch.mean(torch.norm(traj, dim=-1))
                loss.backward()
                opt.step()

        encoders.append(enc)
        gnns.append(gnn)

    return EnsembleWrapper(encoders, gnns)


def evaluate_holdout(
    ensemble: EnsembleWrapper,
    holdout_loader: DataLoader,
    edge_index: torch.Tensor,
    device: str = "cpu",
) -> float:
    crit = nn.MSELoss()
    total = 0.0
    ensemble.eval()
    with torch.no_grad():
        for x, ints, target in holdout_loader:
            x, target = x.to(device), target.to(device)
            z0_k = ensemble.encode(x)
            preds = []
            for k in range(ensemble.k_models):
                zk = apply_intervention(z0_k[k], ints, value=0.0)
                lat, _ = RolloutEngine(ensemble.gnns[k], "deterministic").rollout(
                    zk, edge_index, steps=2, log_trajectory=False
                )
                preds.append(ensemble.gnns[k].decode(lat))
            total += crit(torch.stack(preds).mean(0), target).item()

    return total / max(len(holdout_loader), 1)


def extract_single_gene_cells(dataset: NormanDataset):
    vocab, cell2genes, valid = set(), {}, []
    for i in range(len(dataset)):
        g1, g2 = dataset.interventions[i][0].item(), dataset.interventions[i][1].item()
        if g1 != -1 and (g2 == -1 or g1 == g2):
            vocab.add(g1)
            cell2genes[i] = [g1]
            valid.append(i)
        elif g1 == -1 and g2 == -1:
            cell2genes[i] = []
            valid.append(i)
    return list(vocab), cell2genes, valid


# ---------------------------------------------------------------------------
# Per-Seed Active Learning Execution
# ---------------------------------------------------------------------------

def run_one_seed(
    seed: int,
    full_ds: NormanDataset,
    edge_index: torch.Tensor,
    device: str = "cpu",
    num_epochs: int = 2,
    al_steps: int = 4,
    k_queries: int = 2,
    max_cells_per_gene: int = 80,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"\n{'='*55}\nActive Learning Seed {seed}\n{'='*55}")

    vocab, cell2genes, valid_cells = extract_single_gene_cells(full_ds)
    torch.manual_seed(seed)
    shuffled = [vocab[i] for i in torch.randperm(len(vocab)).tolist()]

    # Structural partition: 4 holdout, 3 initial training, 12 pool
    holdout = set(shuffled[:4])
    init_train = set(shuffled[4:7])
    pool = set(shuffled[7:])

    # Subsample cells per gene for balanced gradient contribution and stable fast training
    gene_to_cells = {}
    for g in vocab:
        c_list = [c for c in valid_cells if cell2genes.get(c) and cell2genes[c][0] == g]
        rng = random.Random(seed + g)
        rng.shuffle(c_list)
        gene_to_cells[g] = c_list[:max_cells_per_gene]

    held_cells = [c for g in holdout for c in gene_to_cells[g]]
    holdout_loader = DataLoader(Subset(full_ds, held_cells), batch_size=128, shuffle=False)
    print(f"  Holdout genes ({len(holdout)}): {sorted(list(holdout))}")
    print(f"  Initial train genes ({len(init_train)}): {sorted(list(init_train))}")
    print(f"  Candidate pool ({len(pool)}): {sorted(list(pool))}")

    base_cells = [c for c in valid_cells if not cell2genes.get(c)][:50]
    base_batch = next(iter(DataLoader(Subset(full_ds, base_cells), batch_size=50)))[0]

    # Shared initial model for Step 0 across all strategies
    train_cells_0 = [c for g in init_train for c in gene_to_cells[g]]
    loader_0 = DataLoader(Subset(full_ds, train_cells_0), batch_size=64, shuffle=True)
    init_ensemble = train_ensemble(
        loader_0, edge_index, full_ds.num_genes,
        num_epochs=num_epochs, device=device, seed_offset=seed
    )
    init_mse = evaluate_holdout(init_ensemble, holdout_loader, edge_index, device)
    print(f"  Step 0 Initial Holdout MSE: {init_mse:.4f}")

    strategies = ["random", "degree", "utraj_only", "atlas_full"]
    seed_results = {}

    for strategy in strategies:
        print(f"\n  -- Strategy: {strategy.upper()} --")
        cur_train = set(init_train)
        cur_pool = list(pool)
        errors = [init_mse]
        ensemble = init_ensemble

        for step in range(1, al_steps + 1):
            if not cur_pool:
                break
            sim = RolloutEngine(ensemble, "deterministic")
            k_act = min(k_queries, len(cur_pool))

            if strategy == "random":
                rng = random.Random(seed + step * 100)
                selected = rng.sample(cur_pool, k_act)
            elif strategy == "degree":
                selected = sorted(
                    cur_pool, key=lambda g: get_node_degree(g, edge_index), reverse=True
                )[:k_act]
            else:
                ablation = "utraj_only" if strategy == "utraj_only" else "full"
                u_raw, z_reps, hubs = compute_candidate_metrics(
                    cur_pool, ensemble, sim, edge_index,
                    base_batch, apply_intervention, rollout_steps=2, device=device
                )
                selected, _ = rank_and_select(
                    cur_pool, u_raw, z_reps, hubs,
                    k=k_act, lambda1=1.0, lambda2=1.0, ablation_mode=ablation
                )

            for g in selected:
                cur_pool.remove(g)
                cur_train.add(g)

            # Retrain with acquired genes
            train_cells = [c for g in cur_train for c in gene_to_cells[g]]
            loader = DataLoader(Subset(full_ds, train_cells), batch_size=64, shuffle=True)
            ensemble = train_ensemble(
                loader, edge_index, full_ds.num_genes,
                num_epochs=num_epochs, device=device, seed_offset=seed + step * 10
            )
            mse = evaluate_holdout(ensemble, holdout_loader, edge_index, device)
            errors.append(mse)
            print(f"    Step {step}: MSE={mse:.4f} (train_genes={len(cur_train)}, acquired={selected})")

        seed_results[strategy] = errors
        print(f"  {strategy} summary: {[f'{e:.4f}' for e in errors]}")

    return seed_results


# ---------------------------------------------------------------------------
# Rigorous Statistical Significance Engine
# ---------------------------------------------------------------------------

def compute_statistical_tests(all_results, seeds, baseline="random", treatment="atlas_full"):
    """
    Computes rigorous paired statistical significance testing:
      1. Paired Student's t-test (two-sided & one-sided)
      2. Wilcoxon Signed-Rank Test (non-parametric paired test)
      3. Non-parametric Bootstrap Confidence Intervals (10,000 resamples)
      4. Cohen's d effect size
    """
    steps_count = len(all_results[str(seeds[0])][baseline])
    round_stats = {}

    for step in range(steps_count):
        y_base = np.array([all_results[str(s)][baseline][step] for s in seeds])
        y_treat = np.array([all_results[str(s)][treatment][step] for s in seeds])
        diff = y_base - y_treat  # positive = treatment has lower error (improvement)

        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
        sem_diff = float(std_diff / np.sqrt(len(diff))) if len(diff) > 1 else 0.0
        cohen_d = float(mean_diff / std_diff) if std_diff > 1e-8 else 0.0

        # Paired t-test
        if np.all(np.abs(diff) < 1e-10):
            t_stat, p_val_two_sided, p_val_one_sided = 0.0, 1.0, 1.0
        else:
            t_res = stats.ttest_rel(y_base, y_treat)
            t_stat = float(t_res.statistic)
            p_val_two_sided = float(t_res.pvalue)
            p_val_one_sided = float(p_val_two_sided / 2.0 if t_stat > 0 else 1.0 - p_val_two_sided / 2.0)

        # Wilcoxon signed-rank test
        try:
            if not np.all(np.abs(diff) < 1e-10):
                w_res = stats.wilcoxon(y_base, y_treat, alternative="greater")
                w_stat = float(w_res.statistic)
                p_val_wilcoxon = float(w_res.pvalue)
            else:
                w_stat, p_val_wilcoxon = 0.0, 1.0
        except Exception:
            w_stat, p_val_wilcoxon = None, None

        # 10,000 Bootstrap iterations for 95% Confidence Interval
        n = len(diff)
        rng = np.random.RandomState(42)
        boot_diffs = []
        boot_pcts = []
        for _ in range(10000):
            idx = rng.choice(n, size=n, replace=True)
            b_diff = diff[idx]
            b_base = y_base[idx]
            b_diff_mean = np.mean(b_diff)
            boot_diffs.append(b_diff_mean)
            if np.mean(b_base) > 1e-6:
                boot_pcts.append((b_diff_mean / np.mean(b_base)) * 100.0)

        ci_95_diff = [float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))]
        ci_95_pct = [float(np.percentile(boot_pcts, 2.5)), float(np.percentile(boot_pcts, 97.5))] if boot_pcts else [0.0, 0.0]
        pct_improvement = float(mean_diff / np.mean(y_base) * 100.0) if np.mean(y_base) > 0 else 0.0

        round_stats[f"step_{step}"] = {
            "mean_baseline_mse": float(np.mean(y_base)),
            "mean_treatment_mse": float(np.mean(y_treat)),
            "mean_difference": mean_diff,
            "std_difference": std_diff,
            "sem_difference": sem_diff,
            "pct_improvement": pct_improvement,
            "pct_improvement_95_ci": ci_95_pct,
            "difference_95_ci": ci_95_diff,
            "t_statistic": t_stat,
            "p_value_paired_ttest_two_sided": p_val_two_sided,
            "p_value_paired_ttest_one_sided": p_val_one_sided,
            "wilcoxon_statistic": w_stat,
            "p_value_wilcoxon": p_val_wilcoxon,
            "cohens_d": cohen_d,
            "statistically_significant_p05": bool(p_val_one_sided < 0.05),
        }

    return round_stats


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = "experiments/results"
    os.makedirs(results_dir, exist_ok=True)

    t0 = time.time()
    full_ds = NormanDataset(processed_dir="data/processed")
    edge_index = get_trrust_graph(processed_dir="data/processed").to(device)

    seeds = args.seeds
    print(f"\nRunning Multi-Seed AL Experiment across {len(seeds)} seeds: {seeds} | device={device}")

    all_results = {}
    for seed in seeds:
        sr = run_one_seed(
            seed=seed,
            full_ds=full_ds,
            edge_index=edge_index,
            device=device,
            num_epochs=args.epochs,
            al_steps=args.al_steps,
            k_queries=args.k_queries,
            max_cells_per_gene=args.max_cells_per_gene,
        )
        all_results[str(seed)] = sr

    # Aggregate across seeds
    strategies = ["random", "degree", "utraj_only", "atlas_full"]
    aggregated = {}
    for strat in strategies:
        per_seed = [all_results[str(s)][strat] for s in seeds if strat in all_results[str(s)]]
        max_len = max(len(e) for e in per_seed)
        padded = [e + [e[-1]] * (max_len - len(e)) for e in per_seed]
        arr = np.array(padded)
        aggregated[strat] = {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "sem": (arr.std(axis=0) / np.sqrt(len(per_seed))).tolist(),
            "per_seed": {str(seeds[i]): per_seed[i] for i in range(len(per_seed))},
        }

    # Compute Statistical Tests: atlas_full vs random
    stat_atlas_vs_random = compute_statistical_tests(
        all_results, seeds, baseline="random", treatment="atlas_full"
    )
    # Also vs degree and utraj_only
    stat_atlas_vs_degree = compute_statistical_tests(
        all_results, seeds, baseline="degree", treatment="atlas_full"
    )
    stat_atlas_vs_utraj = compute_statistical_tests(
        all_results, seeds, baseline="utraj_only", treatment="atlas_full"
    )

    output = {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "per_seed": all_results,
        "aggregated": aggregated,
        "statistical_tests": {
            "atlas_full_vs_random": stat_atlas_vs_random,
            "atlas_full_vs_degree": stat_atlas_vs_degree,
            "atlas_full_vs_utraj": stat_atlas_vs_utraj,
        },
    }

    out_path = f"{results_dir}/al_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    elapsed = (time.time() - t0) / 60
    print(f"\n[DONE] Full AL experiment across {len(seeds)} seeds completed in {elapsed:.1f} min")
    print(f"  Results saved -> {out_path}")

    # Summary table
    final_stats = stat_atlas_vs_random[f"step_{args.al_steps}"]
    print("\n" + "=" * 65)
    print(f"STATISTICAL VERDICT: ATLAS_FULL vs RANDOM (Final Round, N={len(seeds)} seeds)")
    print("=" * 65)
    print(f"Random Final MSE       : {final_stats['mean_baseline_mse']:.4f}")
    print(f"ATLAS Full Final MSE   : {final_stats['mean_treatment_mse']:.4f}")
    print(f"Mean MSE Reduction     : {final_stats['mean_difference']:.4f} +/- {final_stats['sem_difference']:.4f} (SEM)")
    print(f"95% Bootstrap CI       : [{final_stats['difference_95_ci'][0]:.4f}, {final_stats['difference_95_ci'][1]:.4f}]")
    print(f"Percentage Improvement : {final_stats['pct_improvement']:.2f}% (95% CI: [{final_stats['pct_improvement_95_ci'][0]:.2f}%, {final_stats['pct_improvement_95_ci'][1]:.2f}%])")
    print(f"Paired t-test t-stat   : {final_stats['t_statistic']:.4f}")
    print(f"Paired t-test p-value  : {final_stats['p_value_paired_ttest_one_sided']:.4e} (one-sided)")
    if final_stats['p_value_wilcoxon'] is not None:
        print(f"Wilcoxon p-value       : {final_stats['p_value_wilcoxon']:.4e}")
    print(f"Cohen's d Effect Size  : {final_stats['cohens_d']:.3f}")
    print(f"Statistically Sig (p<.05): {final_stats['statistically_significant_p05']}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Seed Active Learning Experiment")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 999, 7, 21, 33, 77, 88, 101, 202, 303, 404],
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--al-steps", type=int, default=4)
    parser.add_argument("--k-queries", type=int, default=2)
    parser.add_argument("--max-cells-per-gene", type=int, default=80)
    args = parser.parse_args()
    main(args)
