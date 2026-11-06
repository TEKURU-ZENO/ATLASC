"""
run_al_full.py  (Phase 9)
--------------------------
Runs the full Active Learning experiment across random seeds with real multi-round acquisitions.
Saves per-seed and aggregated results to experiments/results/al_results.json.
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

from models.encoders.transcriptomic_encoder import TranscriptomicEncoder
from models.gnn.causal_gnn import CausalGNN
from models.ensemble import EnsembleWrapper
from causal.do_operator import apply_intervention
from utils.data_loader import NormanDataset, get_trrust_graph
from simulation.rollout_engine import RolloutEngine
from active_learning.acquisition import compute_candidate_metrics, rank_and_select


# -- Helpers ------------------------------------------------------------------

def get_node_degree(gene_idx, edge_index):
    return ((edge_index[0] == gene_idx).sum() + (edge_index[1] == gene_idx).sum()).item()


def train_ensemble(train_loader, edge_index, num_genes, num_epochs=2,
                   latent_dim=16, device="cpu", seed_offset=0):
    K = 3
    encoders, gnns = [], []
    for k in range(K):
        torch.manual_seed(100 + k + seed_offset)
        enc = TranscriptomicEncoder(num_genes=num_genes, latent_dim=latent_dim).to(device)
        gnn = CausalGNN(in_channels=latent_dim, hidden_channels=latent_dim,
                        out_channels=1, layers_count=3).to(device)
        sim = RolloutEngine(core_model=gnn, execution_mode="deterministic")
        opt = optim.Adam(list(enc.parameters()) + list(gnn.parameters()), lr=1e-3)
        crit = nn.MSELoss()
        enc.train(); gnn.train()
        for _ in range(num_epochs):
            for x, ints, target in train_loader:
                x, target = x.to(device), target.to(device)
                opt.zero_grad()
                z0 = enc(x)
                z_int = apply_intervention(z0, ints, value=0.0)
                lat, traj = sim.rollout(z_int, edge_index, steps=2,
                                        log_trajectory=True, enforce_stability=True)
                pred = gnn.decode(lat)
                loss = crit(pred, target)
                if traj is not None:
                    loss = loss + 1e-2 * torch.mean(torch.norm(traj, dim=-1))
                loss.backward(); opt.step()
        encoders.append(enc); gnns.append(gnn)
    return EnsembleWrapper(encoders, gnns)


def evaluate_holdout(ensemble, holdout_loader, edge_index, device):
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
                    zk, edge_index, steps=2, log_trajectory=False)
                preds.append(ensemble.gnns[k].decode(lat))
            total += crit(torch.stack(preds).mean(0), target).item()
    return total / max(len(holdout_loader), 1)


def extract_single_gene_cells(dataset):
    vocab, cell2genes, valid = set(), {}, []
    for i in range(len(dataset)):
        g1, g2 = dataset.interventions[i][0].item(), dataset.interventions[i][1].item()
        if g1 != -1 and (g2 == -1 or g1 == g2):
            vocab.add(g1); cell2genes[i] = [g1]; valid.append(i)
        elif g1 == -1 and g2 == -1:
            cell2genes[i] = []; valid.append(i)
    return list(vocab), cell2genes, valid


def run_one_seed(seed, full_ds, edge_index, device, num_epochs=2,
                 al_steps=4, k_queries=2, max_cells_per_gene=80,
                 results_dir="experiments/results"):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    print(f"\n{'='*55}\nSeed {seed}\n{'='*55}")

    vocab, cell2genes, valid_cells = extract_single_gene_cells(full_ds)
    torch.manual_seed(seed)
    shuffled = [vocab[i] for i in torch.randperm(len(vocab)).tolist()]

    # Structural partition: 4 holdout, 3 init_train, 12 pool
    holdout    = set(shuffled[:4])
    init_train = set(shuffled[4:7])
    pool       = set(shuffled[7:])

    # Subsample cells per gene for balanced gradient contribution and stable fast training
    gene_to_cells = {}
    for g in vocab:
        c_list = [c for c in valid_cells if cell2genes.get(c) and cell2genes[c][0] == g]
        rng = random.Random(seed + g)
        rng.shuffle(c_list)
        gene_to_cells[g] = c_list[:max_cells_per_gene]

    held_cells = [c for g in holdout for c in gene_to_cells[g]]
    holdout_loader = DataLoader(Subset(full_ds, held_cells), batch_size=128, shuffle=False)
    print(f"  Holdout: {len(holdout)} genes, {len(held_cells)} cells")
    print(f"  Init train: {len(init_train)} genes, Pool: {len(pool)} genes")

    base_cells = [c for c in valid_cells if not cell2genes.get(c)][:50]
    base_batch = next(iter(DataLoader(Subset(full_ds, base_cells), batch_size=50)))[0]

    strategies = ["random", "degree", "utraj_only", "atlas_full"]
    seed_results = {}

    for strategy in strategies:
        print(f"\n  -- Strategy: {strategy.upper()} --")
        cur_train = set(init_train); cur_pool = list(pool)
        errors = []

        for step in range(al_steps + 1):
            train_cells = [c for g in cur_train for c in gene_to_cells[g]]
            loader = DataLoader(Subset(full_ds, train_cells), batch_size=64, shuffle=True)
            ensemble = train_ensemble(loader, edge_index, full_ds.num_genes,
                                     num_epochs=num_epochs, device=device,
                                     seed_offset=seed)
            sim = RolloutEngine(ensemble, "deterministic")
            mse = evaluate_holdout(ensemble, holdout_loader, edge_index, device)
            errors.append(mse)
            print(f"    Step {step}: MSE={mse:.4f} (train_genes={len(cur_train)})")

            if step == al_steps or not cur_pool:
                break

            k_act = min(k_queries, len(cur_pool))
            if strategy == "random":
                rng = random.Random(seed + step * 100)
                selected = rng.sample(cur_pool, k_act)
            elif strategy == "degree":
                selected = sorted(cur_pool, key=lambda g: get_node_degree(g, edge_index),
                                  reverse=True)[:k_act]
            else:
                ablation = "utraj_only" if strategy == "utraj_only" else "full"
                u_raw, z_reps, hubs = compute_candidate_metrics(
                    cur_pool, ensemble, sim, edge_index,
                    base_batch, apply_intervention, rollout_steps=2, device=device)
                selected, _ = rank_and_select(cur_pool, u_raw, z_reps, hubs,
                                              k=k_act, lambda1=1.0, lambda2=1.0,
                                              ablation_mode=ablation)
            print(f"    Acquired {k_act} genes: {selected}")
            for g in selected:
                cur_pool.remove(g); cur_train.add(g)

        seed_results[strategy] = errors
        print(f"  {strategy}: {[f'{e:.4f}' for e in errors]}")

    return seed_results


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Full AL Experiment | device={device}")
    results_dir = "experiments/results"
    os.makedirs(results_dir, exist_ok=True)

    t0 = time.time()
    full_ds    = NormanDataset(processed_dir="data/processed")
    edge_index = get_trrust_graph(processed_dir="data/processed").to(device)

    seeds = args.seeds
    all_results = {}

    for seed in seeds:
        sr = run_one_seed(seed, full_ds, edge_index, device,
                          num_epochs=args.epochs,
                          al_steps=args.al_steps,
                          k_queries=args.k_queries,
                          max_cells_per_gene=args.max_cells_per_gene,
                          results_dir=results_dir)
        all_results[str(seed)] = sr

    # Aggregate across seeds
    strategies = list(next(iter(all_results.values())).keys())
    aggregated = {}
    for strat in strategies:
        per_seed = [all_results[str(s)][strat] for s in seeds if strat in all_results[str(s)]]
        max_len = max(len(e) for e in per_seed)
        # Pad shorter runs
        padded = [e + [e[-1]] * (max_len - len(e)) for e in per_seed]
        arr = np.array(padded)
        aggregated[strat] = {
            "mean": arr.mean(axis=0).tolist(),
            "std":  arr.std(axis=0).tolist(),
            "per_seed": {str(seeds[i]): per_seed[i] for i in range(len(per_seed))}
        }

    output = {"seeds": seeds, "per_seed": all_results, "aggregated": aggregated}
    out_path = f"{results_dir}/al_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    elapsed = (time.time() - t0) / 60
    print(f"\n[DONE] Full AL experiment done in {elapsed:.1f} min")
    print(f"  Results → {out_path}")

    # Quick summary
    print("\n-- Aggregated Convergence --")
    for strat, v in aggregated.items():
        means = [f"{m:.4f}" for m in v["mean"]]
        stds  = [f"{s:.4f}" for s in v["std"]]
        print(f"  {strat:<12}: {means}  (±{stds})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Active Learning Experiment")
    parser.add_argument("--seeds",              nargs="+", type=int, default=[42, 123, 999])
    parser.add_argument("--epochs",             type=int,  default=2)
    parser.add_argument("--al-steps",           type=int,  default=4)
    parser.add_argument("--k-queries",          type=int,  default=2)
    parser.add_argument("--max-cells-per-gene", type=int,  default=80)
    args = parser.parse_args()
    main(args)
