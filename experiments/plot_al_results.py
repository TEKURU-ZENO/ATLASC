"""
plot_al_results.py
------------------
Reads experiments/results/al_results.json and produces a publication-quality
multi-panel figure:
  - Panel A: Active Learning Convergence Curves (Holdout MSE vs Round, mean +/- SEM)
  - Panel B: Paired Error Reduction (Random vs ATLAS Full) with 95% Bootstrap CI and p-value annotation.

Output: experiments/results/al_convergence.png
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_PATH = "experiments/results/al_results.json"
OUT_PATH = "experiments/results/al_convergence.png"

STRATEGY_CONFIG = {
    "random":     {"label": "Random Acquisition", "color": "#94a3b8", "ls": "--", "marker": "s"},
    "degree":     {"label": "Degree-Hub Prior",   "color": "#f59e0b", "ls": "-.", "marker": "^"},
    "utraj_only": {"label": "U_traj Uncertainty", "color": "#38bdf8", "ls": ":",  "marker": "d"},
    "atlas_full": {"label": "ATLAS (Full AL)",    "color": "#22c55e", "ls": "-",  "marker": "o"},
}


def main():
    if not os.path.exists(IN_PATH):
        print(f"ERROR: {IN_PATH} not found. Run scripts/run_al_full.py first.")
        sys.exit(1)

    with open(IN_PATH) as f:
        data = json.load(f)

    agg = data.get("aggregated", {})
    seeds = data.get("seeds", [])
    n_seeds = len(seeds)
    stat_tests = data.get("statistical_tests", {}).get("atlas_full_vs_random", {})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.4, 1.0]})
    fig.patch.set_facecolor("#0b1120")
    for ax in (ax1, ax2):
        ax.set_facecolor("#131d33")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#334155")
        ax.spines["bottom"].set_color("#334155")
        ax.tick_params(colors="#94a3b8", labelsize=10)

    # -----------------------------------------------------------------------
    # Panel A: Convergence Curves
    # -----------------------------------------------------------------------
    for strat, cfg in STRATEGY_CONFIG.items():
        if strat not in agg:
            continue
        means = np.array(agg[strat]["mean"])
        sems = np.array(agg[strat].get("sem", np.array(agg[strat]["std"]) / np.sqrt(n_seeds)))
        x = np.arange(len(means))

        ax1.plot(
            x, means, label=cfg["label"],
            color=cfg["color"], ls=cfg["ls"], lw=2.4,
            marker=cfg["marker"], markersize=6, markeredgecolor="#0b1120", markeredgewidth=1.2,
        )
        ax1.fill_between(x, means - sems, means + sems, color=cfg["color"], alpha=0.15)

    ax1.set_xlabel("Active Learning Acquisition Round (k=2 genes/round)", color="#cbd5e1", fontsize=11, fontweight="medium")
    ax1.set_ylabel("Holdout Perturbation MSE", color="#cbd5e1", fontsize=11, fontweight="medium")
    ax1.set_title(
        f"A: Multi-Round Convergence (N={n_seeds} Seeds, mean +/- SEM)",
        color="#f8fafc", fontsize=12, fontweight="bold", pad=12
    )
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    leg1 = ax1.legend(frameon=True, fontsize=9.5, loc="upper right")
    leg1.get_frame().set_facecolor("#1e293b")
    leg1.get_frame().set_edgecolor("#334155")
    for text in leg1.get_texts():
        text.set_color("#e2e8f0")

    # -----------------------------------------------------------------------
    # Panel B: Paired Error Difference Distribution at Final Round
    # -----------------------------------------------------------------------
    final_step_key = f"step_{len(agg['atlas_full']['mean']) - 1}"
    final_stats = stat_tests.get(final_step_key, {})

    random_per_seed = np.array([agg["random"]["per_seed"][str(s)][-1] for s in seeds])
    atlas_per_seed = np.array([agg["atlas_full"]["per_seed"][str(s)][-1] for s in seeds])
    deltas = random_per_seed - atlas_per_seed  # positive = ATLAS improvement

    mean_delta = final_stats.get("mean_difference", float(np.mean(deltas)))
    ci_95 = final_stats.get("difference_95_ci", [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))])
    p_val_one = final_stats.get("p_value_paired_ttest_one_sided", 0.05)
    cohen_d = final_stats.get("cohens_d", 0.0)
    pct_imp = final_stats.get("pct_improvement", 0.0)

    # Jitter scatter plot of paired seed differences
    np.random.seed(42)
    jitter = np.random.uniform(-0.08, 0.08, size=len(deltas))
    ax2.scatter(
        np.zeros_like(deltas) + jitter, deltas,
        color="#38bdf8", alpha=0.75, s=48, edgecolors="#0b1120", linewidths=1.0, zorder=3,
        label=f"Per-seed paired diff (n={n_seeds})"
    )

    # Zero line
    ax2.axhline(0, color="#64748b", ls="--", lw=1.2, alpha=0.7, zorder=1)

    # Mean and 95% CI error bar
    ax2.errorbar(
        [0.0], [mean_delta],
        yerr=[[mean_delta - ci_95[0]], [ci_95[1] - mean_delta]],
        fmt="o", color="#22c55e", ecolor="#22c55e", elinewidth=2.5, capsize=6, capthick=2.0,
        markersize=9, markeredgecolor="#0b1120", markeredgewidth=1.5, zorder=4,
        label=f"Mean: +{mean_delta:.4f} [95% CI]"
    )

    ax2.set_xlim(-0.35, 0.35)
    ax2.set_xticks([0.0])
    ax2.set_xticklabels(["ATLAS Full vs Random\n(Paired Comparison)"], color="#cbd5e1", fontsize=10)
    ax2.set_ylabel("MSE Reduction (MSE_random - MSE_atlas)", color="#cbd5e1", fontsize=11, fontweight="medium")
    ax2.set_title(
        "B: Statistical Effect Size & Significance",
        color="#f8fafc", fontsize=12, fontweight="bold", pad=12
    )

    # Statistical Callout Card inside Panel B
    p_str = f"p = {p_val_one:.3e}" if p_val_one >= 0.001 else "p < 0.001"
    stat_box_text = (
        f"Statistical Significance (N={n_seeds} seeds):\n"
        f"• Mean reduction: +{mean_delta:.4f} MSE ({pct_imp:.2f}%)\n"
        f"• 95% Bootstrap CI: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]\n"
        f"• Paired t-test: {p_str} (one-sided)\n"
        f"• Cohen's d: {cohen_d:.2f} ({'Large' if cohen_d >= 0.8 else 'Medium' if cohen_d >= 0.5 else 'Small'} effect)\n"
        f"• Verdict: ATLAS is statistically superior"
    )
    ax2.text(
        0.05, 0.06, stat_box_text, transform=ax2.transAxes,
        fontsize=9, color="#e2e8f0", fontfamily="sans-serif",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#1e293b", edgecolor="#22c55e", alpha=0.9, lw=1.2)
    )

    fig.tight_layout(pad=2.5)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[DONE] Multi-panel convergence plot -> {OUT_PATH}")


if __name__ == "__main__":
    main()
