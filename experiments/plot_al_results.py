"""
plot_al_results.py  (Phase 9)
------------------------------
Reads experiments/results/al_results.json and produces a publication-quality
convergence plot: Error vs AL Iterations per strategy (mean ± std across seeds).

Output: experiments/results/al_convergence.png
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_PATH  = "experiments/results/al_results.json"
OUT_PATH = "experiments/results/al_convergence.png"

STRATEGY_CONFIG = {
    "random":     {"label": "Random",      "color": "#94a3b8", "ls": "--"},
    "degree":     {"label": "Degree-Hub",  "color": "#f59e0b", "ls": "-."},
    "utraj_only": {"label": "U_traj Only", "color": "#38bdf8", "ls": ":"},
    "atlas_full": {"label": "ATLAS Full",  "color": "#22c55e", "ls": "-"},
}


def main():
    if not os.path.exists(IN_PATH):
        print(f"ERROR: {IN_PATH} not found. Run scripts/run_al_full.py first.")
        sys.exit(1)

    with open(IN_PATH) as f:
        data = json.load(f)

    agg = data.get("aggregated", {})
    seeds = data.get("seeds", [])

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    for strat, cfg in STRATEGY_CONFIG.items():
        if strat not in agg:
            continue
        means = np.array(agg[strat]["mean"])
        stds  = np.array(agg[strat]["std"])
        x = np.arange(len(means))

        ax.plot(x, means, label=cfg["label"],
                color=cfg["color"], ls=cfg["ls"], lw=2.5, marker="o",
                markersize=7, markeredgecolor="#0f172a", markeredgewidth=1.5)
        ax.fill_between(x, means - stds, means + stds,
                        color=cfg["color"], alpha=0.15)

    ax.set_xlabel("Active Learning Iterations", color="#94a3b8", fontsize=11)
    ax.set_ylabel("Holdout MSE", color="#94a3b8", fontsize=11)
    ax.set_title(
        f"Active Learning Convergence  (seeds={seeds})",
        color="#f1f5f9", fontsize=13, fontweight="bold"
    )
    ax.tick_params(colors="#64748b")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#334155"); ax.spines["bottom"].set_color("#334155")

    legend = ax.legend(frameon=True, fontsize=10, loc="upper right")
    legend.get_frame().set_facecolor("#1e293b")
    legend.get_frame().set_edgecolor("#334155")
    for text in legend.get_texts():
        text.set_color("#e2e8f0")

    # Annotation: ATLAS convergence diff
    if "atlas_full" in agg and "random" in agg:
        final_atlas  = agg["atlas_full"]["mean"][-1]
        final_random = agg["random"]["mean"][-1]
        if final_atlas < final_random:
            delta = final_random - final_atlas
            ax.annotate(
                f"ATLAS saves {delta:.4f} MSE\nvs Random at final step",
                xy=(len(agg["atlas_full"]["mean"]) - 1, final_atlas),
                xytext=(len(agg["atlas_full"]["mean"]) - 2, final_atlas + delta * 0.6),
                color="#86efac", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#86efac", lw=1.2)
            )

    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"✓ Convergence plot → {OUT_PATH}")


if __name__ == "__main__":
    main()
