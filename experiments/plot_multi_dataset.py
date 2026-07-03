"""
plot_multi_dataset.py  (Phase 8)
---------------------------------
Reads multi_dataset_metrics.json (written by cross_dataset_transfer.py) and
generates a publication-quality comparison figure:

  Left panel  – MSE: Norman vs Dixit vs Replogle
  Right panel – U_traj: Norman vs Dixit vs Replogle

Output: experiments/results/multi_dataset_comparison.png
"""

import os
import json
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

METRICS_PATH = "experiments/results/multi_dataset_metrics.json"
OUT_PATH     = "experiments/results/multi_dataset_comparison.png"

COLORS = {
    "Norman":   "#22c55e",   # green  – in-domain
    "Dixit":    "#f59e0b",   # amber  – mild OOD
    "Replogle": "#ef4444",   # red    – large OOD
}
LABELS = {
    "Norman":   "Norman\n(in-domain)",
    "Dixit":    "Dixit\n(mild OOD)",
    "Replogle": "Replogle\n(large OOD)",
}


def load_metrics(path: str) -> dict:
    if not os.path.exists(path):
        print(f"ERROR: Metrics file not found at {path}")
        print("Run experiments/cross_dataset_transfer.py first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def build_figure(metrics: dict):
    # Determine which datasets are available
    dataset_keys = ["norman", "dixit"]
    if "replogle" in metrics:
        dataset_keys.append("replogle")
    cap_map = {"norman": "Norman", "dixit": "Dixit", "replogle": "Replogle"}

    mse_vals = [metrics[k]["mse_mean"] for k in dataset_keys]
    u_vals   = [metrics[k]["u_mean"]   for k in dataset_keys]
    names    = [cap_map[k] for k in dataset_keys]
    colors   = [COLORS[n] for n in names]
    x        = np.arange(len(names))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_facecolor("#0f172a")
    for ax in axes:
        ax.set_facecolor("#1e293b")

    bar_kw = dict(width=0.5, edgecolor="#334155", linewidth=1.2)

    # ── Left: MSE ────────────────────────────────────────────────────────────
    ax = axes[0]
    bars = ax.bar(x, mse_vals, color=colors, **bar_kw)
    for bar, val in zip(bars, mse_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9,
                color="#e2e8f0", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([LABELS[n] for n in names],
                                          color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Mean MSE (masked to intersection genes)", color="#94a3b8", fontsize=10)
    ax.set_title("Prediction Error by Dataset", color="#f1f5f9", fontsize=12, fontweight="bold")
    ax.tick_params(colors="#64748b")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#334155"); ax.spines["bottom"].set_color("#334155")
    ax.yaxis.label.set_color("#94a3b8")
    # Annotation: expected direction
    if len(mse_vals) > 1 and mse_vals[-1] > mse_vals[0]:
        ax.annotate("↑ error increases OOD\n(expected)", xy=(x[-1], mse_vals[-1]),
                    xytext=(x[-1] - 0.6, mse_vals[-1] * 0.7),
                    color="#86efac", fontsize=8,
                    arrowprops=dict(arrowstyle="->", color="#86efac", lw=1))

    # ── Right: U_traj ─────────────────────────────────────────────────────────
    ax = axes[1]
    bars = ax.bar(x, u_vals, color=colors, **bar_kw)
    for bar, val in zip(bars, u_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(u_vals) * 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9,
                color="#e2e8f0", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([LABELS[n] for n in names],
                                          color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Mean U_traj (epistemic uncertainty)", color="#94a3b8", fontsize=10)
    ax.set_title("Epistemic Uncertainty by Dataset", color="#f1f5f9", fontsize=12, fontweight="bold")
    ax.tick_params(colors="#64748b")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#334155"); ax.spines["bottom"].set_color("#334155")

    # ── Legend + meta ─────────────────────────────────────────────────────────
    legend_patches = [mpatches.Patch(color=COLORS[n], label=n) for n in names]
    fig.legend(handles=legend_patches, loc="lower center", ncol=len(names),
               frameon=False, fontsize=9, labelcolor="#cbd5e1",
               bbox_to_anchor=(0.5, -0.02))

    # Corr annotations
    info_lines = []
    for k in dataset_keys:
        m = metrics[k]
        r = m.get("corr", float("nan")); p = m.get("pval", float("nan"))
        n = cap_map[k]
        kl_str = f" | KL={m['kl_from_norman']:.1f}" if "kl_from_norman" in m else ""
        info_lines.append(f"{n}: corr(U,MSE)={r:+.4f} p={p:.2e}{kl_str}")

    fig.text(0.5, -0.08, "\n".join(info_lines),
             ha="center", va="top", fontsize=8, color="#64748b",
             fontfamily="monospace")

    fig.suptitle("ATLAS — Multi-Dataset Generalisation Analysis",
                 color="#f8fafc", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"[DONE] Saved -> {OUT_PATH}")
    return OUT_PATH


if __name__ == "__main__":
    metrics = load_metrics(METRICS_PATH)
    path = build_figure(metrics)
    print(f"\nDatasets in JSON: {list(metrics.keys())}")
    for ds, m in metrics.items():
        print(f"  {ds}: MSE={m['mse_mean']:.4f}  U={m['u_mean']:.4f}  corr={m.get('corr', 'N/A'):.4f}")
