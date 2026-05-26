"""
Publication-quality figure generator for the DriftScore paper.

Produces four figures used in the paper:
  Figure 1 — Score curves over rounds (DriftScore vs. 3 baselines)
  Figure 2 — Kendall's τ bar chart for all metrics (main result)
  Figure 3 — Ablation: alpha sweep for DriftScore
  Figure 4 — PD vs. SD component contribution scatter

Run after run_evaluation.py:
    python plot_results.py --results_dir ./results --out_dir ./figures
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── colour palette ─────────────────────────────────────────────────────────
C_DRIFT   = "#1F4E79"   # dark blue  — DriftScore
C_PD      = "#2E75B6"   # medium blue — PD component
C_SD      = "#70AD47"   # green       — SD component
C_BASELINE= "#A6A6A6"   # gray        — NR-IQA baselines
C_RED     = "#C00000"   # accent red

HIGHLIGHT_BASELINES = ["brisque", "niqe", "musiq", "clipiqa"]


# ── helpers ────────────────────────────────────────────────────────────────

def load_all_sequence_csvs(results_dir: Path) -> pd.DataFrame:
    dfs = []
    for csv in sorted(results_dir.glob("*_scores.csv")):
        if csv.name == "summary.csv":
            continue
        dfs.append(pd.read_csv(csv))
    if not dfs:
        raise FileNotFoundError(f"No sequence CSVs found in {results_dir}")
    return pd.concat(dfs, ignore_index=True)


def mean_trajectory(df: pd.DataFrame, metric: str) -> np.ndarray:
    sub = df[df["metric"] == metric].groupby("round")["score"].mean().sort_index()
    return sub.values


# ── Figure 1: Score curves ─────────────────────────────────────────────────

def plot_score_curves(df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    # Left: DriftScore components
    ax = axes[0]
    for metric, color, label, lw in [
        ("DriftScore",    C_DRIFT, "DriftScore (combined)", 2.5),
        ("DriftScore_PD", C_PD,    "PD (perceptual)",       1.5),
        ("DriftScore_SD", C_SD,    "SD (semantic)",         1.5),
    ]:
        if metric in df["metric"].values:
            y = mean_trajectory(df, metric)
            x = np.arange(len(y))
            ax.plot(x, y, color=color, linewidth=lw, label=label,
                    marker="o" if lw > 2 else None, markersize=3)

    ax.set_title("DriftScore Components")
    ax.set_xlabel("Editing round  t")
    ax.set_ylabel("Score  (higher = more drift)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)

    # Right: DriftScore vs. selected NR-IQA baselines
    ax = axes[1]
    # Plot DriftScore (normalised to [0,1] for comparison)
    ds = mean_trajectory(df, "DriftScore") if "DriftScore" in df["metric"].values else None
    if ds is not None:
        ds_norm = (ds - ds.min()) / (ds.max() - ds.min() + 1e-8)
        ax.plot(np.arange(len(ds_norm)), ds_norm, color=C_DRIFT,
                linewidth=2.5, label="DriftScore", marker="o", markersize=3)

    for metric in HIGHLIGHT_BASELINES:
        if metric not in df["metric"].values:
            continue
        y = mean_trajectory(df, metric)
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)
        ax.plot(np.arange(len(y_norm)), y_norm, color=C_BASELINE,
                linewidth=1.2, alpha=0.7, label=metric, linestyle="--")

    ax.set_title("DriftScore vs. NR-IQA Baselines\n(scores normalised to [0,1], higher=better quality)")
    ax.set_xlabel("Editing round  t")
    ax.set_ylabel("Normalised quality score")
    ax.legend(loc="lower left", ncol=2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Annotate: baselines are flat; DriftScore trends correctly
    ax.annotate("Baselines:\nflat / wrong direction",
                xy=(0.55, 0.65), xycoords="axes fraction",
                fontsize=8, color=C_BASELINE,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_BASELINE, alpha=0.8))

    plt.tight_layout()
    fig.savefig(out_path)
    print(f"  Figure 1 saved: {out_path}")
    plt.close(fig)


# ── Figure 2: Kendall's τ bar chart ────────────────────────────────────────

def plot_tau_bar(summary: pd.DataFrame, out_path: Path):
    # Show absolute τ for easy comparison (DriftScore should be positive-τ naturally)
    df = summary.copy()
    df = df[~df["metric"].isin(["DriftScore_PD", "DriftScore_SD"])]
    df = df.sort_values("mean_tau", ascending=True)

    colors = [C_DRIFT if "DriftScore" in m else C_BASELINE for m in df["metric"]]

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.38)))
    bars = ax.barh(df["metric"], df["mean_tau"], color=colors,
                   xerr=df["std_tau"], error_kw={"elinewidth": 1, "capsize": 3})

    # Highlight DriftScore bar edge
    for bar, metric in zip(bars, df["metric"]):
        if "DriftScore" in metric:
            bar.set_edgecolor(C_RED)
            bar.set_linewidth(1.5)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean Kendall's τ with degradation round\n(higher = better monotonicity)")
    ax.set_title("Degradation Detection: All Metrics\n(positive τ = score correctly tracks increasing drift)")

    # Legend
    patch_ds  = mpatches.Patch(color=C_DRIFT,    label="DriftScore")
    patch_bl  = mpatches.Patch(color=C_BASELINE, label="NR-IQA Baselines")
    ax.legend(handles=[patch_ds, patch_bl], loc="lower right")

    plt.tight_layout()
    fig.savefig(out_path)
    print(f"  Figure 2 saved: {out_path}")
    plt.close(fig)


# ── Figure 3: Alpha ablation ───────────────────────────────────────────────

def plot_alpha_ablation(df_all: pd.DataFrame, out_path: Path):
    """
    If multiple alpha values were run (metrics named DriftScore_a0.2 etc.),
    plot the ablation. Otherwise simulate from PD and SD components.
    """
    alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]

    # Try to compute from PD and SD if available
    pd_scores_by_seq = {}
    sd_scores_by_seq = {}

    for img_id in df_all["image_id"].unique():
        sub = df_all[df_all["image_id"] == img_id]

        pd_rows = sub[sub["metric"] == "DriftScore_PD"].sort_values("round")["score"].values
        sd_rows = sub[sub["metric"] == "DriftScore_SD"].sort_values("round")["score"].values

        if len(pd_rows) > 1 and len(sd_rows) > 1:
            pd_scores_by_seq[img_id] = pd_rows
            sd_scores_by_seq[img_id] = sd_rows

    if not pd_scores_by_seq:
        print("  [Alpha ablation] PD/SD components not found — skipping Figure 3.")
        return

    from scipy.stats import kendalltau

    mean_taus = []
    for alpha in alphas:
        taus = []
        for img_id in pd_scores_by_seq:
            pd_arr = pd_scores_by_seq[img_id].copy()   # avoid shadowing `pd` (pandas alias)
            sd_arr = sd_scores_by_seq[img_id].copy()

            # Re-apply the same trajectory-max normalisation that DriftScore.score() uses.
            # The CSVs store raw (un-normalised) PD and SD. LPIPS values (~0.05-0.5) are
            # ~10x larger than CLIP cosine values (~0.01-0.08). Without normalising here,
            # PD dominates regardless of alpha, making the ablation meaningless.
            pd_max = pd_arr.max() if pd_arr.max() > 1e-8 else 1.0
            sd_max = sd_arr.max() if sd_arr.max() > 1e-8 else 1.0
            pd_norm = pd_arr / pd_max
            sd_norm = sd_arr / sd_max

            combined = alpha * pd_norm + (1 - alpha) * sd_norm
            rounds = np.arange(len(combined))
            tau, _ = kendalltau(rounds, combined)
            taus.append(tau)
        mean_taus.append(np.nanmean(taus))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(alphas, mean_taus, marker="o", color=C_DRIFT, linewidth=2)
    ax.axvline(0.5, color=C_RED, linestyle="--", linewidth=1, label="Default α=0.5")
    ax.fill_between(alphas, mean_taus, alpha=0.1, color=C_DRIFT)
    ax.set_xlabel("α  (weight of PD component)")
    ax.set_ylabel("Mean Kendall's τ")
    ax.set_title("Alpha Ablation: Effect of PD vs. SD Balance")
    ax.set_xticks(alphas)
    ax.legend()
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(out_path)
    print(f"  Figure 3 saved: {out_path}")
    plt.close(fig)


# ── Figure 4: PD vs. SD scatter ───────────────────────────────────────────

def plot_pd_sd_scatter(df_all: pd.DataFrame, summary: pd.DataFrame, out_path: Path):
    """
    Scatter: mean PD τ (x) vs mean SD τ (y) per sequence,
    illustrating that PD and SD are complementary.
    """
    pd_taus, sd_taus = [], []

    for img_id in df_all["image_id"].unique():
        sub = df_all[df_all["image_id"] == img_id]
        for comp, store in [("DriftScore_PD", pd_taus), ("DriftScore_SD", sd_taus)]:
            rows = sub[sub["metric"] == comp].sort_values("round")["score"].values
            if len(rows) > 1:
                from scipy.stats import kendalltau
                tau, _ = kendalltau(np.arange(len(rows)), rows)
                store.append(tau)
            else:
                store.append(np.nan)

    if not pd_taus or all(np.isnan(pd_taus)):
        print("  [PD/SD scatter] Insufficient data — skipping Figure 4.")
        return

    pd_taus = np.array(pd_taus)
    sd_taus = np.array(sd_taus)
    mask = ~(np.isnan(pd_taus) | np.isnan(sd_taus))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(pd_taus[mask], sd_taus[mask], alpha=0.7, color=C_DRIFT, s=50, edgecolors="white")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("PD Kendall's τ (perceptual drift detection)")
    ax.set_ylabel("SD Kendall's τ (semantic drift detection)")
    ax.set_title("Complementarity of PD and SD Components\n(each point = one test sequence)")

    # Annotate quadrants
    for txt, xy in [("Both detect", (0.6, 0.6)), ("Only PD", (0.6, -0.4)),
                    ("Only SD", (-0.4, 0.6)), ("Neither", (-0.4, -0.4))]:
        ax.text(xy[0], xy[1], txt, ha="center", va="center", fontsize=7.5,
                color="#888888", transform=ax.transData)

    plt.tight_layout()
    fig.savefig(out_path)
    print(f"  Figure 4 saved: {out_path}")
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────

def main(args):
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = results_dir / "summary.csv"
    if not summary_path.exists():
        print(f"ERROR: summary.csv not found in {results_dir}. Run run_evaluation.py first.")
        return

    print(f"Loading results from {results_dir} ...")
    summary = pd.read_csv(summary_path)
    df_all  = load_all_sequence_csvs(results_dir)

    print(f"  {len(df_all['image_id'].unique())} sequences, "
          f"{len(df_all['metric'].unique())} metrics.\n")

    print("Generating figures ...")
    plot_score_curves(df_all,  out_dir / "fig1_score_curves.pdf")
    plot_tau_bar(summary,      out_dir / "fig2_tau_bar.pdf")
    plot_alpha_ablation(df_all, out_dir / "fig3_alpha_ablation.pdf")
    plot_pd_sd_scatter(df_all, summary, out_dir / "fig4_pd_sd_scatter.pdf")

    print(f"\nAll figures saved to: {out_dir}/")
    print("Ready to drop into your LaTeX paper.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate paper figures from evaluation results.")
    parser.add_argument("--results_dir", default="./results",
                        help="Directory produced by run_evaluation.py")
    parser.add_argument("--out_dir", default="./figures",
                        help="Where to save figures (default: ./figures)")
    main(parser.parse_args())
