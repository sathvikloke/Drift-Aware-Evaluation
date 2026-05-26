"""
Main evaluation script for the DriftScore paper.

Runs DriftScore and all NR-IQA baselines on a dataset of degradation
trajectories, then computes:
  - Kendall's τ  (monotonicity with degradation round)
  - Spearman's ρ (rank correlation with round)
  - Inflection detection accuracy (within ±1 round)

Results are saved to CSV and printed as a ranked table.

Quick start (no external images needed):
    python run_evaluation.py --make_samples --n_images 10 --n_rounds 15

Full run:
    python run_evaluation.py --source_dir ./my_images --n_rounds 20
"""

from __future__ import annotations
import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

# Local modules (model imports are deferred inside run() so helpers are usable standalone)
from generate_data import build_dataset, make_sample_images, load_dataset


# ── evaluation helpers ─────────────────────────────────────────────────────

def kendall_tau_with_rounds(scores: List[float]) -> float:
    """
    Kendall's τ between metric scores and degradation round index.

    Convention used throughout this codebase:
      - DriftScore (higher = more drift):   positive τ is good (score rises with rounds).
      - NR-IQA baselines passed here are already inverted (higher = more degraded),
        so positive τ is also good for them.

    Returns raw τ (not |τ|). The caller handles sign interpretation.
    """
    rounds = np.arange(len(scores))
    tau, _ = kendalltau(rounds, scores)
    return float(tau)


def spearman_with_rounds(scores: List[float]) -> float:
    """Spearman's ρ between metric scores and degradation round."""
    rounds = np.arange(len(scores))
    rho, _ = spearmanr(rounds, scores)
    return float(rho)


def inflection_detection_accuracy(
    scores: List[float], true_inflection: int, window: int = 1
) -> bool:
    """
    Check whether the metric first crosses a degradation threshold
    within `window` rounds of the true inflection point.

    Inflection is defined as the first round where the score deviates
    more than 1 std dev from the round-0 score.

    Args:
        scores:           metric scores for rounds 0..N
        true_inflection:  ground-truth round where degradation begins
        window:           tolerance in rounds (default ±1)
    """
    if len(scores) < 2:
        return False
    baseline = scores[0]
    std = np.std(scores)
    threshold = baseline + std if scores[-1] > scores[0] else baseline - std

    for t, s in enumerate(scores):
        if (scores[-1] > scores[0] and s > threshold) or \
           (scores[-1] < scores[0] and s < threshold):
            return abs(t - true_inflection) <= window

    return False


# ── per-trajectory evaluation ──────────────────────────────────────────────

def evaluate_trajectory(
    trajectory_frames: List,
    prompt: str,
    drift_scorer: DriftScore,
    baselines: NRIQABaselines,
    true_inflection: int = 1,
) -> Dict[str, Dict]:
    """
    Evaluate one trajectory with all metrics.

    Returns:
        dict keyed by metric name, each value is:
        { "scores": [...], "tau": float, "rho": float, "inflection_hit": bool }
    """
    results = {}

    # ── DriftScore ──────────────────────────────────────────────────────
    combined, pd_scores, sd_scores = drift_scorer.score_trajectory(
        trajectory_frames, prompt=prompt
    )
    # Prepend round-0 score of 0 (no drift at origin)
    drift_full = [0.0] + combined
    pd_full    = [0.0] + pd_scores
    sd_full    = [0.0] + sd_scores

    results["DriftScore"] = {
        "scores": drift_full,
        "tau":    kendall_tau_with_rounds(drift_full),
        "rho":    spearman_with_rounds(drift_full),
        "inflection_hit": inflection_detection_accuracy(drift_full, true_inflection),
    }
    results["DriftScore_PD"] = {
        "scores": pd_full,
        "tau":    kendall_tau_with_rounds(pd_full),
        "rho":    spearman_with_rounds(pd_full),
        "inflection_hit": inflection_detection_accuracy(pd_full, true_inflection),
    }
    results["DriftScore_SD"] = {
        "scores": sd_full,
        "tau":    kendall_tau_with_rounds(sd_full),
        "rho":    spearman_with_rounds(sd_full),
        "inflection_hit": inflection_detection_accuracy(sd_full, true_inflection),
    }

    # ── NR-IQA baselines ────────────────────────────────────────────────
    baseline_traj = baselines.score_trajectory(trajectory_frames)
    for metric_name, scores in baseline_traj.items():
        # Baselines are higher=better; for τ/ρ we want to detect
        # that scores go DOWN as degradation increases.
        # We negate so that "higher = more degraded" (matches DriftScore convention).
        inv_scores = [-s for s in scores]
        results[metric_name] = {
            "scores": scores,           # raw (higher=better) for plotting
            "scores_inv": inv_scores,   # inverted for τ/ρ computation
            "tau":    kendall_tau_with_rounds(inv_scores),
            "rho":    spearman_with_rounds(inv_scores),
            "inflection_hit": inflection_detection_accuracy(inv_scores, true_inflection),
        }

    return results


# ── aggregate over dataset ─────────────────────────────────────────────────

def aggregate_results(per_sequence: List[Dict]) -> pd.DataFrame:
    """
    Aggregate per-trajectory results into a summary DataFrame.

    Columns: metric, mean_tau, mean_rho, inflection_acc, n_sequences
    """
    if not per_sequence:
        return pd.DataFrame()

    metric_names = list(per_sequence[0].keys())
    rows = []

    for metric in metric_names:
        taus = [seq[metric]["tau"] for seq in per_sequence if metric in seq]
        rhos = [seq[metric]["rho"] for seq in per_sequence if metric in seq]
        hits = [seq[metric]["inflection_hit"] for seq in per_sequence if metric in seq]

        rows.append({
            "metric":         metric,
            "mean_tau":       float(np.nanmean(taus)),
            "std_tau":        float(np.nanstd(taus)),
            "mean_rho":       float(np.nanmean(rhos)),
            "std_rho":        float(np.nanstd(rhos)),
            "inflection_acc": float(np.mean(hits)) if hits else float("nan"),
            "n_sequences":    len(taus),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("mean_tau", ascending=False).reset_index(drop=True)
    return df


# ── main ───────────────────────────────────────────────────────────────────

def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data
    if args.load_existing:
        print(f"Loading existing dataset from {args.data_dir} ...")
        dataset = load_dataset(args.data_dir)
    else:
        data_dir = out_dir / "data"
        if args.make_samples:
            sample_dir = out_dir / "sample_images"
            print("Generating synthetic sample images ...")
            make_sample_images(str(sample_dir), n=args.n_images or 10)
            args.source_dir = str(sample_dir)

        print(f"\nGenerating degradation sequences (mode={args.mode}) ...")
        dataset = build_dataset(
            source_dir=args.source_dir,
            out_dir=str(data_dir),
            n_rounds=args.n_rounds,
            mode=args.mode,
            n_images=args.n_images,
            seed=args.seed,
        )

    if not dataset:
        print("ERROR: No sequences generated. Check --source_dir.")
        sys.exit(1)

    print(f"\n{len(dataset)} sequences loaded, {dataset[0]['n_rounds']} rounds each.\n")

    # 2. Load models (deferred import so evaluation helpers work without torch)
    from drift_score import DriftScore
    try:
        import torch
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        default_device = "cpu"
    device = args.device or default_device
    print(f"Using device: {device}\n")

    drift_scorer = DriftScore(alpha=args.alpha, device=device)

    if args.skip_baselines:
        baselines = None
        print("[Baselines] Skipped (--skip_baselines).\n")
    else:
        from baselines import NRIQABaselines
        baselines = NRIQABaselines(device=device)

    # 3. Evaluate
    print("Running evaluation ...\n")
    per_sequence_results = []

    for i, item in enumerate(dataset):
        print(f"  Sequence {i+1}/{len(dataset)}: {item['image_id']}")
        frames = item["frames"]
        prompt = item.get("prompt", None)

        if baselines is not None:
            result = evaluate_trajectory(
                frames, prompt, drift_scorer, baselines,
                true_inflection=args.true_inflection,
            )
        else:
            # DriftScore only
            combined, pd_s, sd_s = drift_scorer.score_trajectory(frames, prompt=prompt)
            drift_full = [0.0] + combined
            pd_full    = [0.0] + pd_s
            sd_full    = [0.0] + sd_s
            result = {
                "DriftScore": {
                    "scores": drift_full,
                    "tau":    kendall_tau_with_rounds(drift_full),
                    "rho":    spearman_with_rounds(drift_full),
                    "inflection_hit": inflection_detection_accuracy(
                        drift_full, args.true_inflection),
                },
                "DriftScore_PD": {
                    "scores": pd_full,
                    "tau":    kendall_tau_with_rounds(pd_full),
                    "rho":    spearman_with_rounds(pd_full),
                    "inflection_hit": inflection_detection_accuracy(
                        pd_full, args.true_inflection),
                },
                "DriftScore_SD": {
                    "scores": sd_full,
                    "tau":    kendall_tau_with_rounds(sd_full),
                    "rho":    spearman_with_rounds(sd_full),
                    "inflection_hit": inflection_detection_accuracy(
                        sd_full, args.true_inflection),
                },
            }

        per_sequence_results.append(result)

        # Save per-sequence scores to CSV for later plotting
        rows = []
        for metric, data in result.items():
            for t, s in enumerate(data["scores"]):
                rows.append({"image_id": item["image_id"], "metric": metric,
                              "round": t, "score": s})
        seq_df = pd.DataFrame(rows)
        seq_df.to_csv(out_dir / f"{item['image_id']}_scores.csv", index=False)

    # 4. Aggregate
    summary = aggregate_results(per_sequence_results)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    # 5. Print table
    print("\n" + "="*75)
    print("RESULTS: Mean Kendall's τ and Spearman's ρ (higher = better monotonicity)")
    print("="*75)
    print(f"{'Metric':<20} {'τ (mean)':>10} {'τ (std)':>9} {'ρ (mean)':>10} {'Inflect.Acc':>12}")
    print("-"*75)
    for _, row in summary.iterrows():
        marker = " ◄" if row["metric"].startswith("DriftScore") and "_" not in row["metric"][10:] else ""
        print(
            f"{row['metric']:<20} "
            f"{row['mean_tau']:>+10.4f} "
            f"{row['std_tau']:>9.4f} "
            f"{row['mean_rho']:>+10.4f} "
            f"{row['inflection_acc']:>12.2%}"
            f"{marker}"
        )
    print("="*75)
    print(f"\nSummary saved to: {summary_path}")
    print(f"Per-sequence CSVs saved to: {out_dir}/")
    print("\nNext step: run  python plot_results.py --results_dir", out_dir)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DriftScore evaluation.")
    # Data args
    parser.add_argument("--source_dir", default="./sample_images")
    parser.add_argument("--data_dir", default="./data",
                        help="Used with --load_existing to reload saved sequences")
    parser.add_argument("--out_dir", default="./results",
                        help="Where to save CSVs and summary (default: ./results)")
    parser.add_argument("--make_samples", action="store_true",
                        help="Auto-generate synthetic images (no external data needed)")
    parser.add_argument("--load_existing", action="store_true",
                        help="Reload trajectories from --data_dir instead of generating")
    parser.add_argument("--n_images", type=int, default=None)
    parser.add_argument("--n_rounds", type=int, default=20)
    parser.add_argument("--mode", choices=["jpeg", "noise", "combined"], default="combined")
    parser.add_argument("--seed", type=int, default=42)
    # Model args
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="DriftScore PD weight (default 0.5)")
    parser.add_argument("--device", default=None, help="cuda | cpu")
    parser.add_argument("--skip_baselines", action="store_true",
                        help="Only run DriftScore (faster, no pyiqa needed)")
    # Evaluation args
    parser.add_argument("--true_inflection", type=int, default=1,
                        help="Round at which degradation begins (default 1 = immediate)")

    args = parser.parse_args()
    run(args)
