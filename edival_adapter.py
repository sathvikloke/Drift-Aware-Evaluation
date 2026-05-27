"""
EdiVal-Bench adapter for DriftScore evaluation.

The C-Tianyu/EdiVal repo stores everything as ZIP files:
  input_images_resize_512.zip  — original 512x512 source images
  {ModelName}_generation.zip   — edited images from each model (16 models)

This adapter:
  1. Downloads the originals ZIP + one (or more) model ZIPs via hf_hub_download
  2. Extracts them to a local cache
  3. Prints the folder structure (--inspect) so we can see how edits are stored
  4. Builds DriftScore-ready sequences: [original, edit_1, edit_2, ...]
  5. Runs DriftScore (PD + SD) and plots the results

Usage:
    # Step 1 — inspect folder structure inside the ZIPs
    python edival_adapter.py --inspect

    # Step 2 — run DriftScore on 10 sequences from one model
    python edival_adapter.py --n_sequences 10 --model IP2P --out_dir ./edival_results
"""

from __future__ import annotations
import argparse
import os
import sys
import zipfile
from pathlib import Path
from typing import List, Dict, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_ID = "C-Tianyu/EdiVal"

# All model ZIP names (without _generation.zip suffix)
ALL_MODELS = [
    "AnyEdit", "Flux2_max", "Flux_dev", "Flux_max", "GPT4o",
    "GPT_Image_1_5", "GeminiFlash", "IP2P", "MagicBrush",
    "NanoBanana", "Nano_Banana2", "OmniGen", "QWEN",
    "Seedream4", "StepEdit", "UltraEdit",
]

CACHE_DIR = Path.home() / ".cache" / "edival_bench"

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_zip(filename: str, cache_dir: Path) -> Path:
    """Download one ZIP from the HF Hub and return its local path."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run:")
        print("  pip install huggingface_hub --break-system-packages")
        sys.exit(1)

    dest = cache_dir / filename
    if dest.exists():
        print(f"  [cache] {filename}")
        return dest

    print(f"  [download] {filename} ...")
    local = hf_hub_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        filename=filename,
        local_dir=str(cache_dir),
    )
    return Path(local)


def _extract_zip(zip_path: Path, extract_to: Path) -> Path:
    """Extract a ZIP to extract_to/<stem>/ and return that directory."""
    stem = zip_path.stem           # e.g. "IP2P_generation"
    out_dir = extract_to / stem
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"  [cache] already extracted: {out_dir.name}/")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [extract] {zip_path.name} → {out_dir.name}/ ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    return out_dir


# ---------------------------------------------------------------------------
# Inspect
# ---------------------------------------------------------------------------

def inspect_dataset(cache_dir: Path = CACHE_DIR) -> None:
    """
    Download the originals ZIP + one small model ZIP, extract them,
    and print the full folder/file tree so we can understand the layout.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = cache_dir / "extracted"

    print("\n=== Downloading originals ZIP ===")
    orig_zip = _download_zip("input_images_resize_512.zip", cache_dir)
    orig_dir = _extract_zip(orig_zip, extract_dir)

    print("\n=== Downloading one model ZIP (IP2P) ===")
    model_zip = _download_zip("IP2P_generation.zip", cache_dir)
    model_dir = _extract_zip(model_zip, extract_dir)

    print("\n=== Originals folder tree (first 40 entries) ===")
    _print_tree(orig_dir, max_entries=40)

    print("\n=== IP2P model folder tree (first 60 entries) ===")
    _print_tree(model_dir, max_entries=60)

    print("\n=== Image file extensions found ===")
    from collections import Counter
    exts = Counter(
        p.suffix.lower()
        for p in list(orig_dir.rglob("*")) + list(model_dir.rglob("*"))
        if p.is_file()
    )
    print(dict(exts))


def _print_tree(root: Path, max_entries: int = 60) -> None:
    entries = sorted(root.rglob("*"))
    for i, p in enumerate(entries):
        if i >= max_entries:
            print(f"  ... ({len(entries)} total entries)")
            break
        indent = "  " + "  " * (len(p.relative_to(root).parts) - 1)
        marker = "/" if p.is_dir() else ""
        print(f"{indent}{p.name}{marker}")


# ---------------------------------------------------------------------------
# Sequence building
# ---------------------------------------------------------------------------

def _find_image_files(directory: Path) -> List[Path]:
    """Return sorted list of image files under directory."""
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    files = sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    )
    return files


def build_sequences_from_disk(
    orig_dir: Path,
    model_dir: Path,
    model_name: str,
    n_sequences: int = 10,
) -> List[Dict]:
    """
    Build DriftScore sequences from the confirmed EdiVal layout:

      orig_dir/
        {id}_input_raw.jpg            ← anchor (round 0)

      model_dir/multipass/
        {id}_input_raw.png            ← model output before any edit turns
        {id}_input_raw_turn_1.png     ← after editing turn 1
        {id}_input_raw_turn_2.png     ← after editing turn 2
        {id}_input_raw_turn_3.png     ← after editing turn 3

    Sequence: [original .jpg, turn_1, turn_2, turn_3]  → 3 editing rounds.
    The model's {id}_input_raw.png is NOT included because we use the
    true unedited original as the DriftScore anchor.
    """
    from PIL import Image

    # All original images: {id}_input_raw.jpg  (may be in a nested subfolder)
    orig_files = sorted(orig_dir.glob("*_input_raw.jpg"))
    if not orig_files:
        orig_files = sorted(orig_dir.rglob("*_input_raw.jpg"))
    if not orig_files:
        # Fallback: any jpg anywhere under orig_dir
        orig_files = sorted(orig_dir.rglob("*.jpg"))
    if not orig_files:
        print(f"ERROR: No original images found in {orig_dir}")
        sys.exit(1)

    print(f"  Found {len(orig_files)} original images")

    # Locate the multipass/ directory — it may be nested one level deeper
    # if the ZIP extracted as model_dir/{model_name}/multipass/
    multipass_dir = model_dir / "multipass"
    if not multipass_dir.is_dir():
        # Search one level deeper (ZIP extracted with extra root folder)
        candidates = sorted(model_dir.rglob("multipass"))
        multipass_dir = next((c for c in candidates if c.is_dir()), model_dir)
    print(f"  Edit dir: {multipass_dir}")

    sequences = []

    for orig_path in orig_files:
        if len(sequences) >= n_sequences:
            break

        # Extract the image id  ("0", "100", "101", ...)
        # Stem is like "0_input_raw" → id_part = "0"
        stem = orig_path.stem                        # e.g. "0_input_raw"
        id_part = stem.split("_")[0]                 # e.g. "0"

        # Find turn files: {id_part}_input_raw_turn_1.png, _turn_2.png, _turn_3.png
        turns = []
        for t in range(1, 10):   # up to 9 turns
            turn_path = multipass_dir / f"{id_part}_input_raw_turn_{t}.png"
            if turn_path.is_file():
                turns.append(turn_path)
            else:
                break   # turns are sequential; stop at first missing

        if not turns:
            continue   # no edits for this image, skip

        # Build frame list: [original] + [turn_1, turn_2, turn_3]
        try:
            orig_img = Image.open(orig_path).convert("RGB")
            frames = [orig_img] + [Image.open(p).convert("RGB") for p in turns]
        except Exception as e:
            print(f"    Warning: skipping {id_part} ({e})")
            continue

        sequences.append({
            "image_id": f"edival_{model_name}_{id_part}",
            "n_rounds": len(frames) - 1,
            "frames":   frames,
            "prompt":   None,   # no prompt metadata in this ZIP; SD uses visual mode
        })
        print(f"    sequence {len(sequences):>3}: id={id_part:>4}  "
              f"({len(frames)-1} turns)")

    return sequences


def load_edival_sequences(
    n_sequences: int = 10,
    model: str = "IP2P",
    cache_dir: Path = CACHE_DIR,
) -> List[Dict]:
    """Download, extract, and return DriftScore-ready sequences."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = cache_dir / "extracted"

    print(f"\n=== Downloading EdiVal-Bench (model={model}) ===")
    orig_zip  = _download_zip("input_images_resize_512.zip", cache_dir)
    model_zip = _download_zip(f"{model}_generation.zip", cache_dir)

    orig_dir  = _extract_zip(orig_zip,  extract_dir)
    model_dir = _extract_zip(model_zip, extract_dir)

    print(f"\n=== Building sequences ===")
    sequences = build_sequences_from_disk(orig_dir, model_dir, model, n_sequences)
    print(f"\n{len(sequences)} sequences ready.\n")
    return sequences


# ---------------------------------------------------------------------------
# Run DriftScore
# ---------------------------------------------------------------------------

def run_on_edival(sequences: List[Dict], out_dir: str, alpha: float = 0.5,
                  device: str = "cpu") -> None:
    import numpy as np
    import pandas as pd
    from scipy.stats import kendalltau

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from drift_score import DriftScore
    scorer = DriftScore(alpha=alpha, device=device)

    all_rows = []
    tau_combined, tau_pd, tau_sd = [], [], []

    for seq in sequences:
        image_id = seq["image_id"]
        frames   = seq["frames"]
        prompt   = seq.get("prompt", None)
        print(f"  Scoring {image_id} ({seq['n_rounds']} rounds) ...")

        combined, pd_s, sd_s = scorer.score_trajectory(frames, prompt=prompt)
        combined_full = [0.0] + combined
        pd_full       = [0.0] + pd_s
        sd_full       = [0.0] + sd_s

        rounds = np.arange(len(combined_full))
        from scipy.stats import kendalltau as kt
        tau_c, _ = kt(rounds, combined_full)
        tau_p, _ = kt(rounds, pd_full)
        tau_s, _ = kt(rounds, sd_full)

        tau_combined.append(float(tau_c))
        tau_pd.append(float(tau_p))
        tau_sd.append(float(tau_s))

        for t, (c, p, s) in enumerate(zip(combined_full, pd_full, sd_full)):
            all_rows.append({
                "image_id": image_id,
                "round": t,
                "DriftScore": c,
                "DriftScore_PD": p,
                "DriftScore_SD": s,
            })

    df = pd.DataFrame(all_rows)
    df.to_csv(out / "edival_scores.csv", index=False)

    summary = pd.DataFrame({
        "metric":      ["DriftScore", "DriftScore_PD", "DriftScore_SD"],
        "mean_tau":    [np.mean(tau_combined), np.mean(tau_pd), np.mean(tau_sd)],
        "std_tau":     [np.std(tau_combined),  np.std(tau_pd),  np.std(tau_sd)],
        "n_sequences": [len(sequences)] * 3,
    })
    summary.to_csv(out / "edival_summary.csv", index=False)

    print("\n" + "="*55)
    print("EdiVal-Bench DriftScore Results")
    print("="*55)
    print(f"{'Metric':<20} {'τ (mean)':>10} {'τ (std)':>9}")
    print("-"*55)
    for _, row in summary.iterrows():
        print(f"{row['metric']:<20} {row['mean_tau']:>+10.4f} {row['std_tau']:>9.4f}")
    print("="*55)
    print(f"\nCSVs saved to {out}/")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_edival_curves(results_csv: str, out_path: str) -> None:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    df = pd.read_csv(results_csv)
    max_round = df["round"].max()
    rounds = np.arange(max_round + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    for col, label, color in [
        ("DriftScore_PD", "PD (perceptual)", "steelblue"),
        ("DriftScore_SD", "SD (semantic)",   "darkorange"),
        ("DriftScore",    "DriftScore",      "black"),
    ]:
        means = df.groupby("round")[col].mean().reindex(rounds)
        stds  = df.groupby("round")[col].std().reindex(rounds).fillna(0)
        ax.plot(rounds, means, label=label, color=color, linewidth=2)
        ax.fill_between(rounds, means - stds, means + stds, alpha=0.15, color=color)

    ax.set_xlabel("Editing round", fontsize=11)
    ax.set_ylabel("DriftScore (↑ = more drift)", fontsize=11)
    ax.set_title("DriftScore on EdiVal-Bench", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    for iid in df["image_id"].unique():
        sub = df[df["image_id"] == iid].sort_values("round")
        ax2.plot(sub["round"], sub["DriftScore_PD"], alpha=0.4, linewidth=1, color="steelblue")
    means_pd = df.groupby("round")["DriftScore_PD"].mean().reindex(rounds)
    ax2.plot(rounds, means_pd, color="steelblue", linewidth=2.5, label="Mean PD", zorder=5)
    ax2.set_xlabel("Editing round", fontsize=11)
    ax2.set_ylabel("PD (perceptual drift)", fontsize=11)
    ax2.set_title("Per-Sequence PD Trajectories", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EdiVal-Bench adapter for DriftScore.")
    parser.add_argument("--inspect", action="store_true",
                        help="Download one model ZIP, extract, print folder tree, and exit.")
    parser.add_argument("--n_sequences", type=int, default=10)
    parser.add_argument("--model", default="IP2P",
                        choices=ALL_MODELS,
                        help="Which model's edits to use (default: IP2P).")
    parser.add_argument("--out_dir", default="./edival_results")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache_dir", default=str(CACHE_DIR),
                        help="Where to cache downloaded ZIPs and extracted files.")
    parser.add_argument("--plot_only", action="store_true",
                        help="Skip scoring; regenerate figure from existing CSV.")
    args = parser.parse_args()

    cache = Path(args.cache_dir)
    out   = Path(args.out_dir)

    if args.inspect:
        inspect_dataset(cache_dir=cache)
        sys.exit(0)

    scores_csv = out / "edival_scores.csv"

    if not args.plot_only:
        sequences = load_edival_sequences(
            n_sequences=args.n_sequences,
            model=args.model,
            cache_dir=cache,
        )
        if not sequences:
            print("ERROR: No sequences built. Run --inspect to debug the folder layout.")
            sys.exit(1)
        run_on_edival(sequences, str(out), alpha=args.alpha, device=args.device)

    fig_path = out / "fig_edival_curves.pdf"
    plot_edival_curves(str(scores_csv), str(fig_path))
    print(f"\nNext: copy {fig_path} to your figures/ folder and update paper.tex.")
