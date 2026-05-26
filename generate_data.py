"""
Synthetic iterative degradation data generator.

Produces two types of sequences that model real multi-turn quality drift:

  1. JPEG recompression (Banana100-style):
       Each round: save as JPEG at quality Q, reload.
       Models quantisation artifact accumulation.

  2. Noise accumulation:
       Each round: add a small amount of Gaussian noise.
       Models stochastic diffusion-style drift.

  3. Combined (default): JPEG + noise, alternating.
       Most realistic proxy for iterative model editing.

Seeds are fixed so experiments are reproducible.

Quick start:
    python generate_data.py --source_dir ./sample_images --out_dir ./data --n_rounds 20
"""

from __future__ import annotations
import argparse
import io
import os
import random
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
from PIL import Image

# ── degradation primitives ─────────────────────────────────────────────────

def jpeg_round(img: Image.Image, quality: int = 70) -> Image.Image:
    """One round of JPEG recompression at the given quality."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def noise_round(img: Image.Image, sigma: float = 5.0) -> Image.Image:
    """One round of additive Gaussian noise (sigma in pixel units 0-255)."""
    arr = np.array(img.convert("RGB")).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def blur_round(img: Image.Image, radius: float = 0.8) -> Image.Image:
    """One round of slight Gaussian blur (models high-frequency detail loss)."""
    from PIL import ImageFilter
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


# ── sequence generators ────────────────────────────────────────────────────

def generate_jpeg_sequence(
    img: Image.Image,
    n_rounds: int = 20,
    quality: int = 70,
    seed: int = 0,
) -> List[Image.Image]:
    """
    Iterative JPEG recompression sequence.
    Returns [I_0, I_1, ..., I_n_rounds].
    """
    np.random.seed(seed)
    frames = [img.copy()]
    current = img.copy()
    for _ in range(n_rounds):
        current = jpeg_round(current, quality=quality)
        frames.append(current.copy())
    return frames


def generate_noise_sequence(
    img: Image.Image,
    n_rounds: int = 20,
    sigma: float = 6.0,
    seed: int = 0,
) -> List[Image.Image]:
    """
    Cumulative Gaussian noise accumulation sequence.
    Each round adds independent noise; artifacts compound visually.
    Returns [I_0, I_1, ..., I_n_rounds].
    """
    np.random.seed(seed)
    frames = [img.copy()]
    current = img.copy()
    for _ in range(n_rounds):
        current = noise_round(current, sigma=sigma)
        frames.append(current.copy())
    return frames


def generate_combined_sequence(
    img: Image.Image,
    n_rounds: int = 20,
    jpeg_quality: int = 75,
    noise_sigma: float = 4.0,
    blur_radius: float = 0.5,
    seed: int = 0,
) -> List[Image.Image]:
    """
    Combined degradation: JPEG + noise + occasional blur, mimicking
    the compounding artifacts seen in real iterative model editing.
    Returns [I_0, I_1, ..., I_n_rounds].
    """
    np.random.seed(seed)
    random.seed(seed)
    frames = [img.copy()]
    current = img.copy()
    for t in range(n_rounds):
        # Always JPEG compress
        current = jpeg_round(current, quality=jpeg_quality)
        # Add noise every round
        current = noise_round(current, sigma=noise_sigma)
        # Blur every 3rd round (models diffusion upsampling artifacts)
        if t % 3 == 2:
            current = blur_round(current, radius=blur_radius)
        frames.append(current.copy())
    return frames


# ── dataset builder ────────────────────────────────────────────────────────

GENERATORS = {
    "jpeg":     generate_jpeg_sequence,
    "noise":    generate_noise_sequence,
    "combined": generate_combined_sequence,
}

# Default prompts for sequences (used when scoring with DriftScore + prompt).
# If your source image comes with a caption, pass it in instead.
DEFAULT_PROMPTS = [
    "a high quality photograph of a natural scene",
    "a detailed image with rich textures and vivid colors",
    "a sharp and well-lit portrait photograph",
    "a photorealistic image with fine details",
    "a clear and crisp outdoor photograph",
]


def build_dataset(
    source_dir: str,
    out_dir: str,
    n_rounds: int = 20,
    mode: Literal["jpeg", "noise", "combined"] = "combined",
    n_images: Optional[int] = None,
    seed: int = 42,
) -> List[dict]:
    """
    Build a dataset of degradation trajectories from source images.

    Args:
        source_dir: Directory containing source PNG/JPG images.
        out_dir:    Where to save trajectory frames.
        n_rounds:   Number of degradation rounds per sequence.
        mode:       Degradation type (jpeg | noise | combined).
        n_images:   Cap on number of source images (None = use all).
        seed:       Random seed for reproducibility.

    Returns:
        List of dicts, each with keys:
            image_id, prompt, frames (list of PIL Images), out_subdir
    """
    source_dir = Path(source_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted([p for p in source_dir.iterdir() if p.suffix.lower() in exts])

    if not image_paths:
        raise FileNotFoundError(f"No images found in {source_dir}")

    if n_images is not None:
        image_paths = image_paths[:n_images]

    generator_fn = GENERATORS[mode]
    dataset = []
    random.seed(seed)

    for idx, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert("RGB")
        # Resize to 512x512 for consistency
        img = img.resize((512, 512), Image.LANCZOS)

        prompt = DEFAULT_PROMPTS[idx % len(DEFAULT_PROMPTS)]
        img_id = f"{mode}_{idx:04d}"
        subdir = out_dir / img_id
        subdir.mkdir(exist_ok=True)

        frames = generator_fn(img, n_rounds=n_rounds, seed=seed + idx)

        # Save frames to disk
        for t, frame in enumerate(frames):
            frame.save(subdir / f"round_{t:03d}.png")

        dataset.append({
            "image_id":  img_id,
            "source":    str(img_path),
            "prompt":    prompt,
            "frames":    frames,
            "out_subdir": str(subdir),
            "n_rounds":  n_rounds,
            "mode":      mode,
        })

        print(f"  [{idx+1}/{len(image_paths)}] {img_id}: {n_rounds} rounds saved → {subdir}")

    return dataset


def load_dataset(data_dir: str) -> List[dict]:
    """
    Reload a previously generated dataset from disk.

    Args:
        data_dir: Directory produced by build_dataset (contains subdirs per sequence).

    Returns:
        Same format as build_dataset return value (without frames pre-loaded into memory).
    """
    data_dir = Path(data_dir)
    dataset = []

    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir():
            continue
        frame_paths = sorted(subdir.glob("round_*.png"))
        if not frame_paths:
            continue
        frames = [Image.open(p).convert("RGB") for p in frame_paths]
        dataset.append({
            "image_id":  subdir.name,
            "source":    None,
            "prompt":    DEFAULT_PROMPTS[0],  # override with real prompts if available
            "frames":    frames,
            "out_subdir": str(subdir),
            "n_rounds":  len(frames) - 1,
            "mode":      "unknown",
        })

    return dataset


# ── sample image generator (no external images needed) ────────────────────

def make_sample_images(out_dir: str, n: int = 5, size: int = 512) -> List[str]:
    """
    Generate n synthetic test images using procedural patterns.
    Useful for a quick smoke-test when you don't have source images.
    """
    from PIL import ImageDraw
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    rng = np.random.default_rng(0)

    for i in range(n):
        img = Image.new("RGB", (size, size), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Random geometric pattern to give metrics something to measure
        for _ in range(30):
            x0, y0 = rng.integers(0, size, 2)
            x1, y1 = x0 + rng.integers(20, 150), y0 + rng.integers(20, 150)
            color = tuple(rng.integers(0, 255, 3).tolist())
            draw.rectangle([x0, y0, x1, y1], fill=color)

        for _ in range(20):
            x, y = rng.integers(0, size, 2)
            r = rng.integers(10, 80)
            color = tuple(rng.integers(0, 255, 3).tolist())
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

        # Add text-like noise to simulate content
        arr = np.array(img)
        texture = rng.integers(0, 30, arr.shape, dtype=np.uint8)
        arr = np.clip(arr.astype(np.int32) + texture, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

        path = out_dir / f"sample_{i:03d}.png"
        img.save(path)
        paths.append(str(path))

    return paths


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate iterative degradation sequences.")
    parser.add_argument("--source_dir", default="./sample_images",
                        help="Directory of source images (default: ./sample_images)")
    parser.add_argument("--out_dir", default="./data",
                        help="Output directory for trajectories (default: ./data)")
    parser.add_argument("--n_rounds", type=int, default=20,
                        help="Number of degradation rounds per sequence (default: 20)")
    parser.add_argument("--mode", choices=["jpeg", "noise", "combined"], default="combined",
                        help="Degradation type (default: combined)")
    parser.add_argument("--n_images", type=int, default=None,
                        help="Max number of source images to use")
    parser.add_argument("--make_samples", action="store_true",
                        help="Generate synthetic sample images first (no external data needed)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.make_samples:
        print("Generating synthetic sample images ...")
        paths = make_sample_images(args.source_dir, n=10)
        print(f"  Saved {len(paths)} images to {args.source_dir}\n")

    print(f"Building {args.mode} degradation dataset ...")
    dataset = build_dataset(
        source_dir=args.source_dir,
        out_dir=args.out_dir,
        n_rounds=args.n_rounds,
        mode=args.mode,
        n_images=args.n_images,
        seed=args.seed,
    )
    print(f"\nDone. {len(dataset)} sequences with {args.n_rounds} rounds each.")
    print(f"Data saved to: {args.out_dir}")
