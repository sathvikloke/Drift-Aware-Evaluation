"""
NR-IQA Baseline Metrics wrapper.

Uses the pyiqa library (https://github.com/chaofengc/IQA-PyTorch) which
provides a unified interface to ~40 no-reference image quality metrics.

All metrics are normalised to "higher = better quality" so they can be
compared on equal footing. We then invert to "higher = more degraded"
to match DriftScore's convention when running evaluation.

Metrics included (14 total):
  Classical / signal-based:
    brisque, niqe, piqe, ilniqe
  Deep CNN-based:
    cnniqa, dbcnn, hyperiqa, musiq
  Transformer / attention-based:
    maniqa, tres
  CLIP / VLM-based:
    clipiqa, clipiqa+
  Ranking / ensemble:
    topiq_nr, arniqa
"""

from __future__ import annotations
import numpy as np
from PIL import Image
from typing import List, Dict, Optional
import warnings

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# Metrics that pyiqa scores as lower = better quality (we flip these).
# Check pyiqa docs / source: iqa_metric.lower_better attribute.
_LOWER_BETTER = {"brisque", "niqe", "piqe", "ilniqe"}

# Some metrics are slow or require extra downloads; mark optional ones.
_OPTIONAL_METRICS = {"tres", "maniqa"}

# Full list to attempt loading
METRIC_NAMES = [
    "brisque",
    "niqe",
    "piqe",
    "ilniqe",
    "cnniqa",
    "dbcnn",
    "hyperiqa",
    "musiq",
    "maniqa",
    "tres",
    "clipiqa",
    "clipiqa+",
    "topiq_nr",
    "arniqa",
]


class NRIQABaselines:
    """
    Loads and runs a collection of NR-IQA metrics from pyiqa.

    Usage:
        bl = NRIQABaselines(device="cpu")
        scores = bl.score_image(pil_image)   # dict {metric_name: float}
        trajectory_scores = bl.score_trajectory(frames)  # dict {metric: [float x N]}
    """

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        device: Optional[str] = None,
    ):
        try:
            import pyiqa
        except ImportError:
            raise ImportError(
                "pyiqa not installed. Run: pip install pyiqa"
            )

        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required. Run: pip install torch torchvision")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        target = metrics or METRIC_NAMES
        self.metrics: Dict[str, object] = {}
        self.lower_better: Dict[str, bool] = {}

        print(f"[Baselines] Loading {len(target)} NR-IQA metrics on {device} ...")
        for name in target:
            try:
                m = pyiqa.create_metric(name, device=device, as_loss=False)
                self.metrics[name] = m
                # pyiqa exposes lower_better; fall back to our hardcoded set
                lb = getattr(m, "lower_better", name in _LOWER_BETTER)
                self.lower_better[name] = lb
                print(f"  ✓ {name:15s}  (lower_better={lb})")
            except Exception as e:
                if name not in _OPTIONAL_METRICS:
                    warnings.warn(f"  ✗ {name}: failed to load — {e}")
                else:
                    print(f"  – {name:15s}  (optional, skipped: {e})")

        print(f"[Baselines] Loaded {len(self.metrics)} metrics.\n")

    # ── internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _pil_to_tensor(img: Image.Image, device: str) -> torch.Tensor:
        """Convert PIL image to pyiqa-compatible tensor: (1,3,H,W) in [0,1]."""
        img = img.convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return t.to(device)

    def _raw_score(self, name: str, tensor: torch.Tensor) -> float:
        """Return raw metric score for a single image tensor."""
        with torch.no_grad():
            try:
                val = self.metrics[name](tensor)
                if isinstance(val, torch.Tensor):
                    val = val.item()
                return float(val)
            except Exception as e:
                warnings.warn(f"[Baselines] {name} inference failed: {e}")
                return float("nan")

    # ── public API ─────────────────────────────────────────────────────────

    def score_image(self, img: Image.Image) -> Dict[str, float]:
        """
        Score a single image with all loaded metrics.

        Returns dict of {metric_name: score} where higher = better quality.
        (lower_better metrics are negated so the convention is uniform.)
        """
        tensor = self._pil_to_tensor(img, self.device)
        result = {}
        for name in self.metrics:
            raw = self._raw_score(name, tensor)
            # Normalise: flip lower_better metrics so higher always = better
            result[name] = -raw if self.lower_better[name] else raw
        return result

    def score_trajectory(
        self, trajectory: List[Image.Image]
    ) -> Dict[str, List[float]]:
        """
        Score every frame in a trajectory.

        Args:
            trajectory: list of PIL Images (including round 0).

        Returns:
            dict {metric_name: [score_round_0, score_round_1, ..., score_round_N]}
            Higher score = better quality (convention consistent with score_image).
        """
        per_metric: Dict[str, List[float]] = {n: [] for n in self.metrics}
        for i, frame in enumerate(trajectory):
            scores = self.score_image(frame)
            for name, val in scores.items():
                per_metric[name].append(val)
        return per_metric

    @property
    def metric_names(self) -> List[str]:
        return list(self.metrics.keys())
