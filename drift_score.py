"""
DriftScore: Anchor-relative metric for detecting quality drift
in multi-turn multimodal generation.

Components:
  PD(t) = cumulative average LPIPS distance from anchor I_0
  SD(t) = CLIP-based semantic drift from anchor I_0
           - if prompt provided: measures drop in image-text alignment vs. round 0
           - if prompt omitted:  measures cosine drift in CLIP image space

  DriftScore(t) = alpha * PD(t) + (1 - alpha) * SD(t)
  Higher score = more drift = worse quality.
"""

from __future__ import annotations  # makes all annotations lazy strings — safe with optional torch

import numpy as np
from PIL import Image
from typing import List, Optional, Tuple

# torch is imported lazily inside the class so this file is importable
# even before torch is installed (useful for running CLI helpers standalone).
try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ── lazy imports so the class is importable even if deps aren't installed ──

def _load_lpips(device):
    try:
        import lpips
        return lpips.LPIPS(net="alex", verbose=False).to(device)
    except ImportError:
        raise ImportError("lpips not installed. Run: pip install lpips")


def _load_clip(device):
    try:
        import clip as openai_clip
        model, preprocess = openai_clip.load("ViT-B/32", device=device)
        model.eval()
        return model, preprocess
    except ImportError:
        raise ImportError(
            "openai-clip not installed. Run: pip install git+https://github.com/openai/CLIP.git"
        )


# ── helpers ────────────────────────────────────────────────────────────────

def pil_to_lpips_tensor(img: Image.Image, device: str) -> torch.Tensor:
    """Convert PIL image to LPIPS-compatible tensor in [-1, 1], shape (1,3,H,W)."""
    img = img.convert("RGB").resize((256, 256), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0  # [0,255] -> [-1,1]
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


def pil_to_clip_tensor(img: Image.Image, preprocess, device: str) -> torch.Tensor:
    """Preprocess PIL image for CLIP."""
    return preprocess(img).unsqueeze(0).to(device)


# ── main class ─────────────────────────────────────────────────────────────

class DriftScore:
    """
    Computes DriftScore for a generation trajectory.

    Usage:
        ds = DriftScore(alpha=0.5, device="cpu")
        scores, pd_scores, sd_scores = ds.score(frames, anchor=frames[0], prompt="a cat")
    """

    def __init__(self, alpha: float = 0.5, device: Optional[str] = None):
        """
        Args:
            alpha:  Weight for perceptual drift (PD). 1-alpha goes to semantic drift (SD).
                    Default 0.5 per paper ablation sweet-spot.
            device: "cuda" | "cpu". Auto-detected if None.
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required. Run: pip install torch torchvision")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.alpha = alpha

        print(f"[DriftScore] Loading LPIPS (net=alex) on {device} ...")
        self.lpips_fn = _load_lpips(device)

        print(f"[DriftScore] Loading CLIP (ViT-B/32) on {device} ...")
        self.clip_model, self.clip_preprocess = _load_clip(device)

        print("[DriftScore] Ready.\n")

    # ── perceptual drift ───────────────────────────────────────────────────

    def _compute_pd(
        self, frames: List[Image.Image], anchor: Image.Image
    ) -> List[float]:
        """
        PD(t) = cumulative average LPIPS distance from anchor I_0.

        Cumulative average penalises sustained drift even when any single
        round looks minor, matching the compounding nature of degradation.
        """
        anchor_t = pil_to_lpips_tensor(anchor, self.device)
        running_sum = 0.0
        pd_scores = []

        with torch.no_grad():
            for i, frame in enumerate(frames, start=1):
                frame_t = pil_to_lpips_tensor(frame, self.device)
                d = self.lpips_fn(frame_t, anchor_t).item()
                running_sum += d
                pd_scores.append(running_sum / i)   # cumulative average

        return pd_scores

    # ── semantic drift ─────────────────────────────────────────────────────

    def _compute_sd_visual(
        self, frames: List[Image.Image], anchor: Image.Image
    ) -> List[float]:
        """
        SD(t) without prompt: cosine drift in CLIP image space from I_0.
        SD(t) = 1 - cos_sim(CLIP_img(I_t), CLIP_img(I_0))
        """
        anchor_t = pil_to_clip_tensor(anchor, self.clip_preprocess, self.device)
        with torch.no_grad():
            anchor_feat = self.clip_model.encode_image(anchor_t).float()
            anchor_feat = F.normalize(anchor_feat, dim=-1)

        sd_scores = []
        with torch.no_grad():
            for frame in frames:
                frame_t = pil_to_clip_tensor(frame, self.clip_preprocess, self.device)
                frame_feat = self.clip_model.encode_image(frame_t).float()
                frame_feat = F.normalize(frame_feat, dim=-1)
                cos_sim = (frame_feat * anchor_feat).sum().item()
                sd_scores.append(1.0 - cos_sim)

        return sd_scores

    def _compute_sd_prompt(
        self,
        frames: List[Image.Image],
        anchor: Image.Image,
        prompt: str,
    ) -> List[float]:
        """
        SD(t) with prompt: measures drop in image-text alignment relative to round 0.
        SD(t) = max(0, align(I_0, p_0) - align(I_t, p_0))

        Returns raw alignment drop (not normalised here — scale normalisation
        happens in score() to ensure PD and SD are on comparable scales).
        A positive score means the edit has drifted from the original prompt intent.
        """
        import clip as openai_clip

        text_t = openai_clip.tokenize([prompt]).to(self.device)
        anchor_t = pil_to_clip_tensor(anchor, self.clip_preprocess, self.device)

        with torch.no_grad():
            text_feat = self.clip_model.encode_text(text_t).float()
            text_feat = F.normalize(text_feat, dim=-1)

            anchor_feat = self.clip_model.encode_image(anchor_t).float()
            anchor_feat = F.normalize(anchor_feat, dim=-1)
            anchor_align = (anchor_feat * text_feat).sum().item()

        sd_scores = []
        with torch.no_grad():
            for frame in frames:
                frame_t = pil_to_clip_tensor(frame, self.clip_preprocess, self.device)
                frame_feat = self.clip_model.encode_image(frame_t).float()
                frame_feat = F.normalize(frame_feat, dim=-1)
                frame_align = (frame_feat * text_feat).sum().item()
                # Raw alignment drop — clipped at 0 (improvement vs. anchor is not drift)
                drift = max(0.0, anchor_align - frame_align)
                sd_scores.append(drift)

        return sd_scores

    # ── combined score ─────────────────────────────────────────────────────

    def score(
        self,
        frames: List[Image.Image],
        anchor: Image.Image,
        prompt: Optional[str] = None,
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Compute DriftScore for a generation trajectory.

        Args:
            frames: List of PIL Images for rounds t=1..N (do NOT include anchor).
            anchor: The round-0 image (I_0).
            prompt: Original text prompt p_0. If None, visual-only SD is used.

        Returns:
            (drift_scores, pd_scores, sd_scores)
            All lists have length == len(frames).
            Higher values indicate more drift (worse quality).
        """
        pd = self._compute_pd(frames, anchor)

        if prompt is not None:
            sd = self._compute_sd_prompt(frames, anchor, prompt)
        else:
            sd = self._compute_sd_visual(frames, anchor)

        # Normalise both components to their trajectory maximum before combining.
        # LPIPS distances (~0.05-0.5) and CLIP cosine distances (~0.01-0.08) live
        # on very different scales; without this, PD dominates by ~5-10x regardless
        # of alpha. Dividing by the per-trajectory max maps each to [0, 1] so that
        # alpha=0.5 genuinely gives equal weight to both components.
        pd_max = max(pd) if max(pd) > 1e-8 else 1.0
        sd_max = max(sd) if max(sd) > 1e-8 else 1.0
        pd_norm = [p / pd_max for p in pd]
        sd_norm = [s / sd_max for s in sd]

        combined = [self.alpha * p + (1 - self.alpha) * s
                    for p, s in zip(pd_norm, sd_norm)]
        # Return raw (un-normalised) pd/sd so callers can inspect absolute magnitudes,
        # but return normalised combined score as the primary DriftScore.
        return combined, pd, sd

    def score_trajectory(
        self,
        trajectory: List[Image.Image],
        prompt: Optional[str] = None,
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Convenience wrapper: pass the full trajectory including round 0.
        trajectory[0] is treated as the anchor (I_0).
        Returns scores for rounds 1..N.
        """
        assert len(trajectory) >= 2, "Trajectory must have at least 2 frames (anchor + 1 edit)."
        return self.score(trajectory[1:], anchor=trajectory[0], prompt=prompt)
