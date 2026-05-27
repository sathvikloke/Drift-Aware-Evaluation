# DriftScore: Anchor-Relative Metric for Multi-Turn Image Quality Drift

Code for the paper **"DriftScore: An Anchor-Relative Metric for Detecting Quality Drift in Multi-Turn Multimodal Generation"**, presented at EvalMG @ ACM SIGIR 2026.

> **TL;DR** — Existing NR-IQA metrics score each frame in isolation. Classical ones (BRISQUE, NIQE, PIQE) actually *improve* as images degrade (τ ≈ −0.8). DriftScore fixes this by measuring all drift relative to the original round-zero image.

---

## What's in this repo

| File | Description |
|---|---|
| `drift_score.py` | Core DriftScore implementation (PD + SD) |
| `baselines.py` | Wrapper for 14 NR-IQA baselines via pyiqa |
| `generate_data.py` | Synthetic degradation benchmark generator |
| `run_evaluation.py` | Main evaluation script — runs all metrics, outputs CSVs |
| `plot_results.py` | Generates the paper figures from results CSVs |
| `edival_adapter.py` | Downloads EdiVal-Bench and runs DriftScore on real editing trajectories |
| `requirements.txt` | Python dependencies |

---

## Quick start (no external data needed)

```bash
# Install dependencies
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git

# Generate synthetic images + run DriftScore only (fast, ~1 min)
python run_evaluation.py --make_samples --n_images 10 --n_rounds 20 --skip_baselines

# Run everything including all 14 NR-IQA baselines (~20 min on CPU)
python run_evaluation.py --make_samples --n_images 10 --n_rounds 20

# Generate figures
python plot_results.py --results_dir ./results
```

---

## How DriftScore works

Given a trajectory $\mathcal{T} = (I_0, I_1, \ldots, I_N)$ where $I_0$ is the original image:

**Perceptual Drift (PD)** — cumulative-average LPIPS distance from the anchor:

$$\text{PD}(t) = \frac{1}{t} \sum_{k=1}^{t} \text{LPIPS}(I_0, I_k)$$

**Semantic Drift (SD)** — drop in CLIP alignment relative to the original prompt:

$$\text{SD}(t) = \max\left(0,\ \cos(\text{CLIP}_I(I_0), \text{CLIP}_T(p_0)) - \cos(\text{CLIP}_I(I_t), \text{CLIP}_T(p_0))\right)$$

Both are normalised to [0,1] per trajectory before combining (necessary because LPIPS and CLIP operate on scales ~10× apart):

$$\text{DriftScore}(t) = \alpha \cdot \widetilde{\text{PD}}(t) + (1 - \alpha) \cdot \widetilde{\text{SD}}(t)$$

Default α = 0.5. Higher DriftScore = more drift from the anchor.

---

## Using DriftScore in your own project

```python
from PIL import Image
from drift_score import DriftScore

scorer = DriftScore(alpha=0.5, device="cpu")  # or "cuda"

# Load your trajectory as a list of PIL Images, anchor first
frames = [Image.open(f) for f in ["round0.jpg", "round1.jpg", "round2.jpg"]]
prompt = "a cat sitting on a red sofa"  # optional

combined, pd_scores, sd_scores = scorer.score_trajectory(frames, prompt=prompt)
# combined[i] = DriftScore after round i+1
# pd_scores[i] = PD component
# sd_scores[i] = SD component
```

---

## Results

On 10 synthetic degradation trajectories (20 rounds, combined JPEG + noise):

| Metric | τ (mean) | τ (std) | Infl. Acc. |
|---|---|---|---|
| **DriftScore-PD (ours)** | **+1.000** | 0.000 | 0% |
| MUSIQ | +0.885 | 0.027 | 0% |
| MANIQA | +0.826 | 0.015 | 100% |
| CLIP-IQA | +0.717 | 0.095 | 100% |
| **DriftScore (ours)** | **+0.505** | 0.265 | **80%** |
| *BRISQUE* | *−0.705* | 0.066 | 100%† |
| *PIQE* | *−0.787* | 0.057 | 100%† |
| *NIQE* | *−0.885* | 0.027 | 100%† |

† 100% inflection accuracy with negative τ means the metric detects a rapid *perceived quality gain* — the opposite of what it should detect.

On 10 real EdiVal-Bench trajectories (IP2P model, 3 editing turns):

| Metric | τ (mean) |
|---|---|
| DriftScore-PD | +0.867 |
| DriftScore (combined) | +0.633 |
| DriftScore-SD | +0.500 |

SD goes from τ ≈ 0 on synthetic images to +0.500 on real text-guided editing, confirming it measures semantic drift that only appears in real generation.

---

## Real data: EdiVal-Bench case study

```bash
pip install datasets huggingface_hub --break-system-packages

# Inspect the dataset structure
python edival_adapter.py --inspect

# Run DriftScore on 10 sequences from the IP2P model
python edival_adapter.py --n_sequences 10 --model IP2P --out_dir ./edival_results
```

Available models: `AnyEdit`, `Flux_dev`, `Flux_max`, `GPT4o`, `IP2P`, `MagicBrush`, `QWEN`, `UltraEdit`, and more. See `edival_adapter.py` for the full list.

---

## Citation

```bibtex
@inproceedings{loke2026driftscore,
  title     = {DriftScore: An Anchor-Relative Metric for Detecting Quality Drift in Multi-Turn Multimodal Generation},
  author    = {Loke, Sathvik},
  booktitle = {Workshop on Evaluation of Multimodal Generation (EvalMG) @ ACM SIGIR},
  year      = {2026}
}
```
