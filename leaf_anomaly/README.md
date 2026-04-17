# Leaf Anomaly Detector (PaDiM)

A pixel-accurate anomaly detector for plant leaves. The system trains on a small
set of reference images of healthy leaves and — without any labelled defect data —
produces per-pixel Mahalanobis-distance heatmaps for new test images. The approach
is **PaDiM** (Patch Distribution Modeling, Defard et al. 2020): a frozen ResNet18
extracts multi-scale patch features; per-patch multivariate Gaussians are fitted
over the reference set; at inference, distance from the Gaussian model reveals
anomalous regions (holes, spots, discolouration, deformation).

Target deployment platform: **NVIDIA Jetson Orin Nano / NX** (JetPack 6.x).

---

## Installation — Desktop (Linux / macOS)

```bash
pip install -r requirements.txt
```

Python 3.10+ recommended.

---

## Installation — Jetson Orin Nano / NX (JetPack 6.x)

> **Do not** install PyTorch or torchvision via pip on Jetson — the generic wheels
> lack CUDA / cuDNN support for the ARM architecture.

1. **PyTorch & torchvision**: Download the official NVIDIA Jetson wheels for your
   exact JetPack version from:
   ```
   https://developer.download.nvidia.com/compute/redist/jp/
   ```
   The exact wheel filenames vary by JetPack release. Consult the current
   [NVIDIA Jetson Forums](https://forums.developer.nvidia.com/) and the JetPack
   release notes to find the correct URLs for your setup, then install:
   ```bash
   pip install torch-*.whl torchvision-*.whl
   ```

2. **OpenCV**: Use the hardware-accelerated system package rather than the pip
   wheel (better performance on Jetson):
   ```bash
   sudo apt install python3-opencv
   ```

3. **Remaining dependencies**:
   ```bash
   pip install numpy pillow
   # Do NOT install opencv-python via pip if you used apt above
   ```

---

## Usage

### 1. (Optional) Generate synthetic test data

```bash
cd leaf_anomaly/
python make_synthetic.py
```

Creates `data/reference/` (20 normal leaves) and `data/test/` (5 normal + 9
anomalous images) so you can run a full end-to-end test without real photos.

### 2. Train

```bash
python src/train.py --ref data/reference --out models/padim.pkl
# With GPU and larger batches:
python src/train.py --ref data/reference --out models/padim.pkl --device cuda --batch-size 8
# With post-training sanity check:
python src/train.py --ref data/reference --out models/padim.pkl --sanity-check
```

### 3. Detect

```bash
# Single image
python src/detect.py --model models/padim.pkl --input data/test/spot_000.png --output output

# Full folder, with side-by-side comparison panels and fixed color scale
python src/detect.py --model models/padim.pkl --input data/test --output output \
    --vmin 0 --vmax 50 --comparison

# With threshold masking (only highlight pixels above score 0.3)
python src/detect.py --model models/padim.pkl --input data/test --output output \
    --threshold 0.3 --vmin 0 --vmax 50
```

Output files per image:
- `<stem>_overlay.png` — heatmap blended over the original
- `<stem>_comparison.png` — Original | Heatmap | Overlay panel (with `--comparison`)

---

## Performance Tips — Jetson Orin Nano

**Maximize clock speeds** before running inference:

```bash
sudo nvpmodel -m 0     # switch to MAXN power mode
sudo jetson_clocks     # lock all clocks to maximum frequency
```

**Expected latency** at 224×224 input:

| Mode         | Approx. latency |
|--------------|-----------------|
| MAXN (15 W)  | ~50–100 ms/image |
| Low-power    | ~100–150 ms/image |

The bottleneck is the ResNet18 forward pass. The Mahalanobis computation is small
in comparison.

**Further acceleration — ONNX → TensorRT (FP16)**:

For production workloads, export the ResNet18 feature extractor to ONNX and
compile it with TensorRT FP16. The Mahalanobis distance computation (small
matrix ops on the CPU/GPU) can stay in PyTorch. This typically reduces the
ResNet forward to ~15–30 ms on the Orin Nano.

---

## Reference Image Guidelines

- **Quantity**: minimum 10 images; 20–50 is ideal for stable covariance estimates.
- **Consistency**: uniform lighting, consistent camera angle, leaf approximately
  centred in frame.
- **Alignment matters**: PaDiM is **not** rotation- or translation-invariant. If
  the leaf shifts noticeably between shots, the model will flag the misaligned
  regions as anomalous. For variable setups, pre-register images first (e.g. crop
  to the leaf bounding box) before training and inference.
- **Content**: only healthy, defect-free leaves in `data/reference/`.

---

## Choosing Threshold and Color Scale

Run inference on a small validation set that includes both good and defective
leaves. Then:

1. Examine the score distribution (the `score=` values printed by `detect.py`).
2. Pick `--vmin` / `--vmax` so that the normal score range maps to the cool
   (blue/green) end of the colormap and the anomalous range to red.
3. Set `--threshold` to the score value that separates the two populations in
   the histogram.

Fixing `--vmin` and `--vmax` is important for consistent visualizations: without
it, auto-scaling makes every image look maximally red, masking relative severity.

---

## Reference

Defard, T., Setkov, A., Loesch, A., Audigier, R. (2020).
*PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization.*
ICPR 2020. https://arxiv.org/abs/2011.08785
