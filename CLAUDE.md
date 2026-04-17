# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Leaf anomaly detector implementing **PaDiM** (Defard et al. 2020). Trains on reference images of healthy leaves and produces pixel-level Mahalanobis-distance heatmaps for test images. Target deployment is **NVIDIA Jetson Orin Nano/NX** (JetPack 6.x); development is on desktop Linux/macOS.

The entire application lives under `leaf_anomaly/`. The repo root holds only the spec (`CLAUDE_CODE_PROMPT.md`) and devcontainer setup.

`CLAUDE_CODE_PROMPT.md` is the authoritative spec for this project's behavior, CLI shape, and algorithmic details. When in doubt about requirements or constraints, read it first.

## Common commands

All commands run from `leaf_anomaly/`:

```bash
# Generate synthetic test data (20 reference + 14 test images)
python make_synthetic.py

# Train
python src/train.py --ref data/reference --out models/padim.pkl
python src/train.py --ref data/reference --out models/padim.pkl --device cuda --batch-size 8
python src/train.py --ref data/reference --out models/padim.pkl --sanity-check

# Detect (single image or folder)
python src/detect.py --model models/padim.pkl --input data/test --output output \
    --vmin 0 --vmax 50 --comparison

# Install on desktop
pip install -r requirements.txt
```

There is no test suite, linter config, or build step. The devcontainer ships Python 3.12 + Node 22 but no Python deps — install from `requirements.txt` before running.

**On Jetson, do not `pip install torch/torchvision`.** Use the JetPack-matching NVIDIA wheels and `sudo apt install python3-opencv`. See `leaf_anomaly/README.md` for details.

## Architecture

### Pipeline

`train.py`/`detect.py` are thin CLIs. All real work is in `src/padim.py`:

- **`FeatureExtractor`** — frozen ResNet18 with forward hooks on `FEATURE_LAYERS = ["layer1", "layer2", "layer3"]`. `forward()` runs stem+layer1+layer2+layer3 manually and stops; layer4/avgpool/fc are never executed (~25% FLOPs saved). Hooks populate `self._features`; the dict is cleared at start and end of each forward to release captured tensors.
- **`PaDiM.fit()`** — accumulates `sum_x` and `sum_xxT` (per-patch running sums) instead of holding all features in memory. Peak memory is **O(H·W·C²)**, independent of N — large reference sets fit on constrained targets. Covariance is recovered at the end as `Σ = (Σxxᵀ − N·μμᵀ) / (N−1)`, regularized with `0.01·I`, and inverted.
- **`PaDiM.predict()`** — entire Mahalanobis compute stays in torch (no numpy round-trip). `mean` and `inv_cov` are torch tensors on `self.device`; exported to numpy only when pickling.

### Critical invariants

- **`feat_idx` must be identical at fit and predict time.** It's generated via `random.Random(seed).sample(range(C), 100)` and saved in the `.pkl`. Changing `seed` or `N_FEATURES_REDUCED` breaks existing models.
- **ImageNet normalization is mandatory.** ResNet18 features are nonsensical without it.
- **Model persistence**: `.pkl` stores `mean`, `inv_cov`, `feat_idx`, `feat_h`, `feat_w` as numpy arrays. The ResNet backbone is deliberately **not** saved — it reloads from Torch's cache on next instantiation.
- **Corrupt images are skipped in `fit()`** (logged, counted) but raise in `predict()` — fit is batch-mode, predict is per-call.

### CLI import pattern

`train.py` and `detect.py` use `sys.path.insert(0, str(Path(__file__).parent))` so they can run as `python src/train.py` from the project root. This matches the exact CLI shape mandated by the spec. If refactoring toward a proper package, note that the spec's CLI invocations must remain working.

### Spec compliance

The spec explicitly prohibits: finetuning the backbone, alternative approaches (autoencoder, PatchCore), web UI, Docker. It requires: type hints throughout, docstrings on modules and functions, `[fit]`/`[load]`/`[detect]` status prefixes, and no dead code.
