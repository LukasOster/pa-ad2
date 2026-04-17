"""PaDiM: Patch Distribution Modeling for anomaly detection and localization.

Reference: Defard et al., "PaDiM: a Patch Distribution Modeling Framework for
Anomaly Detection and Localization", ICPR 2020 (https://arxiv.org/abs/2011.08785).
"""

from pathlib import Path
import pickle
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

IMG_SIZE = 224
FEATURE_LAYERS = ["layer1", "layer2", "layer3"]
N_FEATURES_REDUCED = 100

_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    # ImageNet normalization — without this ResNet produces nonsense activations
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class FeatureExtractor(nn.Module):
    """Frozen ResNet18 that returns concatenated features from FEATURE_LAYERS,
    captured via forward hooks.

    layer4 is intentionally omitted: its 7×7 spatial resolution is too coarse
    for pixel-level anomaly localization. The forward pass also stops after the
    deepest listed layer to skip layer4/avgpool/fc compute (~25% of the backbone).
    """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone.eval()
        self._features: dict[str, torch.Tensor] = {}
        for name in FEATURE_LAYERS:
            getattr(self.backbone, name).register_forward_hook(self._make_hook(name))
        for p in self.parameters():
            p.requires_grad = False

    def _make_hook(self, name: str):
        def hook(_module, _inp, out: torch.Tensor) -> None:
            self._features[name] = out
        return hook

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return concatenated multi-scale features of shape (B, C_total, H, W)."""
        self._features.clear()
        # Run the backbone through the deepest layer we need — skip layer4/avgpool/fc.
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)  # hook fires
        x = self.backbone.layer2(x)  # hook fires
        x = self.backbone.layer3(x)  # hook fires

        base = self._features[FEATURE_LAYERS[0]]
        target_hw = base.shape[-2:]
        tensors = [base]
        for name in FEATURE_LAYERS[1:]:
            tensors.append(F.interpolate(self._features[name], size=target_hw, mode="nearest"))
        result = torch.cat(tensors, dim=1)
        self._features.clear()  # release captured tensors for GC
        return result


def _gaussian_blur(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """Apply a separable Gaussian blur to a (1, 1, H, W) tensor via F.conv2d."""
    coords = torch.arange(kernel_size, dtype=torch.float32, device=x.device) - kernel_size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    pad = kernel_size // 2
    x = F.conv2d(x, g.view(1, 1, 1, kernel_size), padding=(0, pad))
    x = F.conv2d(x, g.view(1, 1, kernel_size, 1), padding=(pad, 0))
    return x


class PaDiM:
    """PaDiM anomaly detector.

    Workflow:
        model = PaDiM()
        model.fit(reference_image_paths)
        model.save("models/padim.pkl")

        model2 = PaDiM()
        model2.load("models/padim.pkl")
        heatmap, score = model2.predict(test_image_path)
    """

    def __init__(self, device: Optional[str] = None, seed: int = 42) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.seed = seed
        self.extractor = FeatureExtractor().to(self.device).eval()

        # Stored as torch tensors on self.device so predict() stays on the accelerator.
        self.mean: Optional[torch.Tensor] = None       # (H*W, 100)
        self.inv_cov: Optional[torch.Tensor] = None    # (H*W, 100, 100)
        self.feat_idx: Optional[list[int]] = None      # selected channel indices
        self.feat_h: Optional[int] = None
        self.feat_w: Optional[int] = None

    def _load_image(self, path: Path) -> torch.Tensor:
        """Load and preprocess one image. Raises RuntimeError on unreadable files."""
        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            raise RuntimeError(f"unreadable image {path}: {e}") from e
        return _transform(img)

    @torch.no_grad()
    def fit(self, image_paths: list[Path], batch_size: int = 4) -> None:
        """Fit PaDiM on reference (normal) images using running sums.

        Memory is O(H·W·C²) — independent of the number of reference images, so
        large reference sets (100s of images) fit on constrained targets.
        Corrupt or unreadable images are skipped with a warning.
        """
        print(f"[fit] Training on {len(image_paths)} images  (device={self.device})")

        sum_x: Optional[np.ndarray] = None     # (H*W, 100)
        sum_xxT: Optional[np.ndarray] = None   # (H*W, 100, 100)
        n_samples = 0
        n_skipped = 0

        for i in range(0, len(image_paths), batch_size):
            chunk = image_paths[i : i + batch_size]
            tensors: list[torch.Tensor] = []
            for p in chunk:
                try:
                    tensors.append(self._load_image(p))
                except RuntimeError as e:
                    print(f"[fit]   skipping {p.name}: {e}")
                    n_skipped += 1
            if not tensors:
                continue

            batch = torch.stack(tensors).to(self.device)
            feats = self.extractor(batch)          # (B, 448, H, W)
            B, C, H, W = feats.shape

            if sum_x is None:
                # Lazy init — shapes are only known after the first real batch.
                self.feat_h, self.feat_w = H, W
                # Deterministic random channel selection (must match at inference).
                self.feat_idx = random.Random(self.seed).sample(range(C), N_FEATURES_REDUCED)
                sum_x = np.zeros((H * W, N_FEATURES_REDUCED), dtype=np.float32)
                sum_xxT = np.zeros((H * W, N_FEATURES_REDUCED, N_FEATURES_REDUCED), dtype=np.float32)

            feats = feats[:, self.feat_idx, :, :]  # (B, 100, H, W)
            # (H*W, B, 100) — patch-major for per-patch statistics
            batch_np = (
                feats.permute(2, 3, 0, 1)
                     .reshape(H * W, B, N_FEATURES_REDUCED)
                     .cpu().numpy()
            )

            sum_x += batch_np.sum(axis=1)                                # (H*W, 100)
            sum_xxT += np.einsum("pni,pnj->pij", batch_np, batch_np)     # (H*W, 100, 100)
            n_samples += B

            print(f"[fit]   {min(i + batch_size, len(image_paths))}/{len(image_paths)} images processed")

        if n_samples == 0:
            raise RuntimeError("No valid reference images could be loaded.")
        if n_skipped:
            print(f"[fit] Skipped {n_skipped} unreadable image(s).")

        print(f"[fit] Estimating Gaussians for {self.feat_h * self.feat_w} patches (N={n_samples})...")

        mean = sum_x / n_samples
        # Sample covariance from running sums: Σ = (Σ xxᵀ − N·μμᵀ) / (N − 1)
        cov = (sum_xxT - n_samples * np.einsum("pi,pj->pij", mean, mean)) / max(n_samples - 1, 1)
        del sum_x, sum_xxT
        # Regularize Σ += 0.01·I so the matrix stays invertible even when N < C
        cov += 0.01 * np.eye(N_FEATURES_REDUCED, dtype=np.float32)[np.newaxis]
        inv_cov = np.linalg.inv(cov)

        self.mean = torch.from_numpy(mean.astype(np.float32)).to(self.device)
        self.inv_cov = torch.from_numpy(inv_cov.astype(np.float32)).to(self.device)

        print(f"[fit] Done. Feature grid: {self.feat_h}×{self.feat_w}, {N_FEATURES_REDUCED} features.")

    @torch.no_grad()
    def predict(self, image_path: Path) -> tuple[np.ndarray, float]:
        """Run inference on a single image.

        Returns:
            heatmap: (224, 224) float32 array of smoothed Mahalanobis distances.
            score:   Maximum value — usable as an image-level anomaly indicator.
        """
        if self.mean is None or self.inv_cov is None:
            raise RuntimeError("Call fit() or load() before predict().")

        img = self._load_image(image_path).unsqueeze(0).to(self.device)
        feats = self.extractor(img)                              # (1, 448, H, W)
        feats = feats[0, self.feat_idx, :, :]                    # (100, H, W)
        feats = feats.reshape(N_FEATURES_REDUCED, -1).T          # (H*W, 100)

        diff = feats - self.mean                                 # (H*W, 100)
        # Mahalanobis²: d² = diff @ Σ⁻¹ @ diffᵀ  (per patch)
        left = torch.einsum("pi,pij->pj", diff, self.inv_cov)    # (H*W, 100)
        d_sq = torch.einsum("pi,pi->p", left, diff)              # (H*W,)
        # Clamp before sqrt to guard against tiny negatives from float arithmetic
        d_sq = torch.clamp(d_sq, min=0.0)

        dist_map = torch.sqrt(d_sq).reshape(1, 1, self.feat_h, self.feat_w)
        dist_map = F.interpolate(dist_map, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        dist_map = _gaussian_blur(dist_map, kernel_size=9, sigma=4.0)
        heatmap = dist_map.squeeze().cpu().numpy().astype(np.float32)

        return heatmap, float(heatmap.max())

    def save(self, path: Path) -> None:
        """Persist model state to a .pkl file.

        The ResNet backbone is NOT saved — it reloads from the Torch cache
        automatically when PaDiM is instantiated.
        """
        if self.mean is None or self.inv_cov is None:
            raise RuntimeError("Model is empty — call fit() before save().")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "mean": self.mean.cpu().numpy(),
                "inv_cov": self.inv_cov.cpu().numpy(),
                "feat_idx": self.feat_idx,
                "feat_h": self.feat_h,
                "feat_w": self.feat_w,
            }, f)
        print(f"[save] Model saved → {path}")

    def load(self, path: Path) -> None:
        """Load model state from a .pkl file."""
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.feat_idx = data["feat_idx"]
        self.feat_h = data["feat_h"]
        self.feat_w = data["feat_w"]
        self.mean = torch.from_numpy(data["mean"]).to(self.device)
        self.inv_cov = torch.from_numpy(data["inv_cov"]).to(self.device)
        print(
            f"[load] Model loaded ← {path}  "
            f"(grid {self.feat_h}×{self.feat_w}, {len(self.feat_idx)} features)"
        )
