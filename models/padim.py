"""PaDiM: Patch Distribution Modeling for anomaly detection and localization.

Reference: Defard et al., "PaDiM: a Patch Distribution Modeling Framework for
Anomaly Detection and Localization", ICPR 2020 (https://arxiv.org/abs/2011.08785).

Self-contained copy for standalone deployment — does not depend on leaf_anomaly/src/.
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
    # Convert to grayscale and replicate to 3 channels so ResNet18 receives the
    # correct input shape while being blind to colour — shapes and textures only.
    transforms.Grayscale(num_output_channels=3),
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
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)

        base = self._features[FEATURE_LAYERS[0]]
        target_hw = base.shape[-2:]
        tensors = [base]
        for name in FEATURE_LAYERS[1:]:
            tensors.append(F.interpolate(self._features[name], size=target_hw, mode="nearest"))
        result = torch.cat(tensors, dim=1)
        self._features.clear()
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
        model.load("padim_leaf.pkl")
        heatmap, score = model.predict_pil(pil_image)
    """

    def __init__(self, device: Optional[str] = None, seed: int = 42) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.seed = seed
        self.extractor = FeatureExtractor().to(self.device).eval()

        self.mean: Optional[torch.Tensor] = None
        self.inv_cov: Optional[torch.Tensor] = None
        self.feat_idx: Optional[list[int]] = None
        self.feat_h: Optional[int] = None
        self.feat_w: Optional[int] = None

    def _load_image(self, path: Path) -> torch.Tensor:
        """Load and preprocess one image from disk."""
        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            raise RuntimeError(f"unreadable image {path}: {e}") from e
        return _transform(img)

    def _infer_tensor(self, tensor: torch.Tensor) -> tuple[np.ndarray, float]:
        """Run Mahalanobis inference on a preprocessed (1, C, H, W) tensor."""
        if self.mean is None or self.inv_cov is None:
            raise RuntimeError("Call fit() or load() before predict.")

        feats = self.extractor(tensor)
        feats = feats[0, self.feat_idx, :, :]
        feats = feats.reshape(N_FEATURES_REDUCED, -1).T

        diff = feats - self.mean
        left = torch.einsum("pi,pij->pj", diff, self.inv_cov)
        d_sq = torch.einsum("pi,pi->p", left, diff)
        d_sq = torch.clamp(d_sq, min=0.0)

        dist_map = torch.sqrt(d_sq).reshape(1, 1, self.feat_h, self.feat_w)
        dist_map = F.interpolate(dist_map, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        dist_map = _gaussian_blur(dist_map, kernel_size=9, sigma=4.0)
        heatmap = dist_map.squeeze().cpu().numpy().astype(np.float32)
        return heatmap, float(heatmap.max())

    @torch.no_grad()
    def predict(self, image_path: Path) -> tuple[np.ndarray, float]:
        """Run inference on a single image file.

        Returns:
            heatmap: (224, 224) float32 array of smoothed Mahalanobis distances.
            score:   Maximum value — usable as an image-level anomaly indicator.
        """
        tensor = self._load_image(image_path).unsqueeze(0).to(self.device)
        return self._infer_tensor(tensor)

    @torch.no_grad()
    def predict_pil(self, img: Image.Image) -> tuple[np.ndarray, float]:
        """Run inference on an in-memory PIL RGB image.

        Returns:
            heatmap: (224, 224) float32 array of smoothed Mahalanobis distances.
            score:   Maximum value — usable as an image-level anomaly indicator.
        """
        tensor = _transform(img).unsqueeze(0).to(self.device)
        return self._infer_tensor(tensor)

    @torch.no_grad()
    def fit(self, image_paths: list[Path], batch_size: int = 4) -> None:
        """Fit PaDiM on reference (normal) images using running sums."""
        print(f"[fit] Training on {len(image_paths)} images  (device={self.device})")

        sum_x: Optional[np.ndarray] = None
        sum_xxT: Optional[np.ndarray] = None
        n_samples = 0
        n_skipped = 0

        for i in range(0, len(image_paths), batch_size):
            chunk = image_paths[i: i + batch_size]
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
            feats = self.extractor(batch)
            B, C, H, W = feats.shape

            if sum_x is None:
                self.feat_h, self.feat_w = H, W
                self.feat_idx = random.Random(self.seed).sample(range(C), N_FEATURES_REDUCED)
                sum_x = np.zeros((H * W, N_FEATURES_REDUCED), dtype=np.float32)
                sum_xxT = np.zeros((H * W, N_FEATURES_REDUCED, N_FEATURES_REDUCED), dtype=np.float32)

            feats = feats[:, self.feat_idx, :, :]
            batch_np = (
                feats.permute(2, 3, 0, 1)
                     .reshape(H * W, B, N_FEATURES_REDUCED)
                     .cpu().numpy()
            )
            sum_x += batch_np.sum(axis=1)
            sum_xxT += np.einsum("pni,pnj->pij", batch_np, batch_np)
            n_samples += B
            print(f"[fit]   {min(i + batch_size, len(image_paths))}/{len(image_paths)} images processed")

        if n_samples == 0:
            raise RuntimeError("No valid reference images could be loaded.")
        if n_skipped:
            print(f"[fit] Skipped {n_skipped} unreadable image(s).")

        mean = sum_x / n_samples
        cov = (sum_xxT - n_samples * np.einsum("pi,pj->pij", mean, mean)) / max(n_samples - 1, 1)
        del sum_x, sum_xxT
        cov += 0.01 * np.eye(N_FEATURES_REDUCED, dtype=np.float32)[np.newaxis]
        inv_cov = np.linalg.inv(cov)

        self.mean = torch.from_numpy(mean.astype(np.float32)).to(self.device)
        self.inv_cov = torch.from_numpy(inv_cov.astype(np.float32)).to(self.device)
        print(f"[fit] Done. Feature grid: {self.feat_h}×{self.feat_w}, {N_FEATURES_REDUCED} features.")

    def save(self, path: Path) -> None:
        """Persist model statistics to a .pkl file (backbone not saved)."""
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
        """Load model statistics from a .pkl file."""
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
