"""Heatmap overlay and side-by-side comparison output for PaDiM results."""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def _colorize(
    heatmap: np.ndarray,
    vmin: Optional[float],
    vmax: Optional[float],
) -> np.ndarray:
    """Normalize heatmap to [0, 255] and apply COLORMAP_JET.

    Fixing vmin/vmax across images ensures colors are comparable; without it,
    auto-scaling makes even normal images look red.
    """
    lo = vmin if vmin is not None else heatmap.min()
    hi = vmax if vmax is not None else heatmap.max()
    norm = np.clip((heatmap - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)


def make_overlay(
    image_path: Path,
    heatmap: np.ndarray,
    out_path: Path,
    threshold: Optional[float] = None,
    alpha: float = 0.5,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Blend the anomaly heatmap over the original image and save.

    Args:
        image_path: Source image (any resolution — will be resized to 224×224).
        heatmap:    (224, 224) float32 Mahalanobis distance map.
        out_path:   Destination file path.
        threshold:  When set, pixels with score < threshold are not highlighted.
        alpha:      Heatmap blend strength (0 = original only, 1 = heatmap only).
        vmin:       Lower bound for color normalization.
        vmax:       Upper bound for color normalization.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img = cv2.resize(img, (224, 224))
    colored = _colorize(heatmap, vmin, vmax)

    if threshold is not None:
        # Only overlay where the score exceeds the threshold
        mask = (heatmap >= threshold)
        overlay = img.copy()
        blended = cv2.addWeighted(img, 1 - alpha, colored, alpha, 0)
        overlay[mask] = blended[mask]
    else:
        overlay = cv2.addWeighted(img, 1 - alpha, colored, alpha, 0)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


def make_comparison(
    image_path: Path,
    heatmap: np.ndarray,
    out_path: Path,
    score: float,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Save a side-by-side panel: Original | Heatmap | Overlay, with score label.

    Args:
        image_path: Source image.
        heatmap:    (224, 224) float32 Mahalanobis distance map.
        out_path:   Destination file path.
        score:      Image-level anomaly score burned into the top-left corner.
        vmin:       Lower bound for color normalization.
        vmax:       Upper bound for color normalization.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img = cv2.resize(img, (224, 224))
    colored = _colorize(heatmap, vmin, vmax)
    overlay = cv2.addWeighted(img, 0.5, colored, 0.5, 0)

    panel = np.hstack([img, colored, overlay])
    cv2.putText(
        panel,
        f"score: {score:.2f}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)
