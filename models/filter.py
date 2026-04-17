"""PaDiM leaf anomaly filter.

Applies the trained PaDiM model to incoming BGR frames and returns the frame
with the anomaly heatmap blended over it and the anomaly score printed in the
top-left corner.

Expected files next to this script:
    padim.py          — PaDiM implementation (self-contained copy)
    padim_leaf.pkl    — trained model statistics
"""

import os
import sys

import cv2
import numpy as np
from PIL import Image

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from padim import PaDiM

# ── colour-scale bounds ──────────────────────────────────────────────────────
# Keep these fixed so the colour interpretation is consistent across frames.
# A pixel with Mahalanobis distance ≥ VMAX is shown as pure red.
# Tune VMAX on a representative set of good/bad leaves (start with 50).
VMAX: float = 50.0
VMIN: float = 0.0
ALPHA: float = 0.5   # heatmap blend strength (0 = original only, 1 = heatmap only)

# ── load model once at import time ───────────────────────────────────────────
_model = PaDiM()
_model.load(os.path.join(_DIR, "padim_leaf.pkl"))


def _colorize(heatmap: np.ndarray) -> np.ndarray:
    """Map a float32 distance map to a BGR colour image via COLORMAP_JET."""
    norm = np.clip((heatmap - VMIN) / (VMAX - VMIN + 1e-8), 0.0, 1.0)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)


def apply_filter(frame: np.ndarray) -> np.ndarray:
    """Apply PaDiM anomaly detection to a BGR frame.

    Args:
        frame: Input image as a NumPy BGR array (any resolution).

    Returns:
        A copy of the frame (original resolution) with the anomaly heatmap
        blended over it and the anomaly score printed in the top-left corner.
    """
    # Normalise input to 3-channel BGR
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 1:
        frame = cv2.cvtColor(frame.squeeze(-1), cv2.COLOR_GRAY2BGR)

    orig_h, orig_w = frame.shape[:2]

    # PaDiM expects a PIL RGB image; it internally resizes to 224×224
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    heatmap, score = _model.predict_pil(pil_img)  # heatmap: (224, 224) float32

    # Scale heatmap back to original frame resolution for a full-size overlay
    heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    colored = _colorize(heatmap_resized)

    result = cv2.addWeighted(frame, 1.0 - ALPHA, colored, ALPHA, 0)

    # Burn anomaly score into the top-left corner
    label = f"anomaly score: {score:.1f}"
    cv2.putText(result, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(result, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 1, cv2.LINE_AA)

    return result
