"""Segmentation wrapper around RFDETRSegPreview for the masked PaDiM pipeline.

The class isolates the heavy `rfdetr` import and exposes two operations:
    extract_mask(frame)  — returns a single boolean mask (union of all
                            detected instances), or None if nothing
                            exceeds the min_area threshold.
    apply_mask(frame, mask, fill)
                          — composites the original pixels where mask is True
                            over a constant-fill background.
"""

from pathlib import Path
from typing import Optional

import numpy as np

try:
    from rfdetr import RFDETRSegPreview
except ImportError as e:
    raise ImportError(
        "rfdetr is not installed. Install with: pip install rfdetr supervision"
    ) from e


class Segmenter:
    """Thin wrapper around RFDETRSegPreview for video-frame masking."""

    def __init__(
        self,
        weights_path: Path,
        device: str = "cuda",
        threshold: float = 0.75,
        min_area: int = 200,
    ) -> None:
        self.device = device
        self.threshold = threshold
        self.min_area = min_area
        self.model = RFDETRSegPreview(
            pretrained=True, pretrain_weights=str(weights_path)
        )

    def extract_mask(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Run segmentation and return the union of all instance masks.

        Returns:
            (H, W) boolean mask, or None when no mask exceeds min_area.
        """
        detections = self.model.predict(
            frame_bgr,
            device=self.device,
            return_masks=True,
            threshold=self.threshold,
        )
        masks = getattr(detections, "mask", None)
        if masks is None or len(detections) == 0:
            return None
        union = np.any(masks.astype(bool), axis=0)
        if union.sum() < self.min_area:
            return None
        return union

    @staticmethod
    def apply_mask(
        frame_bgr: np.ndarray,
        mask: np.ndarray,
        fill: int = 255,
    ) -> np.ndarray:
        """Composite: original pixels where mask is True, `fill` colour elsewhere."""
        out = np.full_like(frame_bgr, fill)
        out[mask] = frame_bgr[mask]
        return out
