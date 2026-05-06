"""Run masked PaDiM anomaly detection on a video file.

For each frame:
    1. Extract a segmentation mask via RFDETRSegPreview.
    2. Composite a masked frame (white background outside the mask).
    3. Run PaDiM on the masked frame to get a 224x224 anomaly heatmap.
    4. Resize the heatmap to the original resolution and blend it over the
       ORIGINAL (unmasked) frame as a JET-coloured overlay.
    5. Display the result and/or write it to an output video (mp4v).

If no segmentation mask is found in a frame, the frame is shown unmodified
with a `no mask` label.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "models"))
from padim import PaDiM  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from segmenter import Segmenter  # noqa: E402

WINDOW_NAME = "pa-ad2"


def _draw_label(frame: np.ndarray, text: str,
                pos: tuple[int, int] = (10, 30),
                color: tuple[int, int, int] = (255, 255, 255)) -> None:
    """Burn a text label with a dark outline directly into `frame`."""
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 1, cv2.LINE_AA)


def overlay_heatmap(
    frame_bgr: np.ndarray,
    heatmap_full: np.ndarray,
    score: float,
    vmin: float,
    vmax: float,
    alpha: float,
) -> np.ndarray:
    """Blend a JET-coloured heatmap over a BGR frame and burn the score in."""
    norm = np.clip((heatmap_full - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    out = cv2.addWeighted(frame_bgr, 1.0 - alpha, colored, alpha, 0)
    _draw_label(out, f"score: {score:.1f}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Run masked PaDiM on a video file.")
    p.add_argument("--video", type=Path, required=True,
                   help="Input video file.")
    p.add_argument("--seg-weights", type=Path, required=True, dest="seg_weights",
                   help="Path to RFDETRSegPreview checkpoint (.pth).")
    p.add_argument("--model", type=Path, required=True,
                   help="Trained PaDiM model (.pkl).")
    p.add_argument("--save-output", type=Path, default=None, dest="save_output",
                   help="Optional path to write the processed video (mp4v).")
    p.add_argument("--threshold", type=float, default=0.75,
                   help="Segmentation confidence threshold (default 0.75).")
    p.add_argument("--min-area", type=int, default=200, dest="min_area",
                   help="Treat union masks below this many pixels as 'no mask'.")
    p.add_argument("--vmin", type=float, default=0.0,
                   help="Heatmap colour-scale lower bound (default 0).")
    p.add_argument("--vmax", type=float, default=50.0,
                   help="Heatmap colour-scale upper bound (default 50).")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Heatmap blend strength (default 0.5).")
    p.add_argument("--mask-overlay-only", action="store_true", dest="mask_overlay_only",
                   help="Zero the heatmap outside the segmentation mask before "
                        "overlay (cosmetic — does not affect the score).")
    p.add_argument("--no-display", action="store_true", dest="no_display",
                   help="Skip the preview window (headless processing).")
    p.add_argument("--device", type=str, default=None,
                   help="cuda or cpu; auto-detected if omitted.")
    args = p.parse_args()

    if not args.video.is_file():
        print(f"[run] ERROR: video not found: {args.video}")
        raise SystemExit(1)
    if not args.seg_weights.is_file():
        print(f"[run] ERROR: seg-weights not found: {args.seg_weights}")
        raise SystemExit(1)
    if not args.model.is_file():
        print(f"[run] ERROR: model not found: {args.model}")
        raise SystemExit(1)

    seg_device = "cpu" if args.device == "cpu" else "cuda"
    print(f"[run] Loading segmenter (device={seg_device})...")
    segmenter = Segmenter(
        weights_path=args.seg_weights,
        device=seg_device,
        threshold=args.threshold,
        min_area=args.min_area,
    )
    print("[run] Loading PaDiM model...")
    model = PaDiM(device=args.device)
    model.load(args.model)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"[run] ERROR: failed to open video: {args.video}")
        raise SystemExit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer: cv2.VideoWriter | None = None
    if args.save_output is not None:
        args.save_output.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.save_output), fourcc, fps, (width, height))
        if not writer.isOpened():
            print(f"[run] ERROR: failed to open writer for {args.save_output}")
            raise SystemExit(1)
        print(f"[run] Writing output → {args.save_output}  "
              f"({width}×{height} @ {fps:.1f} fps)")

    if not args.no_display:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    frame_idx = 0
    latencies: list[float] = []
    print("[run] Processing... (press q in the preview window to abort)")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t0 = time.perf_counter()
            mask = segmenter.extract_mask(frame)
            if mask is None:
                display = frame.copy()
                _draw_label(display, "no mask", color=(180, 180, 180))
            else:
                masked = segmenter.apply_mask(frame, mask, fill=255)
                pil = Image.fromarray(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
                heatmap, score = model.predict_pil(pil)  # (224, 224) float32
                heatmap_full = cv2.resize(heatmap, (width, height),
                                          interpolation=cv2.INTER_LINEAR)
                if args.mask_overlay_only:
                    heatmap_full = heatmap_full * mask.astype(np.float32)
                display = overlay_heatmap(
                    frame, heatmap_full, score,
                    vmin=args.vmin, vmax=args.vmax, alpha=args.alpha,
                )
            latencies.append((time.perf_counter() - t0) * 1000.0)

            if writer is not None:
                writer.write(display)
            if not args.no_display:
                cv2.imshow(WINDOW_NAME, display)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    print("[run] Aborted by user.")
                    break
            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    if latencies:
        avg = sum(latencies) / len(latencies)
        print(f"[run] {frame_idx} frame(s) processed, "
              f"average latency: {avg:.1f} ms/frame")


if __name__ == "__main__":
    main()
