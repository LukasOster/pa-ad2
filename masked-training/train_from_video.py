"""Train PaDiM on a video, using a segmentation model to mask each frame.

Workflow:
    1. Iterate over the video at --frame-stride.
    2. For each sampled frame, run segmentation; if a sufficient mask is
       found, composite the frame against a white background and save it
       to --frames-dir as `frame_NNNNNN.png`.
    3. After extraction, call PaDiM.fit() on the saved frames and persist
       the model with PaDiM.save().
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2

# Reuse the self-contained PaDiM copy from models/
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "models"))
from padim import PaDiM  # noqa: E402

# Local import for the segmenter wrapper
sys.path.insert(0, str(Path(__file__).parent))
from segmenter import Segmenter  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Train masked PaDiM from a video.")
    p.add_argument("--video", type=Path, required=True,
                   help="Input video file.")
    p.add_argument("--seg-weights", type=Path, required=True, dest="seg_weights",
                   help="Path to RFDETRSegPreview checkpoint (.pth).")
    p.add_argument("--out", type=Path, required=True,
                   help="Output path for trained PaDiM model (.pkl).")
    p.add_argument("--threshold", type=float, default=0.75,
                   help="Segmentation confidence threshold (default 0.75).")
    p.add_argument("--min-area", type=int, default=200, dest="min_area",
                   help="Skip frames whose union mask covers fewer pixels than this.")
    p.add_argument("--frame-stride", type=int, default=5, dest="frame_stride",
                   help="Sample every Nth frame from the video (default 5).")
    p.add_argument("--batch-size", type=int, default=4, dest="batch_size",
                   help="Batch size for PaDiM feature extraction (default 4).")
    p.add_argument("--frames-dir", type=Path,
                   default=_REPO_ROOT / "masked-training" / "training_frames",
                   dest="frames_dir",
                   help="Directory where masked training frames are written.")
    p.add_argument("--show", action="store_true",
                   help="Display masked frames during extraction (q to abort).")
    p.add_argument("--device", type=str, default=None,
                   help="cuda or cpu; auto-detected if omitted.")
    args = p.parse_args()

    if not args.video.is_file():
        print(f"[train] ERROR: video not found: {args.video}")
        raise SystemExit(1)
    if not args.seg_weights.is_file():
        print(f"[train] ERROR: seg-weights not found: {args.seg_weights}")
        raise SystemExit(1)

    # Recreate frames-dir so a new run never mixes with stale data
    if args.frames_dir.exists():
        shutil.rmtree(args.frames_dir)
    args.frames_dir.mkdir(parents=True)

    seg_device = "cpu" if args.device == "cpu" else "cuda"
    print(f"[train] Loading segmenter (device={seg_device})...")
    segmenter = Segmenter(
        weights_path=args.seg_weights,
        device=seg_device,
        threshold=args.threshold,
        min_area=args.min_area,
    )

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"[train] ERROR: failed to open video: {args.video}")
        raise SystemExit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    print(f"[train] Reading {args.video} ({total_frames} frames), "
          f"sampling every {args.frame_stride} frame(s)...")

    frame_idx = 0
    saved = 0
    skipped = 0
    aborted = False
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % args.frame_stride == 0:
                mask = segmenter.extract_mask(frame)
                if mask is not None:
                    masked = segmenter.apply_mask(frame, mask, fill=255)
                    out_path = args.frames_dir / f"frame_{saved:06d}.png"
                    cv2.imwrite(str(out_path), masked)
                    saved += 1
                    if args.show:
                        cv2.imshow("masked-train", masked)
                        if (cv2.waitKey(1) & 0xFF) == ord("q"):
                            print("[train] Aborted by user.")
                            aborted = True
                            break
                else:
                    skipped += 1
            frame_idx += 1
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()

    print(f"[train] Extraction done: {saved} masked frames saved, "
          f"{skipped} sampled frames skipped (mask too small / not found).")

    if saved == 0:
        print("[train] ERROR: no usable frames extracted — nothing to train on.")
        raise SystemExit(1)
    if aborted:
        print("[train] Continuing with the frames captured before abort.")

    image_paths = sorted(args.frames_dir.glob("*.png"))
    print(f"[train] Fitting PaDiM on {len(image_paths)} frames...")
    model = PaDiM(device=args.device)
    model.fit(image_paths, batch_size=args.batch_size)
    model.save(args.out)
    print(f"[train] Done. Model saved → {args.out}")


if __name__ == "__main__":
    main()
