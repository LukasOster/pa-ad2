"""CLI: train PaDiM on a folder of reference (normal) leaf images.

Usage:
    python src/train.py --ref data/reference --out models/padim.pkl [--device cuda] [--batch-size 4]
"""

import argparse
import sys
from pathlib import Path

# Allow imports from the same src/ directory when invoked as a script
sys.path.insert(0, str(Path(__file__).parent))

from padim import PaDiM

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PaDiM anomaly detector on reference leaf images."
    )
    parser.add_argument("--ref", type=Path, required=True,
                        help="Folder containing reference (normal) images.")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output path for the trained model (.pkl).")
    parser.add_argument("--device", type=str, default=None,
                        help="Compute device: 'cuda' or 'cpu' (auto-detected if omitted).")
    parser.add_argument("--batch-size", type=int, default=4, dest="batch_size",
                        help="Batch size for feature extraction (default: 4).")
    parser.add_argument("--sanity-check", action="store_true", dest="sanity_check",
                        help="Run a quick self-check after training (requires synthetic test data).")
    args = parser.parse_args()

    if not args.ref.is_dir():
        print(f"[train] ERROR: {args.ref} is not a directory.")
        raise SystemExit(1)

    image_paths = collect_images(args.ref)
    if not image_paths:
        print(f"[train] ERROR: No images found in {args.ref}")
        raise SystemExit(1)

    print(f"[train] Found {len(image_paths)} reference images in {args.ref}")

    model = PaDiM(device=args.device)
    model.fit(image_paths, batch_size=args.batch_size)
    model.save(args.out)

    if args.sanity_check:
        _run_sanity_check(model, image_paths)

    print(f"[train] Complete.")


def _run_sanity_check(model: PaDiM, reference_paths: list[Path]) -> None:
    """Sanity-check: a reference image should score lower than a clearly anomalous one.

    Uses the first reference image as the 'normal' sample and checks that a
    synthetic high-noise image scores significantly higher.
    """
    import numpy as np
    from PIL import Image

    print("[sanity] Running sanity check...")

    ref_path = reference_paths[0]
    _, ref_score = model.predict(ref_path)

    # Create a pure noise image as a stand-in for a maximally anomalous input
    noise = (np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        Image.fromarray(noise).save(tmp.name)
        _, noise_score = model.predict(Path(tmp.name))
        os.unlink(tmp.name)

    status = "PASS" if noise_score > ref_score else "FAIL"
    print(f"[sanity] {status}  ref_score={ref_score:.2f}  noise_score={noise_score:.2f}")
    if status == "FAIL":
        print("[sanity] WARNING: noise image did not score higher than reference — "
              "check your reference images or model parameters.")


if __name__ == "__main__":
    main()
