"""CLI: run PaDiM inference on a single image or a folder of images.

Usage:
    python src/detect.py --model models/padim.pkl --input data/test --output output \\
        [--device cuda] [--threshold 0.3] [--vmin 0 --vmax 50] [--comparison]
"""

import argparse
import sys
import time
from pathlib import Path

# Allow imports from the same src/ directory when invoked as a script
sys.path.insert(0, str(Path(__file__).parent))

from padim import PaDiM
from visualize import make_comparison, make_overlay

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def process_image(
    model: PaDiM,
    image_path: Path,
    output_dir: Path,
    threshold: float | None,
    vmin: float | None,
    vmax: float | None,
    comparison: bool,
) -> float:
    """Infer on one image, write overlay (and optionally comparison). Returns latency in ms."""
    t0 = time.perf_counter()
    heatmap, score = model.predict(image_path)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    stem = image_path.stem
    make_overlay(
        image_path, heatmap,
        output_dir / f"{stem}_overlay.png",
        threshold=threshold, vmin=vmin, vmax=vmax,
    )
    if comparison:
        make_comparison(
            image_path, heatmap,
            output_dir / f"{stem}_comparison.png",
            score=score, vmin=vmin, vmax=vmax,
        )

    print(f"[detect] {image_path.name}: score={score:.2f}  latency={latency_ms:.1f} ms")
    return latency_ms


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PaDiM inference on leaf images."
    )
    parser.add_argument("--model", type=Path, required=True,
                        help="Path to trained .pkl model file.")
    parser.add_argument("--input", type=Path, required=True,
                        help="Input image file or folder of images.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output folder for result images.")
    parser.add_argument("--device", type=str, default=None,
                        help="Compute device: 'cuda' or 'cpu' (auto-detected if omitted).")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Heatmap mask threshold: pixels below this score are not highlighted.")
    parser.add_argument("--vmin", type=float, default=None,
                        help="Heatmap color-scale lower bound (fix across images for comparability).")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Heatmap color-scale upper bound.")
    parser.add_argument("--comparison", action="store_true",
                        help="Also save side-by-side Original | Heatmap | Overlay panels.")
    args = parser.parse_args()

    model = PaDiM(device=args.device)
    model.load(args.model)

    args.output.mkdir(parents=True, exist_ok=True)

    if args.input.is_dir():
        image_paths = sorted(
            p for p in args.input.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            print(f"[detect] ERROR: No images found in {args.input}")
            raise SystemExit(1)
    else:
        image_paths = [args.input]

    print(f"[detect] Processing {len(image_paths)} image(s) → {args.output}")

    latencies: list[float] = []
    failures = 0
    for path in image_paths:
        try:
            latencies.append(process_image(
                model, path, args.output,
                threshold=args.threshold,
                vmin=args.vmin,
                vmax=args.vmax,
                comparison=args.comparison,
            ))
        except Exception as e:
            print(f"[detect] ERROR on {path.name}: {e}")
            failures += 1

    if failures:
        print(f"[detect] {failures} image(s) failed.")
    if len(latencies) > 1:
        print(f"[detect] Average latency: {sum(latencies) / len(latencies):.1f} ms/image "
              f"over {len(latencies)} images")


if __name__ == "__main__":
    main()
