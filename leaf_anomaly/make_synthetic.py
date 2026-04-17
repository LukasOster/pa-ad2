"""Generate synthetic leaf images for end-to-end testing without real data.

Produces:
  data/reference/  — 20 normal (healthy) leaf images
  data/test/       —  5 normal + 3×3 anomalous (hole / spot / deformation)

Run from the leaf_anomaly/ directory:
    python make_synthetic.py
"""

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

OUTPUT_REF = Path("data/reference")
OUTPUT_TEST = Path("data/test")
IMG_SIZE = 224


def _draw_leaf_base(draw: ImageDraw.Draw, rng: random.Random) -> tuple[int, int, int, int]:
    """Draw a green elliptical leaf body and return (cx, cy, rx, ry)."""
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    rx = rng.randint(70, 90)
    ry = rng.randint(50, 70)
    g = rng.randint(120, 160)
    green = (rng.randint(40, 70), g, rng.randint(30, 60))
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=green)
    # Central vein
    vein = (max(0, green[0] - 15), max(0, green[1] - 25), max(0, green[2] - 10))
    draw.line([(cx - rx + 5, cy), (cx + rx - 5, cy)], fill=vein, width=2)
    # Lateral veins
    for i in range(1, 5):
        y_off = i * ry // 5
        x_off = int(rx * math.sqrt(max(0.0, 1 - (y_off / ry) ** 2)))
        for sign in (1, -1):
            draw.line([(cx, cy), (cx + x_off, cy + sign * y_off)], fill=vein, width=1)
    return cx, cy, rx, ry


def make_normal_leaf(seed: int) -> Image.Image:
    """Create a synthetic healthy leaf image."""
    rng = random.Random(seed)
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(90, 130, 70))
    _draw_leaf_base(ImageDraw.Draw(img), rng)
    return img.filter(ImageFilter.GaussianBlur(radius=1))


def make_anomalous_leaf(seed: int, anomaly_type: str) -> Image.Image:
    """Create a synthetic anomalous leaf image.

    Args:
        seed:         Random seed.
        anomaly_type: 'hole' | 'spot' | 'deformation'
    """
    img = make_normal_leaf(seed)
    rng = random.Random(seed + 1000)
    draw = ImageDraw.Draw(img)
    cx, cy, rx, ry = IMG_SIZE // 2, IMG_SIZE // 2, 80, 60

    if anomaly_type == "hole":
        hx = cx + rng.randint(-30, 30)
        hy = cy + rng.randint(-20, 20)
        r = rng.randint(10, 18)
        draw.ellipse([hx - r, hy - r, hx + r, hy + r], fill=(25, 15, 5))

    elif anomaly_type == "spot":
        for _ in range(rng.randint(4, 8)):
            sx = cx + rng.randint(-45, 45)
            sy = cy + rng.randint(-35, 35)
            sr = rng.randint(5, 14)
            spot = (rng.randint(140, 180), rng.randint(95, 130), rng.randint(15, 45))
            draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=spot)

    elif anomaly_type == "deformation":
        # Simulate a torn/missing section along the right edge
        pts = []
        for deg in range(0, 360, 8):
            r = rx + rng.randint(-4, 4)
            if 270 <= deg <= 360:
                r = int(r * rng.uniform(0.25, 0.60))
            rad = math.radians(deg)
            pts.append((cx + r * math.cos(rad), cy + (ry / rx) * r * math.sin(rad)))
        # Redraw background color over the torn region
        draw.polygon(pts, fill=(90, 130, 70))

    return img.filter(ImageFilter.GaussianBlur(radius=1))


def main() -> None:
    OUTPUT_REF.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEST.mkdir(parents=True, exist_ok=True)

    n_ref = 20
    print(f"[synthetic] Generating {n_ref} normal reference images → {OUTPUT_REF}")
    for i in range(n_ref):
        make_normal_leaf(seed=i).save(OUTPUT_REF / f"normal_{i:03d}.png")

    print(f"[synthetic] Generating test images → {OUTPUT_TEST}")
    for i in range(5):
        make_normal_leaf(seed=100 + i).save(OUTPUT_TEST / f"normal_{i:03d}.png")

    n_anomaly = 0
    for atype in ("hole", "spot", "deformation"):
        for i in range(3):
            make_anomalous_leaf(seed=200 + i, anomaly_type=atype).save(
                OUTPUT_TEST / f"{atype}_{i:03d}.png"
            )
            n_anomaly += 1

    total_test = 5 + n_anomaly
    print(f"[synthetic] Done. {n_ref} reference + {total_test} test images written.")


if __name__ == "__main__":
    main()
