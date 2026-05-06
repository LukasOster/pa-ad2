#!/usr/bin/env python3
"""Live training and anomaly detection for leaf inspection.

State machine
─────────────
  LIVE      — raw camera stream; press R to start a recording session
  RECORDING — saves frames for RECORD_SECONDS; countdown shown on screen
  TRAINING  — PaDiM.fit() runs on captured frames; status shown on screen
  DETECT    — stream shown with anomaly heatmap overlay; press R to retrain

Key bindings
────────────
  R  start recording (from LIVE or DETECT)
  Q  quit

On startup the script checks for an existing model at MODEL_PATH.  If one is
found it loads it and enters DETECT mode immediately so you don't have to
retrain every time.
"""

import shutil
import sys
import time
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import imagingcontrol4 as ic4

# ── path setup ───────────────────────────────────────────────────────────────
# Import PaDiM from the self-contained models/ copy so this script can be
# deployed alongside models/ on any machine without the full repo.
_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR / "models"))
from padim import PaDiM  # noqa: E402

# ── tunables ─────────────────────────────────────────────────────────────────
RECORD_SECONDS = 10          # length of one recording session
FRAME_INTERVAL = 0.2         # seconds between saved frames → 5 fps = 50 frames
TRAINING_DIR   = _SCRIPT_DIR / "training_frames"
MODEL_PATH     = _SCRIPT_DIR / "models" / "padim_live.pkl"
BATCH_SIZE     = 4
HEATMAP_VMIN   = 0.0
HEATMAP_VMAX   = 50.0        # tune: scores above this show as pure red
HEATMAP_ALPHA  = 0.5
WINDOW_NAME    = "Leaf Anomaly Detector"
# ─────────────────────────────────────────────────────────────────────────────


class State(Enum):
    LIVE      = auto()
    RECORDING = auto()
    TRAINING  = auto()
    DETECT    = auto()


# ── camera helpers ────────────────────────────────────────────────────────────

def open_camera() -> tuple[ic4.Grabber, ic4.SnapSink]:
    """Open the first detected IC4 device at its maximum resolution and return (grabber, sink)."""
    devices = ic4.DeviceEnum.devices()
    if not devices:
        raise RuntimeError("No cameras detected by imagingcontrol4.")
    grabber = ic4.Grabber()
    grabber.device_open(devices[0])

    # Push to the highest resolution the sensor supports before starting the stream.
    prop_map = grabber.device_property_map
    try:
        max_w = prop_map.find_integer(ic4.PropId.WIDTH).maximum
        max_h = prop_map.find_integer(ic4.PropId.HEIGHT).maximum
        prop_map.set_value(ic4.PropId.WIDTH, max_w)
        prop_map.set_value(ic4.PropId.HEIGHT, max_h)
        print(f"[camera] Resolution set to {max_w}×{max_h}.")
    except Exception as e:
        print(f"[camera] Could not set max resolution ({e}), using camera default.")

    sink = ic4.SnapSink()
    grabber.stream_setup(sink, setup_option=ic4.StreamSetupOption.ACQUISITION_START)
    print("[camera] Opened and streaming.")
    return grabber, sink


def close_camera(grabber: ic4.Grabber) -> None:
    """Stop streaming and close the device."""
    try:
        grabber.stream_stop()
    except Exception:
        pass
    try:
        grabber.device_close()
    except Exception:
        pass
    print("[camera] Closed.")


def grab_frame(sink: ic4.SnapSink) -> np.ndarray | None:
    """Snap one frame and return it as a numpy array, or None on error."""
    try:
        image = sink.snap_single(1000)
        return image.numpy_wrap().copy()
    except ic4.IC4Exception as ex:
        print("[camera] snap error:", getattr(ex, "message", str(ex)))
        return None


# ── frame helpers ─────────────────────────────────────────────────────────────

def to_bgr(frame: np.ndarray) -> np.ndarray:
    """Ensure the frame is a 3-channel BGR array suitable for imshow/imencode."""
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 1:
        return cv2.cvtColor(frame.squeeze(-1), cv2.COLOR_GRAY2BGR)
    return frame


def to_pil(frame: np.ndarray) -> Image.Image:
    """Convert a raw camera frame (any channel layout) to PIL RGB."""
    return Image.fromarray(cv2.cvtColor(to_bgr(frame), cv2.COLOR_BGR2RGB))


def draw_label(frame: np.ndarray, text: str,
               pos: tuple[int, int] = (10, 30),
               color: tuple[int, int, int] = (255, 255, 255),
               scale: float = 0.9) -> np.ndarray:
    """Burn a text label with a dark outline onto a copy of frame."""
    out = frame.copy()
    cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    return out


def apply_heatmap_overlay(
    bgr: np.ndarray,
    heatmap: np.ndarray,
    score: float,
    vmin: float,
    vmax: float,
    alpha: float,
) -> np.ndarray:
    """Blend a Mahalanobis distance map over a BGR frame and burn in the score."""
    h, w = bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    norm = np.clip((heatmap_resized - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    result = cv2.addWeighted(bgr, 1.0 - alpha, colored, alpha, 0)
    result = draw_label(result, f"anomaly score: {score:.1f}")
    result = draw_label(result, "Press R to retrain",
                        pos=(10, h - 10), color=(180, 180, 180), scale=0.6)
    return result


# ── main loop ─────────────────────────────────────────────────────────────────

def run() -> None:
    ic4.Library.init()
    grabber, sink = open_camera()

    # Try to load an existing model so the user can skip straight to detect mode.
    model: PaDiM | None = None
    state = State.LIVE
    if MODEL_PATH.exists():
        try:
            model = PaDiM()
            model.load(MODEL_PATH)
            state = State.DETECT
            print("[init] Existing model loaded — starting in DETECT mode.")
        except Exception as e:
            print(f"[init] Could not load model ({e}), starting in LIVE mode.")
            model = None

    recording_start: float = 0.0
    last_frame_save: float = 0.0
    frame_index: int = 0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = grab_frame(sink)
            if frame is None:
                continue

            bgr = to_bgr(frame)

            # ── per-state logic ───────────────────────────────────────────────

            if state == State.LIVE:
                display = draw_label(bgr, "Press R to record reference frames",
                                     color=(200, 200, 200))

            elif state == State.RECORDING:
                elapsed = time.time() - recording_start
                remaining = max(0.0, RECORD_SECONDS - elapsed)

                now = time.time()
                if now - last_frame_save >= FRAME_INTERVAL:
                    path = TRAINING_DIR / f"frame_{frame_index:04d}.png"
                    cv2.imwrite(str(path), frame)
                    frame_index += 1
                    last_frame_save = now

                display = draw_label(
                    bgr,
                    f"Recording {remaining:.1f}s  ({frame_index} frames saved)",
                    color=(0, 0, 255),
                )

                if elapsed >= RECORD_SECONDS:
                    print(f"[record] Saved {frame_index} frames to {TRAINING_DIR}")
                    state = State.TRAINING

            elif state == State.TRAINING:
                # Show a "please wait" frame before blocking on fit().
                cv2.imshow(WINDOW_NAME, draw_label(bgr, "Training... please wait", color=(0, 255, 255)))
                cv2.waitKey(1)

                image_paths = sorted(TRAINING_DIR.glob("*.png"))
                if not image_paths:
                    print("[train] No images found — returning to LIVE mode.")
                    state = State.LIVE
                    continue

                print(f"[train] Fitting PaDiM on {len(image_paths)} images...")
                model = PaDiM()
                model.fit(image_paths, batch_size=BATCH_SIZE)
                model.save(MODEL_PATH)
                print("[train] Done. Switching to DETECT mode.")
                state = State.DETECT
                continue  # redraw next iteration with detect overlay

            elif state == State.DETECT:
                if model is not None:
                    heatmap, score = model.predict_pil(to_pil(frame))
                    display = apply_heatmap_overlay(bgr, heatmap, score,
                                                    HEATMAP_VMIN, HEATMAP_VMAX, HEATMAP_ALPHA)
                else:
                    display = draw_label(bgr, "No model — press R to record", color=(0, 200, 255))

            else:
                display = bgr

            cv2.imshow(WINDOW_NAME, display)

            # ── keyboard ─────────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r") and state in (State.LIVE, State.DETECT):
                print("[record] Starting new recording session...")
                shutil.rmtree(TRAINING_DIR, ignore_errors=True)
                TRAINING_DIR.mkdir(parents=True, exist_ok=True)
                recording_start = time.time()
                last_frame_save = 0.0
                frame_index = 0
                state = State.RECORDING

    finally:
        close_camera(grabber)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
