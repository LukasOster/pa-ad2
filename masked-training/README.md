# Masked PaDiM training & inference

PaDiM trained on segmentation-masked video frames. For each frame, an
`RFDETRSegPreview` model crops out the object of interest, the rest of
the frame is replaced with a white background, and PaDiM is trained or
applied only to that masked composite. This narrows the model's notion
of "normal" to the object itself.

## Requirements

```bash
pip install rfdetr supervision opencv-python pillow torch torchvision numpy
```

You also need a trained `RFDETRSegPreview` checkpoint (`.pth`).

## Train from a video

```bash
python train_from_video.py \
    --video path/to/video.mp4 \
    --seg-weights path/to/checkpoint_best_total.pth \
    --out ../models/padim_masked.pkl \
    --frame-stride 5
```

Optional flags:

| Flag | Default | Purpose |
|---|---|---|
| `--threshold` | `0.75` | Segmentation confidence threshold |
| `--min-area`  | `200`  | Skip frames whose union mask is smaller than this |
| `--frame-stride` | `5` | Sample every Nth frame from the video |
| `--batch-size` | `4`   | PaDiM feature-extraction batch size |
| `--frames-dir` | `masked-training/training_frames/` | Where masked frames are written (recreated each run) |
| `--show` | off | Preview masked frames during extraction (q to abort) |
| `--device` | auto | `cuda` or `cpu` |

## Run on a video

```bash
python run_pa-ad2_on_video_file.py \
    --video path/to/video.mp4 \
    --seg-weights path/to/checkpoint_best_total.pth \
    --model ../models/padim_masked.pkl \
    --save-output processed.mp4
```

Optional flags:

| Flag | Default | Purpose |
|---|---|---|
| `--vmin` / `--vmax` / `--alpha` | `0` / `50` / `0.5` | Heatmap colour scale & blend |
| `--threshold` / `--min-area` | `0.75` / `200` | Segmentation thresholds |
| `--mask-overlay-only` | off | Zero heatmap outside the mask before overlay (cosmetic; does not change scores) |
| `--no-display` | off | Skip preview window for headless runs |
| `--save-output` | – | Write the processed video as mp4v |
| `--device` | auto | `cuda` or `cpu` |

Frames where no segmentation mask is found are passed through unchanged
with a `no mask` label.
