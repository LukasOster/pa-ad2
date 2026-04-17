# Claude Code Prompt: Blatt-Anomaliedetektor (PaDiM) für Jetson Orin Nano

Baue einen vollständigen Anomaliedetektor für Blätter in Python. Der Detektor
wird auf einem oder mehreren Referenzbildern eines "idealen" Blattes trainiert
und erzeugt für neue Testbilder eine pixelgenaue Anomalie-Heatmap, die dem
Originalbild als Overlay angezeigt wird. Zielplattform: **NVIDIA Jetson Orin
Nano / NX** mit JetPack 6.x. Entwicklung auf Desktop (Linux oder macOS), Deployment auf Jetson.

## Technischer Ansatz

Nutze **PaDiM** (Patch Distribution Modeling, Defard et al. 2020):

1. Vortrainiertes **ResNet18** (ImageNet) als Feature-Extraktor, eingefroren — kein
   Finetuning.
2. Features aus `layer1`, `layer2`, `layer3` per Forward-Hooks abgreifen.
   `layer4` weglassen (zu grob für Pixel-Lokalisierung).
3. Feature-Maps auf die Auflösung von `layer1` hoch­interpolieren (nearest
   neighbor) und entlang der Kanalachse konkatenieren.
4. **PaDiM-Dimensionsreduktion**: aus den konkatenierten Kanälen (448 bei
   ResNet18) zufällig 100 auswählen. Random-Seed fest verdrahten, damit fit
   und predict identische Indizes benutzen.
5. Für jede Patch-Position `(i, j)` im Feature-Grid eine multivariate
   Gauß-Verteilung schätzen: Mittelwert `μ_ij ∈ R^100` und Kovarianzmatrix
   `Σ_ij ∈ R^{100×100}` über alle Referenzbilder. Regularisierung:
   `Σ += 0.01 * I`, damit sie auch bei wenigen Bildern (N < C) invertierbar bleibt.
6. Inverse Kovarianz `Σ⁻¹` einmal beim Training berechnen und zusammen mit `μ`
   speichern.
7. Bei Inferenz pro Patch **Mahalanobis-Distanz** `d = √((x-μ)ᵀ Σ⁻¹ (x-μ))`
   berechnen. Das ergibt ein Grid `H_f × W_f`, das per bilinearer Interpolation
   auf die Eingangs­auflösung (224×224) hochgesampelt und mit einem
   Gauß-Filter (kernel 9, sigma 4) geglättet wird.

## Projektstruktur

```
leaf_anomaly/
├── src/
│   ├── padim.py         # PaDiM-Klasse mit fit/predict/save/load
│   ├── visualize.py     # Heatmap-Overlay + Side-by-side-Vergleich
│   ├── train.py         # CLI: trainiert auf Referenzbild-Ordner
│   └── detect.py        # CLI: Inferenz auf Einzelbild oder Ordner
├── data/
│   ├── reference/       # Referenzbilder (nur normale Blätter)
│   └── test/            # Testbilder
├── models/              # Gespeicherte .pkl-Modelle
├── output/              # Generierte Overlays
├── requirements.txt
└── README.md
```

## Implementierungs-Details

### `src/padim.py`

- Konstanten: `IMG_SIZE = 224`, `FEATURE_LAYERS = ["layer1", "layer2", "layer3"]`,
  `N_FEATURES_REDUCED = 100`.
- ImageNet-Normalisierung nutzen (`mean=[0.485, 0.456, 0.406]`,
  `std=[0.229, 0.224, 0.225]`), sonst liefert ResNet Unsinn.
- Klasse `FeatureExtractor(nn.Module)`: lädt ResNet18 mit
  `ResNet18_Weights.DEFAULT`, setzt `.eval()`, registriert Forward-Hooks pro
  Layer, sammelt Outputs in einem Dict. Forward gibt den konkatenierten
  Feature-Tensor `(B, C_total, H, W)` zurück.
- Klasse `PaDiM` mit:
  - `__init__(device=None, seed=42)`: Device auto-detect (`cuda` falls
    verfügbar).
  - `fit(image_paths, batch_size=4)`: extrahiert Features, wählt `feat_idx`
    deterministisch via `random.Random(seed).sample(...)`, berechnet `mean`
    und `inv_cov`. Batch-Loop damit auch 50+ Referenzbilder auf dem Jetson
    reinpassen.
  - `predict(image_path) -> (heatmap: np.ndarray, score: float)`: liefert
    die 224×224-Heatmap und den Maximum-Score als Bild-Level-Indikator.
    Numerischer Safety-Clamp vor `sqrt` (`torch.clamp(d_sq, min=0.0)`).
  - `save(path)` / `load(path)`: picklt `mean`, `inv_cov`, `feat_idx`,
    `feat_h`, `feat_w`. Das ResNet selbst wird **nicht** gespeichert (kommt
    beim Laden automatisch aus dem Torch-Cache).
- Hilfsfunktion `_gaussian_blur(x, kernel_size, sigma)` als separabler
  Gauß-Filter mittels `F.conv2d`.

### `src/visualize.py`

- `make_overlay(image_path, heatmap, out_path, threshold=None, alpha=0.5,
  vmin=None, vmax=None)`: normalisiert Heatmap auf `[0,1]`, wendet
  `cv2.COLORMAP_JET` an, blendet sie pixelweise-transparent über das auf
  224×224 skalierte Original. `vmin`/`vmax` fixieren, damit Farben über
  mehrere Bilder hinweg vergleichbar bleiben (sonst macht die Auto-Skalierung
  auch normale Bilder rot).
- `make_comparison(image_path, heatmap, out_path, score, vmin=None,
  vmax=None)`: erzeugt `hstack` aus Original | Heatmap | Overlay und brennt
  den Score via `cv2.putText` ein — praktisch für Reports.

### `src/train.py` (CLI)

```
python src/train.py --ref data/reference --out models/padim.pkl [--device cuda] [--batch-size 4]
```

Validiert, dass der Input-Ordner Bilder enthält (`.jpg .jpeg .png .bmp .tif
.tiff .webp`), ruft `PaDiM.fit()` und `save()` auf.

### `src/detect.py` (CLI)

```
python src/detect.py --model models/padim.pkl --input data/test --output output \
    [--device cuda] [--threshold 0.3] [--vmin 0 --vmax 50] [--comparison]
```

Unterstützt Einzelbild oder Ordner. Misst Latenz pro Bild (`time.perf_counter`)
und gibt den Durchschnitt am Ende aus. Mit `--comparison` wird zusätzlich das
Side-by-side-Bild erzeugt.

### `requirements.txt`

```
torch>=2.0
torchvision>=0.15
numpy>=1.23
pillow>=9.0
opencv-python>=4.7
```

### `README.md`

Muss folgende Abschnitte enthalten:

1. **Kurze Projektbeschreibung** (1 Absatz: was es tut, welcher Ansatz).
2. **Installation auf dem Desktop**: `pip install -r requirements.txt`.
3. **Installation auf dem Jetson Orin Nano**:
   - JetPack 6.x vorausgesetzt.
   - PyTorch/torchvision nicht per pip installieren! Stattdessen die
     offiziellen **NVIDIA-Jetson-Wheels** von
     `https://developer.download.nvidia.com/compute/redist/jp/` verwenden
     (passend zur JetPack-Version).
   - `sudo apt install python3-opencv` bevorzugen (hardware-beschleunigt auf
     Jetson), statt `opencv-python` per pip.
   - Hinweis: Der Leser soll die genauen Wheel-URLs aus den aktuellen NVIDIA
     Jetson-Foren / dem JetPack-Release-Notes ziehen — URLs dürfen im README
     nicht hart kodiert werden, weil sie je JetPack variieren.
4. **Nutzung**: Die beiden CLI-Aufrufe oben mit kurzer Erklärung.
5. **Performance-Tipps für Jetson Orin Nano**:
   - `sudo nvpmodel -m 0` und `sudo jetson_clocks` für Maximalleistung.
   - Erwartete Latenz: ~50–150 ms pro Bild bei 224×224 auf Orin Nano
     (je nach Power-Mode). Der Bottleneck ist das ResNet-Forward.
   - Für noch mehr Speed: ResNet18 nach **ONNX** → **TensorRT** (FP16)
     konvertieren. Die Mahalanobis-Berechnung bleibt in PyTorch (klein).
6. **Hinweise zu Referenzbildern**:
   - Mindestens 10, idealerweise 20–50 Bilder.
   - Konsistente Beleuchtung, konsistenter Aufnahmewinkel, Blatt ungefähr
     zentriert. PaDiM ist **nicht** rotations-/translations-invariant —
     wenn das Blatt im Testbild stark verschoben ist, "sieht" das Modell
     überall Anomalien. Bei schwankenden Aufnahmen also vorher registrieren
     (z.B. auf die Blatt-Bounding-Box croppen).
7. **Wahl der Schwelle**: Auf einem kleinen Validation-Set mit guten und
   schlechten Blättern Histogramme der Scores anschauen, dann `vmin`/`vmax`
   und ggf. `--threshold` passend setzen.

## Qualitätsanforderungen

- Type-Hints durchgängig (`list[Path]`, `tuple[np.ndarray, float]` etc.).
- Docstrings auf Modul- und Funktionsebene.
- Sinnvolle Kommentare an den nicht-offensichtlichen Stellen (warum
  zufällige Kanalauswahl, warum +0.01·I, warum `layer4` weglassen).
- Kein Dead Code, kein `print`-Spam — nur sinnvolle Status-Ausgaben
  (`[fit] ...`, `[load] ...`).
- Funktioniert sowohl mit CUDA als auch rein auf CPU (für Dev-Tests).

## Optional (schön zu haben, falls Zeit)

- Ein `make_synthetic.py` im Projektwurzel, das ein paar synthetische
  Blattbilder (mit und ohne Anomalien wie Loch, Fleck, Deformation) mit
  PIL erzeugt. Ermöglicht End-to-End-Test ohne echte Daten.
- Minimaler Sanity-Check am Ende des Trainings: Inferenz auf einem
  Referenzbild sollte einen niedrigeren Score liefern als auf einem
  offensichtlich anomalen Bild.

## Referenzen

- Paper: Defard et al., "PaDiM: a Patch Distribution Modeling Framework for
  Anomaly Detection and Localization", ICPR 2020
  (`https://arxiv.org/abs/2011.08785`).
- NVIDIA Jetson PyTorch install guide (PyTorch for Jetson — Version).

## Nicht gewünscht

- Kein Finetuning des Backbones.
- Kein alternativer Ansatz (keine Autoencoder, kein PatchCore-Greedy-
  Coreset — bewusst PaDiM, weil speicherärmer auf dem Orin Nano und bei
  kleinen Referenzmengen robuster).
- Keine Web-UI, kein Docker — reine CLI.
