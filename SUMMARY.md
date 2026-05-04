# Motion-Based Frame Reduction — Project Summary

## Goal

Reduce the number of frames sent to expensive computer vision models (object detectors, trackers) by detecting motion first and dropping frames where nothing changed. The core hypothesis: **most frames in a surveillance video are redundant — static background with no new information.**

---

## Dataset — CDnet 2014

53 real-world surveillance video sequences across 11 difficulty categories, with every pixel of every frame hand-labeled by human annotators.

**Ground truth derivation:** A frame is labeled "has motion" if ≥ 200 foreground pixels exist inside the ROI. This becomes the binary label the detector tries to predict.

**Categories evaluated:**

| Category | Challenge | Sequences |
|---|---|---|
| `baseline` | Clean conditions, stable camera | 4 |
| `cameraJitter` | Unstable / shaky camera | 4 |
| `dynamicBackground` | Moving background (trees, water, fountains) | 6 |

**Maximum theoretical compression** (fraction of static frames) varies widely per sequence — some sequences are nearly all motion (very little room to compress), others are 30–50% static (significant headroom).

---

## Five Approaches

### 1. Raw Frame Differencing
Compares each pixel between consecutive frames. If more than `pixel_threshold=25` intensity units changed, that pixel counts as motion. Motion score = fraction of changed pixels.

### 2. Blurred Frame Differencing
Same as above but applies a 5×5 Gaussian blur before differencing. Removes JPEG compression artifacts and sensor noise that would otherwise cause false positives on static frames.

### 3. MOG2 (Mixture of Gaussians)
Builds a per-pixel statistical background model over the last 500 frames. Each pixel independently maintains multiple Gaussian distributions representing what "background" looks like at that location. Automatically adapts to gradual illumination changes. Detects and excludes shadows separately.

### 4. KNN Background Subtraction
Same concept as MOG2 but stores raw historical pixel values instead of Gaussian parameters. Classifies a pixel as foreground if its current value is far from all K nearest historical samples. Better for pixels with non-Gaussian background patterns (flickering lights, level crossings).

### 5. Optical Flow (Farneback)
Estimates actual 2D displacement vectors per pixel between consecutive frames. Motion score = mean flow magnitude normalised to a `max_magnitude` of 20px/frame. Unlike diff-based methods, a global illumination change produces near-zero flow even though every pixel value changes — making it more robust to lighting shifts.

---

## Results

### Baseline (clean conditions)

| Detector | Recall | Compression | Notes |
|---|---|---|---|
| MOG2 | 0.998 | 5.8% | Near-perfect |
| KNN | 0.982 | 6.5% | Near-perfect |
| blurred_diff | 0.991 | 5.8% | Near-perfect |
| raw_diff | 0.979 | 6.4% | Near-perfect |
| flow | 0.990 | 5.3% | Near-perfect |

Low compression because baseline sequences have near-continuous motion — the theoretical ceiling is only ~5–6%.

### Camera Jitter (shaky camera)

| Detector | Recall | Precision | Compression |
|---|---|---|---|
| MOG2 | 0.980 | 0.883 | 7.9% |
| KNN | 0.985 | 0.880 | 6.8% |
| blurred_diff | 0.977 | 0.845 | 3.7% |
| raw_diff | 0.979 | 0.840 | 3.0% |

Camera shake causes frame diff precision to drop sharply — the entire frame appears to move so everything gets kept. MOG2/KNN are more robust because their background model adapts to the jitter pattern.

### Dynamic Background (waving trees, fountains)

| Detector | Recall | Precision | Compression |
|---|---|---|---|
| MOG2 | 0.965 | 0.665 | **48.9%** |
| KNN | 0.972 | 0.657 | **46.0%** |
| blurred_diff | 0.969 | 0.496 | 26.4% |
| raw_diff | 0.963 | 0.459 | 25.2% |
| flow | 0.964 | 0.483 | 27.0% |

The most revealing category. MOG2 and KNN correctly learn that waving trees are background, compressing ~49% of frames while maintaining 96%+ recall. Frame diff treats every leaf movement as potential motion, keeping far more frames unnecessarily.

### Downstream YOLO (baseline, blurred_diff)

| Sequence | Frame compression | Detection recall |
|---|---|---|
| PETS2006 | 0% | 100% |
| highway | 0% | 100% |
| office | 0% | 100% |
| pedestrians | **26%** | **82.5%** |

The pedestrians sequence is the concerning result — dropping 26% of frames caused YOLO to miss 17.5% of detections. Small/distant pedestrians appear in only a few consecutive frames so dropping any of them loses the detection entirely.

---

## Conclusions

**1. Background subtraction (MOG2) is the right approach for surveillance**
On realistic challenging conditions it compresses 2–8× more than frame differencing while maintaining equal or better recall. The warmup cost (500 frames) is a one-time initialization.

**2. Frame differencing is a good baseline but not production-ready**
Raw and blurred diff are fast and require no warmup, but their precision on dynamic backgrounds is poor. Good for a quick sanity check or very constrained environments.

**3. Optical flow offers no meaningful advantage for this task**
5–10× slower than frame diff with similar or worse compression. Its theoretical illumination robustness did not show up meaningfully on CDnet sequences at these settings.

**4. Compression ceiling is sequence-dependent**
If a scene has continuous motion (busy intersection at rush hour), even a perfect detector can only drop a few percent of frames. The biggest gains come on scenes with clear motion/static alternation — parking lots, empty corridors, low-traffic roads at night.

**5. Downstream safety needs careful threshold tuning**
The 82.5% detection recall on pedestrians shows that aggressive compression hurts downstream tasks. The threshold must be tuned with the downstream task in mind, not just the motion metric.

---

## Recommendations for DeepStream Integration

Use MOG2 as the gate before the detector pipeline. The motion detector runs as a lightweight pre-process step, and frames that score below threshold skip the detector entirely.

| Decision | Recommendation |
|---|---|
| Detector choice | MOG2 with `history=500`, `var_threshold=25` |
| Threshold strategy | Set conservatively — target 98%+ recall, not maximum compression |
| Warmup | Feed the first 500 frames through MOG2 before enabling frame drop |
| Min motion pixels | Tune per camera: distant cameras need lower threshold (small objects), close-up cameras can go higher |
| Static scenes (night, empty lot) | Use longer `history=1000–2000` to avoid absorbing slow-moving objects |
| Downstream sensitivity | If model detects small objects (pedestrians, cyclists), be more conservative — small objects appear in fewer frames so each dropped frame costs more |
| Fallback | Never drop more than N consecutive frames regardless of score — guarantees temporal continuity for any tracker running downstream |

---

## Repository

`github.com:satishvermatrix/cv_motion_detection_video`

### Project structure

```
motion_detection/
├── src/
│   ├── detectors/          # frame_diff.py, bg_subtract.py, optical_flow.py
│   ├── eval/               # metrics.py, pr_curve.py
│   └── utils/              # cdnet_loader.py, video_io.py
├── scripts/
│   ├── run_eval.py         # full evaluation across all detectors and categories
│   ├── visualize.py        # kept vs dropped frame grids and score timeline
│   └── downstream_yolo_eval.py  # YOLO detection recall on reduced frame sets
├── notebooks/
│   ├── tutorial.ipynb      # step-by-step walkthrough of every approach
│   └── exploration.ipynb   # interactive exploration
├── results/                # CSVs and plots from eval runs
└── pyproject.toml          # uv-managed dependencies
```

### Quick start

```bash
uv sync
uv run python scripts/run_eval.py \
    --dataset data/dataset \
    --categories baseline cameraJitter dynamicBackground \
    --detectors raw_diff blurred_diff mog2 knn flow \
    --min-recall 0.95 \
    --results-dir results
```
