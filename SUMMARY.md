# Motion-Based Frame Reduction — Project Summary

## Goal

Reduce the number of frames sent to expensive computer vision models (object detectors, trackers) by detecting motion first and dropping frames where nothing changed. The core hypothesis: **most frames in a surveillance video are redundant — static background with no new information.**

---

## Dataset — CDnet 2014

### Overview

53 real-world surveillance video sequences across 11 difficulty categories, with every pixel of every frame hand-labeled by human annotators (team of 13 researchers from 7 universities). Total: ~160,000 annotated frames.

Videos were recorded with a diverse range of cameras — low-resolution IP cameras (320×240), consumer camcorders, PTZ cameras, and near-infrared cameras. Frame rates vary from 0.17 fps (bandwidth-limited IP cameras) to 30 fps. This diversity means the dataset does not favour any particular family of methods.

### Ground Truth Labels

Each pixel in every GT frame carries one of five values:

| Value | Label | Used in eval as |
|---|---|---|
| 0 | Static background | Negative |
| 50 | Shadow | Negative (treated as background) |
| 85 | Outside ROI | Ignored |
| 170 | Unknown / motion-blurred | Ignored |
| 255 | Foreground (motion) | Positive |

**Frame-level label derivation:** A frame is labeled "has motion" if ≥ 200 foreground pixels (value=255) exist inside the spatial ROI. This threshold is configurable.

**Temporal ROI:** The first N frames of each sequence (e.g. frames 1–469 in `highway`) are a warmup window — ground truth exists but is not evaluated. This lets stateful detectors (MOG2, KNN) build their background model before being judged.

**Spatial ROI:** A binary mask defines which pixels count. Pixels outside the ROI are ignored in all metrics — this prevents static borders, irrelevant background regions, and edge artifacts from distorting results.

### All 11 Categories

| Category | # Seqs | Description | Compression ceiling |
|---|---|---|---|
| `baseline` | 4 | Stable camera, clean conditions, mild challenges | Low (~5%) — near-continuous motion |
| `dynamicBackground` | 6 | Outdoor scenes with strong background motion (waving trees, water, fountains) | High (30–80%) — background motion creates many false static periods |
| `cameraJitter` | 4 | Camera mounted on unstable support — constant slight shake | Low-medium (~5–15%) |
| `shadow` | 6 | Strong and soft moving shadows | Medium (~10–30%) |
| `intermittentObjectMotion` | 6 | Objects that stop and restart — causes ghosting in background models | Medium (~20–40%) |
| `thermal` | 5 | Far-infrared camera footage | Medium (~15–30%) |
| `badWeather` | 4 | Blizzard, snowfall, low visibility outdoor scenes | Low-medium (~10–20%) |
| `lowFramerate` | 4 | 0.17–1 fps due to bandwidth limits — erratic motion patterns | Variable |
| `nightVideos` | 6 | Traffic at night — headlight halos, low visibility | Low (~5–10%) — near-continuous traffic |
| `PTZ` | 4 | Pan/tilt/zoom camera movement | Very low — camera motion makes everything appear to move |
| `turbulence` | 4 | Near-infrared at 5–15km through heat shimmer | Low-medium (~10–20%) |

### What compression can you expect?

Compression ceiling = fraction of frames that are genuinely static (GT=0). Even a perfect detector cannot exceed this.

| Scene type | Real-world example | Typical compression ceiling |
|---|---|---|
| Busy intersection, continuous traffic | Highway at peak hour | 3–8% |
| Moderately busy road | Suburban junction | 15–30% |
| Quiet road / car park | Night, off-peak | 40–70% |
| Outdoor scene with trees/water background | Park, riverside camera | 30–80% (background motion bloats false keeps for frame diff) |
| Indoor office / corridor | After-hours CCTV | 50–80% |
| Parking lot | Overnight | 70–90% |

The dataset sequences evaluated in this project confirmed these ranges — `dynamicBackground` sequences saw 30–80% compression with MOG2, while `baseline` highway sequences compressed only 2–6%.

---

## Approaches

### Approach 1 — Raw Frame Differencing

**How it works:**
Convert consecutive frames to grayscale and compute absolute pixel-wise difference. Pixels that changed more than `pixel_threshold` are counted as motion. Motion score = fraction of changed pixels.

```
gray_t   = grayscale(frame[t])
gray_t-1 = grayscale(frame[t-1])
diff     = |gray_t - gray_t-1|
mask     = diff > pixel_threshold   (default: 25)
score    = mean(mask)               → [0, 1]
```

**Parameters:**
- `pixel_threshold=25` — sensitivity. Lower = more sensitive, more noise. Higher = misses slow/small motion.

**Strengths:** Extremely fast, zero memory, no warmup needed, works on any video immediately.

**Weaknesses:** JPEG artifacts cause false positives on static frames. Camera shake makes every pixel appear to move. Cannot distinguish background motion (trees) from foreground motion.

**Best for:** Controlled indoor environments, static cameras, low-noise video sources.

---

### Approach 2 — Blurred Frame Differencing

**How it works:**
Identical to raw frame diff but applies a Gaussian blur to each frame before differencing. The blur removes high-frequency noise (JPEG ringing, sensor noise) before the comparison.

```
blurred_t   = GaussianBlur(grayscale(frame[t]),   kernel=5×5)
blurred_t-1 = GaussianBlur(grayscale(frame[t-1]), kernel=5×5)
diff        = |blurred_t - blurred_t-1|
mask        = diff > pixel_threshold   (default: 20, slightly lower than raw)
score       = mean(mask)
```

**Parameters:**
- `pixel_threshold=20` — slightly lower than raw since blur reduces diff magnitude
- `blur_ksize=5` — kernel size. Larger = more smoothing, but blurs real motion edges too

**Strengths:** More robust to JPEG noise than raw diff. Minimal overhead over raw diff. Still stateless — no warmup.

**Weaknesses:** Same fundamental limitations as raw diff — still fooled by camera shake and dynamic backgrounds.

**Best for:** Compressed video sources (IP cameras with JPEG artefacts) where raw diff produces too many false positives.

---

### Approach 3 — MOG2 (Mixture of Gaussians)

**How it works:**
Builds a statistical model of the background for each pixel independently, updated incrementally over time. Each pixel maintains a mixture of Gaussian distributions representing the range of values it normally takes. A new pixel value is foreground if it doesn't fit any of its background Gaussians.

```
# Per pixel, maintain K Gaussians (auto-selected, up to 5):
#   Gaussian_i: mean_i, std_i, weight_i

# For each new frame:
fg_mask = subtractor.apply(frame)
#   → 255 if pixel doesn't match any Gaussian  (foreground)
#   → 127 if pixel is darker than background in correct ratio (shadow)
#   → 0   if pixel matches a Gaussian  (background)

score = mean(fg_mask == 255)   # shadows excluded
```

After each frame the Gaussians update: matched Gaussians have their mean nudged toward the new value and weight increased; unmatched Gaussians lose weight; new Gaussians are created for unmatched values.

**Parameters:**
- `history=500` — frames of background memory (~20s at 25fps). Shorter = faster adaptation, more false positives. Longer = more stable, slower to adapt.
- `var_threshold=16` — squared Mahalanobis distance to declare foreground. Higher = less sensitive.
- `detect_shadows=True` — marks shadows separately (127) and excludes from motion score.

**Strengths:** Adapts to gradual illumination changes (day/night transition, indoor lighting). Handles dynamic backgrounds (waving trees) by learning them as background modes. Shadow detection prevents shadow-triggered false keeps. No fixed number of Gaussians — auto-selected per pixel.

**Weaknesses:** Camera jitter causes all pixels to move simultaneously, overwhelming the model. Objects stationary for >history frames get absorbed into background. Requires warmup before evaluation.

**Best for:** Outdoor road cameras, surveillance cameras with gradual scene variation, scenes with dynamic backgrounds.

---

### Approach 4 — KNN Background Subtraction

**How it works:**
Same goal as MOG2 but uses a non-parametric approach. Instead of fitting Gaussians, stores a buffer of K raw historical pixel values per pixel location. A new value is foreground if it is far from all its K nearest neighbours in the history buffer.

```
# Per pixel, maintain buffer of last N observed values:
#   history_buffer = [118, 120, 117, 119, 121, ...]

# For each new pixel value v:
#   find K nearest neighbours in buffer
#   if distance to neighbours > sqrt(dist2_threshold):
#       → FOREGROUND
#   else:
#       → BACKGROUND
```

**Parameters:**
- `history=500` — buffer size per pixel
- `dist2_threshold=400` — squared distance threshold (√400 = 20 intensity units)
- `detect_shadows=True` — same shadow detection as MOG2

**Strengths:** Makes no assumption about distribution shape — works for pixels with non-Gaussian backgrounds (flickering lights, level crossing barriers, rotating machinery). Computationally efficient when most pixels are background (sparse foreground).

**Weaknesses:** Higher memory usage than MOG2 (stores raw values, not compressed Gaussian parameters). Same camera jitter weakness as MOG2.

**When to prefer over MOG2:** Scenes with pixels that snap between two very distinct states with nothing in between — a flickering neon sign, traffic light backgrounds, level crossing barriers.

---

### Approach 5 — Optical Flow (Farneback)

**How it works:**
Estimates a 2D displacement vector `(dx, dy)` per pixel between consecutive frames using polynomial expansion at multiple pyramid levels. The magnitude of the flow vector is the per-pixel motion signal.

```
flow = calcOpticalFlowFarneback(gray_t-1, gray_t, ...)
# flow shape: (H, W, 2)  →  flow[y, x] = (dx, dy)

magnitude = sqrt(dx² + dy²)
score = mean(magnitude / max_magnitude)   # normalised to [0, 1]
```

**Key difference from frame diff:** A global illumination change makes every pixel value shift, but the pixels don't actually *move* — they stay in place with different brightness. Optical flow correctly assigns near-zero displacement to illumination-only changes. Frame diff would spike.

**Parameters:**
- `max_magnitude=20` — flow magnitude (pixels/frame) considered "full motion"
- `score_mode="mean_norm"` — use normalised mean magnitude as score

**Strengths:** Robust to global illumination changes. Provides direction of motion (useful for filtering camera shake — global uniform flow = camera move, not object move). Produces rich visualisation (HSV colour-coded direction map).

**Weaknesses:** 5–10× slower than frame diff due to pyramid computation. Struggles with very large displacements (fast-moving objects that exceed the pyramid resolution). No background model — still compares only to previous frame.

**Best for:** Scenes with significant illumination variation where frame diff produces too many false positives, but compute budget allows it.

---

### Approach 6 — Histogram-Based Methods (not implemented, for reference)

**How it works:**
Summarise each frame as a compact histogram of pixel intensities or colours. Compare consecutive histograms using a distance metric. Histogram distance = motion score.

```
H_t   = histogram(frame[t],   bins=256)
H_t-1 = histogram(frame[t-1], bins=256)
score = histogram_distance(H_t, H_t-1)
```

Common distance metrics:
- **Chi-squared** — sensitive to bins with few counts, fast
- **Bhattacharyya** — measures distribution overlap, range 0–1, most commonly used
- **Correlation** — 1 = identical histograms, 0 = no overlap
- **Earth Mover's Distance** — "work" to transform one histogram to another, handles shifted distributions

**Variants:**
- **Colour histogram (HSV)** — uses 3D HSV histogram, separates colour from brightness change, more discriminative than grayscale
- **Local block histograms** — divide frame into grid (e.g. 4×4), compute histogram per block, compare block-by-block. Catches localised motion that doesn't shift the global histogram much.
- **Gradient histogram** — histogram of edge orientations. Sensitive to structural changes (new objects) rather than illumination changes.

**Strengths:** Extremely fast and stateless. Very robust to illumination changes (HSV histogram barely shifts with brightness change). No warmup. Excellent for scene cut / shot boundary detection.

**Weaknesses:** Global summary — throws away all spatial information. A small vehicle in one corner barely shifts a 1920×1080 frame's overall histogram. This causes small/distant objects to be missed, directly hurting downstream detection recall. Not suitable as a sole frame reduction mechanism.

**Recommended use — two-stage pre-filter:**
Use histogram comparison as a cheap first gate before MOG2:
```
frame → histogram diff → score < low_threshold  → DROP immediately (definitely static)
                      → score > high_threshold  → keep, run MOG2 for precise decision
                      → in between             → run MOG2
```
This saves CPU on truly static frames without the spatial-locality weakness affecting detection recall.

**Best for:** Scene change / shot boundary detection in video editing pipelines. Pre-filtering before a more expensive spatial detector in high-stream-count deployments.

---

## Results Summary

### At ≥95% recall target — mean compression per detector

| Detector | Mean compression | Notes |
|---|---|---|
| **MOG2** | **24.9%** | Best overall |
| **KNN** | **23.5%** | Close second |
| flow | 14.5% | 5–10× slower, no benefit |
| blurred_diff | 14.0% | |
| raw_diff | 13.5% | |

### By category

| Category | Mean compression | Best detector | Notes |
|---|---|---|---|
| `dynamicBackground` | **34.7%** | KNN (up to 81% on `boats`) | Biggest gains — background models learn dynamic bg |
| `baseline` | 6.0% | All similar | Low ceiling — near-continuous motion |
| `cameraJitter` | 5.3% | KNN/MOG2 | Low ceiling + camera shake hurts frame diff precision |

### Downstream YOLO (baseline, blurred_diff)

| Sequence | Motion frame rate | Frame compression | Detection recall |
|---|---|---|---|
| PETS2006 | ~100% | 0% | 100% |
| highway | ~97% | 0% | 100% |
| office | ~100% | 0% | 100% |
| pedestrians | ~74% | **26%** | **82.5%** |

Three sequences show 0% compression because they are nearly 100% motion — no static frames exist to drop. The pedestrians result is the actionable finding: 26% compression caused YOLO to miss 17.5% of detections. Small objects appear in few consecutive frames so dropping any of them loses the detection.

---

## Conclusions

**1. Background subtraction (MOG2) is the right approach for surveillance**
On challenging conditions it compresses 2–8× more than frame differencing while maintaining equal or better recall. The warmup cost is a one-time initialization.

**2. Frame differencing is a useful baseline but not production-ready**
Fast and stateless, but precision on dynamic backgrounds is poor. Use as a first gate or sanity check, not as the primary decision mechanism.

**3. Optical flow offers no meaningful advantage for frame reduction**
5–10× slower than frame diff with similar or worse compression on CDnet. Reserve for scenarios with severe illumination variation where diff-based methods fail completely.

**4. Histogram methods are wrong tool for frame reduction, right tool for scene change**
Too coarse to reliably detect small objects. Best used as a cheap pre-filter before MOG2, not as a standalone frame reduction mechanism.

**5. Compression ceiling is scene-dependent — set expectations accordingly**
Busy scenes with continuous motion compress 3–8%. Quiet scenes with intermittent activity compress 40–80%. Knowing the scene type before deployment lets you predict expected savings.

**6. Downstream safety requires conservative thresholds**
95% motion recall is not sufficient if the downstream model detects small objects. Target 98–99% recall, accept less compression, and never drop more than N consecutive frames regardless of score.

---

## Recommendations for DeepStream Integration

Use MOG2 as the gate before the detector pipeline. For high-stream-count deployments, add a histogram pre-filter as a cheap first stage.

| Decision | Recommendation |
|---|---|
| Primary detector | MOG2 — `history=500`, `var_threshold=25`, `detect_shadows=True` |
| Optional pre-filter | Bhattacharyya histogram distance < 0.02 → drop frame immediately, skip MOG2 |
| Threshold strategy | Target 98%+ recall — be conservative, not aggressive |
| Warmup | Feed first 500 frames through MOG2 before enabling frame drop |
| Distant/small object cameras | Lower `var_threshold` to 12–16, raise min recall target to 99% |
| Night / static scenes | Increase `history=1000–2000` to avoid absorbing slow-moving vehicles into background |
| Tracker continuity | Never drop more than N consecutive frames regardless of score — preserves tracklet continuity downstream |
| Per-camera tuning | Run `scripts/run_eval.py` on a sample of each camera's actual footage to find the correct threshold per deployment |

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
