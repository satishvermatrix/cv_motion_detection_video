# Project Notes — Motion-Based Frame Reduction

## What we're building

Working on a video processing optimization — specifically, a way to reduce how many frames we send to expensive computer vision models like object detectors. The idea is simple: if nothing is moving in a frame, there's no point processing it. So we detect motion first, and only keep frames where something actually changed.

---

## The Dataset

Using CDnet 2014, a standard academic benchmark with 53 real-world surveillance videos. What makes it useful is that every single pixel of every frame has been hand-labeled by researchers — so we know exactly which frames contain real motion and which don't. This gives us a ground truth to measure against.

Dataset path: `data/dataset/`

Categories evaluated so far:
- `baseline` — clean conditions, easy
- `cameraJitter` — unstable/shaky camera
- `dynamicBackground` — waving trees, fountains, water

---

## The Five Approaches

Implemented five motion detection approaches, going from simple to complex:

| # | Detector | How it works |
|---|---|---|
| 1 | **Raw frame diff** | Subtract consecutive frames pixel by pixel |
| 2 | **Blurred frame diff** | Same, but blur first to remove JPEG noise |
| 3 | **MOG2** | Statistical background model, adapts to lighting changes |
| 4 | **KNN** | Similar background model, different algorithm |
| 5 | **Optical flow** | Estimates actual pixel movement direction and speed |

Each outputs a motion score per frame. A threshold is applied to decide keep or drop.

---

## Evaluation Strategy

Treat it as binary classification — did we keep the right frames?

| Outcome | Meaning |
|---|---|
| TP | Kept a motion frame (correct) |
| FP | Kept a static frame (wasted budget, but safe) |
| TN | Dropped a static frame (correct) |
| FN | Dropped a motion frame (bad — missed real motion) |

**Key metrics:**
- **Recall** — did we miss any frames with real motion? Target: ≥ 95%
- **Compression ratio** — what fraction of frames did we drop? Maximize this.
- **F1** — balanced score

We sweep the threshold from 0→1, plot a Precision-Recall curve, and pick the operating point that hits ≥ 95% recall with the highest compression.

---

## Early Results

On clean baseline videos, all detectors hit 97–99% recall.

The interesting finding is on `dynamicBackground` (waving trees, fountains):
- **MOG2 / KNN** compress ~49% of frames at 96%+ recall — they correctly learn that tree movement is background
- **Frame diff** keeps almost everything — it can't distinguish tree movement from real motion

On `cameraJitter`:
- Frame diff precision drops (camera shake looks like motion everywhere)
- MOG2 / KNN hold up better because their background model adapts

---

## Next Steps

- Run full downstream evaluation: does a YOLO object detector miss any detections when fed the reduced frame set vs all frames?
- Extend to harder categories: `intermittentObjectMotion`, `nightVideos`
- Tune `MIN_MOTION_PIXELS` threshold across sequences

---

## How to Run

```bash
# Install dependencies
uv sync

# Run full evaluation
uv run python scripts/run_eval.py \
    --dataset data/dataset \
    --categories baseline cameraJitter dynamicBackground \
    --detectors raw_diff blurred_diff mog2 knn flow \
    --min-recall 0.95 \
    --results-dir results

# Visualise a specific sequence
uv run python scripts/visualize.py \
    --dataset data/dataset \
    --category baseline \
    --sequence highway \
    --detector blurred_diff \
    --output results/vis

# Downstream YOLO eval (requires: uv add ultralytics)
uv run python scripts/downstream_yolo_eval.py \
    --dataset data/dataset \
    --category baseline \
    --detector blurred_diff \
    --output results/yolo

# Open tutorial notebook
uv run jupyter notebook notebooks/tutorial.ipynb
```

---

## Repo

`github.com:satishvermatrix/cv_motion_detection_video`
