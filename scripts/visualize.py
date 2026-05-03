#!/usr/bin/env python3
"""
Visualisation script: side-by-side kept vs dropped frames.

For a given sequence and detector, renders a grid showing:
  - Kept frames   (green border) — detector predicted motion
  - Dropped frames (red border)  — detector predicted static

Also outputs:
  - A timeline plot showing GT labels vs predicted scores over time
  - A side-by-side comparison of kept and dropped frames (sampled)

Usage
-----
    python scripts/visualize.py \
        --dataset data/dataset \
        --category baseline \
        --sequence highway \
        --detector blurred_diff \
        --threshold 0.01 \
        --output results/vis/

Options
-------
    --dataset PATH        CDnet dataset root.
    --category STR        Category name (e.g. "baseline").
    --sequence STR        Sequence name (e.g. "highway").
    --detector STR        Detector to use. Choices: raw_diff blurred_diff mog2 knn flow
    --threshold FLOAT     Decision threshold (overrides auto). Default: auto (best operating point).
    --min-recall FLOAT    Used to pick best operating point if threshold not set. Default: 0.95
    --min-motion-px INT   Min foreground pixels for GT motion label. Default: 200
    --n-sample INT        Number of sample frames per class to show in grid. Default: 8
    --output PATH         Output directory. Default: results/vis/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.detectors.bg_subtract import KNNDetector, MOG2Detector
from src.detectors.frame_diff import BlurredFrameDiff, RawFrameDiff
from src.detectors.optical_flow import FarnebackDetector
from src.eval.metrics import best_operating_point, threshold_sweep
from src.utils.cdnet_loader import CDnetSequence, derive_frame_label, load_category


DETECTOR_MAP = {
    "raw_diff":     lambda: RawFrameDiff(pixel_threshold=25),
    "blurred_diff": lambda: BlurredFrameDiff(pixel_threshold=20, blur_ksize=5),
    "mog2":         lambda: MOG2Detector(history=500, var_threshold=16.0),
    "knn":          lambda: KNNDetector(history=500, dist2_threshold=400.0),
    "flow":         lambda: FarnebackDetector(score_mode="mean_norm", max_magnitude=20.0),
}

# Border colours (BGR)
GREEN = (0, 200, 0)
RED = (0, 0, 200)
BORDER_W = 6


def add_border(frame: np.ndarray, colour: tuple, width: int = BORDER_W) -> np.ndarray:
    out = frame.copy()
    out[:width, :] = colour
    out[-width:, :] = colour
    out[:, :width] = colour
    out[:, -width:] = colour
    return out


def make_frame_grid(
    frames: list[np.ndarray],
    max_frames: int = 8,
    thumb_size: tuple[int, int] = (160, 120),
    cols: int = 8,
) -> np.ndarray:
    """Arrange up to max_frames thumbnails in a grid."""
    sampled = frames[:max_frames]
    thumbs = [cv2.resize(f, thumb_size) for f in sampled]
    # Pad to full grid
    rows = (len(thumbs) + cols - 1) // cols
    blank = np.zeros((*thumb_size[::-1], 3), dtype=np.uint8)
    while len(thumbs) < rows * cols:
        thumbs.append(blank)
    row_imgs = []
    for r in range(rows):
        row = np.hstack(thumbs[r * cols: (r + 1) * cols])
        row_imgs.append(row)
    return np.vstack(row_imgs)


def plot_score_timeline(
    frame_indices: list[int],
    gt_labels: list[int],
    scores: list[float],
    threshold: float,
    sequence_name: str,
    detector_name: str,
    output_path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    x = frame_indices

    # GT labels
    ax1.fill_between(x, gt_labels, step="mid", alpha=0.6, color="steelblue", label="GT (1=motion)")
    ax1.set_ylabel("GT label", fontsize=10)
    ax1.set_ylim(-0.1, 1.3)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title(f"{sequence_name} — GT vs detector score ({detector_name})", fontsize=11)

    # Detector score
    ax2.plot(x, scores, linewidth=0.8, color="darkorange", label="Motion score")
    ax2.axhline(y=threshold, color="red", linestyle="--", linewidth=1.2, label=f"Threshold={threshold:.4f}")
    ax2.fill_between(x, scores, threshold, where=np.array(scores) >= threshold,
                     alpha=0.3, color="darkorange", label="Kept frames")
    ax2.set_ylabel("Motion score", fontsize=10)
    ax2.set_xlabel("Frame index", fontsize=10)
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  Saved timeline: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise kept/dropped frames for a CDnet sequence",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset"))
    parser.add_argument("--category", type=str, default="baseline")
    parser.add_argument("--sequence", type=str, default="highway")
    parser.add_argument("--detector", type=str, default="blurred_diff",
                        choices=list(DETECTOR_MAP.keys()))
    parser.add_argument("--threshold", type=float, default=None,
                        help="Decision threshold. If not set, auto-selected via PR sweep.")
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--min-motion-px", type=int, default=200)
    parser.add_argument("--n-sample", type=int, default=8,
                        help="Number of sample frames per class to show in grid")
    parser.add_argument("--output", type=Path, default=Path("results/vis"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    seq_path = args.dataset / args.category / args.sequence
    if not seq_path.exists():
        print(f"ERROR: Sequence not found: {seq_path}")
        sys.exit(1)

    from src.utils.cdnet_loader import CDnetSequence
    seq = CDnetSequence.load(seq_path, category=args.category)
    detector = DETECTOR_MAP[args.detector]()

    print(f"\nSequence : {seq}")
    print(f"Detector : {args.detector}")
    print(f"Temporal ROI: {seq.temporal_roi}")

    # --- Run detector ---
    frame_indices: list[int] = []
    gt_labels: list[int] = []
    scores: list[float] = []
    frames_cache: dict[int, np.ndarray] = {}

    detector.reset()
    all_records = list(seq.iter_frames(eval_only=False))

    print("Running detector...")
    for record in tqdm(all_records):
        frame = record.load_input()
        score = detector.compute_score(frame)

        if not record.in_temporal_roi:
            continue

        gt_mask = record.load_gt_mask()
        if gt_mask is None:
            continue

        label = int(derive_frame_label(
            gt_mask, seq.roi_mask, min_foreground_pixels=args.min_motion_px
        ))
        frame_indices.append(record.frame_idx)
        gt_labels.append(label)
        scores.append(score)
        frames_cache[record.frame_idx] = frame.copy()

    if not gt_labels:
        print("ERROR: No labeled frames found.")
        sys.exit(1)

    # --- Determine threshold ---
    if args.threshold is not None:
        threshold = args.threshold
        print(f"Using provided threshold: {threshold}")
    else:
        sweep = threshold_sweep(gt_labels, scores, n_thresholds=200)
        best = best_operating_point(sweep, min_recall=args.min_recall)
        threshold = best.threshold
        print(
            f"Auto threshold={threshold:.6f} "
            f"recall={best.recall:.3f} precision={best.precision:.3f} "
            f"f1={best.f1:.3f} compression={best.compression_ratio:.3f}"
        )

    # --- Classify frames ---
    kept_frames_correct: list[np.ndarray] = []    # TP
    kept_frames_wrong: list[np.ndarray] = []      # FP (kept static)
    dropped_frames_correct: list[np.ndarray] = [] # TN
    dropped_frames_wrong: list[np.ndarray] = []   # FN (dropped motion)

    for idx, gt, score in zip(frame_indices, gt_labels, scores):
        f = frames_cache[idx]
        predicted_keep = score >= threshold
        if predicted_keep and gt == 1:
            kept_frames_correct.append(add_border(f, GREEN))
        elif predicted_keep and gt == 0:
            kept_frames_wrong.append(add_border(f, (255, 165, 0)))  # orange = FP
        elif not predicted_keep and gt == 0:
            dropped_frames_correct.append(add_border(f, RED))
        else:
            dropped_frames_wrong.append(add_border(f, (128, 0, 128)))  # purple = FN

    n = args.n_sample
    print(f"\nTP (correctly kept motion)   : {len(kept_frames_correct)}")
    print(f"FP (incorrectly kept static) : {len(kept_frames_wrong)}")
    print(f"TN (correctly dropped static): {len(dropped_frames_correct)}")
    print(f"FN (incorrectly dropped motion): {len(dropped_frames_wrong)}")

    # --- Save frame grids ---
    prefix = f"{args.category}_{args.sequence}_{args.detector}"

    for frames, label in [
        (kept_frames_correct,  "TP_kept_motion"),
        (kept_frames_wrong,    "FP_kept_static"),
        (dropped_frames_correct, "TN_dropped_static"),
        (dropped_frames_wrong,   "FN_dropped_motion"),
    ]:
        if frames:
            grid = make_frame_grid(frames, max_frames=n)
            out_path = args.output / f"{prefix}_{label}.jpg"
            cv2.imwrite(str(out_path), grid)
            print(f"  Saved grid: {out_path}")

    # --- Score timeline ---
    plot_score_timeline(
        frame_indices, gt_labels, scores, threshold,
        sequence_name=f"{args.category}/{args.sequence}",
        detector_name=args.detector,
        output_path=args.output / f"{prefix}_timeline.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
