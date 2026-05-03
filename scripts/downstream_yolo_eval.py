#!/usr/bin/env python3
"""
Downstream YOLO evaluation: compare detection recall on full vs reduced frame sets.

For each sequence, we:
  1. Run a motion detector to get a reduced frame set.
  2. Run YOLOv8n on both the full set and the reduced set.
  3. Compute: what fraction of unique objects detected in the full set
     are also detected in the reduced set?

This measures whether the frame reduction is "safe" for a downstream
object detection task.

Usage
-----
    python scripts/downstream_yolo_eval.py \
        --dataset data/dataset \
        --category baseline \
        --detector blurred_diff \
        --threshold 0.01 \
        --conf 0.3 \
        --output results/yolo/

Requirements
------------
    pip install ultralytics
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from tqdm import tqdm


def try_import_ultralytics():
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError:
        print("ERROR: ultralytics is not installed.")
        print("  Install with: pip install ultralytics")
        sys.exit(1)


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


def iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two xyxy boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def count_matched_detections(
    full_dets: list[dict],
    reduced_dets: list[dict],
    iou_threshold: float = 0.5,
) -> tuple[int, int]:
    """
    Count how many unique detections from full_dets appear in reduced_dets.

    A full-set detection is "found" if any reduced-set detection on the same
    frame (±1 frame) has IoU >= iou_threshold.

    Returns (matched, total_full).
    """
    # Build reduced detections indexed by frame idx
    reduced_by_frame: dict[int, list[np.ndarray]] = {}
    for d in reduced_dets:
        reduced_by_frame.setdefault(d["frame_idx"], []).append(d["box"])

    matched = 0
    for d in full_dets:
        fidx = d["frame_idx"]
        candidates: list[np.ndarray] = []
        for offset in (-1, 0, 1):
            candidates.extend(reduced_by_frame.get(fidx + offset, []))

        found = any(iou(d["box"], c) >= iou_threshold for c in candidates)
        if found:
            matched += 1

    return matched, len(full_dets)


def run_yolo_on_frames(
    model,
    frames: list[tuple[int, np.ndarray]],
    conf: float = 0.3,
    verbose: bool = False,
) -> list[dict]:
    """
    Run YOLO on a list of (frame_idx, bgr_frame) pairs.

    Returns list of dicts: {frame_idx, box (xyxy), cls, conf}.
    """
    detections = []
    for frame_idx, frame in frames:
        results = model(frame, conf=conf, verbose=False)
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                detections.append({
                    "frame_idx": frame_idx,
                    "box": box.xyxy[0].cpu().numpy(),
                    "cls": int(box.cls[0]),
                    "conf": float(box.conf[0]),
                })
    return detections


def evaluate_sequence_downstream(
    seq: CDnetSequence,
    detector,
    model,
    threshold: float,
    conf: float = 0.3,
    min_motion_px: int = 200,
) -> dict:
    detector.reset()

    full_frames: list[tuple[int, np.ndarray]] = []
    reduced_frames: list[tuple[int, np.ndarray]] = []
    gt_labels: list[int] = []
    scores_list: list[float] = []

    all_records = list(seq.iter_frames(eval_only=False))
    for record in all_records:
        frame = record.load_input()
        score = detector.compute_score(frame)

        if not record.in_temporal_roi:
            continue

        full_frames.append((record.frame_idx, frame.copy()))

        if score >= threshold:
            reduced_frames.append((record.frame_idx, frame.copy()))

        gt_mask = record.load_gt_mask()
        if gt_mask is not None:
            label = int(derive_frame_label(gt_mask, seq.roi_mask, min_motion_px))
            gt_labels.append(label)
            scores_list.append(score)

    if not full_frames:
        return {}

    print(f"  Running YOLO on {len(full_frames)} full frames...")
    full_dets = run_yolo_on_frames(model, full_frames, conf=conf)

    print(f"  Running YOLO on {len(reduced_frames)} reduced frames ({100*len(reduced_frames)/len(full_frames):.1f}%)...")
    reduced_dets = run_yolo_on_frames(model, reduced_frames, conf=conf)

    matched, total = count_matched_detections(full_dets, reduced_dets)
    detection_recall = matched / total if total > 0 else 1.0
    frame_compression = 1.0 - len(reduced_frames) / len(full_frames)

    return {
        "category": seq.category,
        "sequence": seq.name,
        "total_frames": len(full_frames),
        "reduced_frames": len(reduced_frames),
        "frame_compression": round(frame_compression, 4),
        "full_detections": total,
        "matched_detections": matched,
        "detection_recall": round(detection_recall, 4),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downstream YOLO evaluation on full vs reduced frame sets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset"))
    parser.add_argument("--category", type=str, default="baseline")
    parser.add_argument("--sequences", nargs="*", default=None,
                        help="Specific sequences. Default: all in category.")
    parser.add_argument("--detector", type=str, default="blurred_diff",
                        choices=list(DETECTOR_MAP.keys()))
    parser.add_argument("--threshold", type=float, default=None,
                        help="Decision threshold. If not set, auto-selected.")
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--min-motion-px", type=int, default=200)
    parser.add_argument("--conf", type=float, default=0.3,
                        help="YOLO confidence threshold.")
    parser.add_argument("--yolo-model", type=str, default="yolov8n.pt",
                        help="YOLO model weights.")
    parser.add_argument("--output", type=Path, default=Path("results/yolo"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    YOLO = try_import_ultralytics()
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading YOLO model: {args.yolo_model}")
    model = YOLO(args.yolo_model)

    sequences = load_category(args.dataset, args.category)
    if args.sequences:
        sequences = [s for s in sequences if s.name in args.sequences]

    detector = DETECTOR_MAP[args.detector]()

    results = []
    for seq in sequences:
        print(f"\n--- {seq.category}/{seq.name} ---")

        # Determine threshold from a quick sweep if not provided
        if args.threshold is None:
            print("  Running quick sweep to find threshold...")
            detector.reset()
            gt_labels: list[int] = []
            scores_list: list[float] = []
            for record in seq.iter_frames(eval_only=True):
                frame = record.load_input()
                score = detector.compute_score(frame)
                gt_mask = record.load_gt_mask()
                if gt_mask is not None:
                    label = int(derive_frame_label(
                        gt_mask, seq.roi_mask, args.min_motion_px
                    ))
                    gt_labels.append(label)
                    scores_list.append(score)
            if gt_labels:
                sweep = threshold_sweep(gt_labels, scores_list, n_thresholds=100)
                best = best_operating_point(sweep, min_recall=args.min_recall)
                threshold = best.threshold
                print(f"  Auto threshold: {threshold:.6f}")
            else:
                threshold = 0.01
        else:
            threshold = args.threshold

        row = evaluate_sequence_downstream(
            seq, detector, model,
            threshold=threshold,
            conf=args.conf,
            min_motion_px=args.min_motion_px,
        )
        if row:
            results.append(row)
            print(
                f"  Frame compression: {row['frame_compression']:.3f} | "
                f"Detection recall: {row['detection_recall']:.3f} "
                f"({row['matched_detections']}/{row['full_detections']} dets)"
            )

    # Save CSV
    if results:
        out_csv = args.output / f"downstream_{args.category}_{args.detector}.csv"
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved: {out_csv}")

    print("Done.")


if __name__ == "__main__":
    main()
