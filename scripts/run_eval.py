#!/usr/bin/env python3
"""
Main evaluation script for the motion-based frame reduction pipeline.

Runs all detectors across specified CDnet categories and outputs:
  - results/summary.csv        — per-sequence metrics at best operating point
  - results/all_thresholds.csv — full threshold sweep data
  - results/plots/             — PR curves and compression-vs-recall plots

Usage
-----
    python scripts/run_eval.py \
        --dataset data/dataset \
        --categories baseline cameraJitter dynamicBackground \
        --min-recall 0.95 \
        --min-motion-px 200

Options
-------
    --dataset PATH          Path to CDnet dataset root.
    --categories LIST       Space-separated list of categories to evaluate.
                            Default: baseline cameraJitter dynamicBackground
    --min-recall FLOAT      Target recall threshold for best operating point.
                            Default: 0.95
    --min-motion-px INT     Min foreground pixels to label a frame as "has motion".
                            Default: 200
    --n-thresholds INT      Number of thresholds in the sweep. Default: 100
    --results-dir PATH      Output directory. Default: results/
    --detectors LIST        Detectors to run. Choices: raw_diff blurred_diff mog2 knn flow
                            Default: all
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

# Make sure src/ is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm

from src.detectors.bg_subtract import KNNDetector, MOG2Detector
from src.detectors.frame_diff import BlurredFrameDiff, RawFrameDiff
from src.detectors.optical_flow import FarnebackDetector
from src.eval.metrics import (
    FrameMetrics,
    aggregate_metrics,
    best_operating_point,
    compute_metrics,
    threshold_sweep,
)
from src.eval.pr_curve import (
    average_precision,
    plot_compression_vs_recall,
    plot_pr_curves,
    pr_curve_from_sweep,
)
from src.utils.cdnet_loader import (
    derive_frame_label,
    load_dataset,
)


# ---------------------------------------------------------------------------
# Detector registry
# ---------------------------------------------------------------------------

ALL_DETECTORS = {
    "raw_diff":     RawFrameDiff(pixel_threshold=25),
    "blurred_diff": BlurredFrameDiff(pixel_threshold=20, blur_ksize=5),
    "mog2":         MOG2Detector(history=500, var_threshold=16.0),
    "knn":          KNNDetector(history=500, dist2_threshold=400.0),
    "flow":         FarnebackDetector(score_mode="mean_norm", max_magnitude=20.0),
}


# ---------------------------------------------------------------------------
# Per-sequence evaluation
# ---------------------------------------------------------------------------

def evaluate_sequence(
    detector,
    sequence,
    min_motion_px: int = 200,
    n_thresholds: int = 100,
    warmup_frames: int = 0,
) -> tuple[list[int], list[float]]:
    """
    Run detector on a single CDnet sequence.

    Returns
    -------
    gt_labels : list[int]   (1 = has motion, 0 = static)
    scores    : list[float] (detector motion score per frame)
    """
    detector.reset()

    gt_labels: list[int] = []
    scores: list[float] = []

    frames = list(sequence.iter_frames(eval_only=False))
    start_roi, end_roi = sequence.temporal_roi

    for record in frames:
        frame = record.load_input()
        score = detector.compute_score(frame)

        # Only collect metrics inside the temporal ROI
        if not record.in_temporal_roi:
            continue

        gt_mask = record.load_gt_mask()
        if gt_mask is None:
            continue

        label = int(derive_frame_label(
            gt_mask,
            sequence.roi_mask,
            min_foreground_pixels=min_motion_px,
        ))
        gt_labels.append(label)
        scores.append(score)

    return gt_labels, scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate motion detectors on CDnet 2014",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/dataset"),
        help="Path to CDnet dataset root",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["baseline", "cameraJitter", "dynamicBackground"],
        help="Categories to evaluate",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.95,
        help="Minimum recall for best operating point selection",
    )
    parser.add_argument(
        "--min-motion-px",
        type=int,
        default=200,
        help="Min foreground pixels to label a frame as having motion",
    )
    parser.add_argument(
        "--n-thresholds",
        type=int,
        default=100,
        help="Number of threshold points in the sweep",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for CSVs and plots",
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        choices=list(ALL_DETECTORS.keys()),
        default=list(ALL_DETECTORS.keys()),
        help="Which detectors to run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Motion Frame Reduction Evaluation")
    print(f"  Dataset  : {args.dataset}")
    print(f"  Categories: {args.categories}")
    print(f"  Detectors : {args.detectors}")
    print(f"  Min recall: {args.min_recall}")
    print(f"{'='*60}\n")

    # Load dataset
    dataset = load_dataset(args.dataset, categories=args.categories)
    total_seqs = sum(len(v) for v in dataset.values())
    print(f"Loaded {total_seqs} sequences from {len(dataset)} categories.\n")

    selected_detectors = {k: ALL_DETECTORS[k] for k in args.detectors}

    # Storage for results
    summary_rows: list[dict] = []
    all_threshold_rows: list[dict] = []

    # Per-category PR curve data: {category: {detector: (recalls, precisions)}}
    pr_data: dict[str, dict[str, tuple]] = {}
    comp_data: dict[str, dict[str, list]] = {}

    for category, sequences in dataset.items():
        print(f"\n--- Category: {category} ({len(sequences)} sequences) ---")
        pr_data[category] = {}
        comp_data[category] = {}

        for det_name, detector in selected_detectors.items():
            print(f"  Detector: {det_name}")

            # Accumulate across all sequences in category for aggregate PR curve
            all_gt: list[int] = []
            all_scores: list[float] = []
            per_seq_best: list[FrameMetrics] = []

            for seq in tqdm(sequences, desc=f"    {category}/{det_name}", leave=False):
                t0 = time.time()
                gt_labels, scores = evaluate_sequence(
                    detector,
                    seq,
                    min_motion_px=args.min_motion_px,
                    n_thresholds=args.n_thresholds,
                )
                elapsed = time.time() - t0

                if len(gt_labels) == 0:
                    print(f"    WARNING: no labeled frames in {seq.name}, skipping")
                    continue

                all_gt.extend(gt_labels)
                all_scores.extend(scores)

                # Threshold sweep for this sequence
                sweep = threshold_sweep(gt_labels, scores, n_thresholds=args.n_thresholds)
                best = best_operating_point(sweep, min_recall=args.min_recall)
                per_seq_best.append(best)

                motion_frames = sum(gt_labels)
                static_frames = len(gt_labels) - motion_frames

                row = {
                    "category": category,
                    "sequence": seq.name,
                    "detector": det_name,
                    "total_frames": len(gt_labels),
                    "motion_frames": motion_frames,
                    "static_frames": static_frames,
                    "recall": round(best.recall, 4),
                    "precision": round(best.precision, 4),
                    "f1": round(best.f1, 4),
                    "compression_ratio": round(best.compression_ratio, 4),
                    "threshold": round(best.threshold, 6),
                    "tp": best.tp,
                    "fp": best.fp,
                    "tn": best.tn,
                    "fn": best.fn,
                    "elapsed_s": round(elapsed, 2),
                }
                summary_rows.append(row)

                for m in sweep:
                    all_threshold_rows.append({
                        "category": category,
                        "sequence": seq.name,
                        "detector": det_name,
                        **m.to_dict(),
                    })

                print(
                    f"    {seq.name:20s} | recall={best.recall:.3f} "
                    f"precision={best.precision:.3f} f1={best.f1:.3f} "
                    f"compression={best.compression_ratio:.3f} "
                    f"@ thr={best.threshold:.4f}"
                )

            # Aggregate sweep across all sequences in the category
            if all_gt:
                agg_sweep = threshold_sweep(all_gt, all_scores, n_thresholds=args.n_thresholds)
                recalls, precisions, _ = pr_curve_from_sweep(agg_sweep)
                pr_data[category][det_name] = (recalls, precisions)
                comp_data[category][det_name] = agg_sweep

            # Print category summary for this detector
            if per_seq_best:
                agg = aggregate_metrics(per_seq_best)
                print(
                    f"    >> {det_name} avg: recall={agg['mean_recall']:.3f} "
                    f"precision={agg['mean_precision']:.3f} "
                    f"f1={agg['mean_f1']:.3f} "
                    f"compression={agg['mean_compression_ratio']:.3f}"
                )

        # Save per-category PR curve plot
        if pr_data[category]:
            plot_pr_curves(
                pr_data[category],
                title=f"PR Curve — {category}",
                output_path=plots_dir / f"pr_curve_{category}.png",
                min_recall_line=args.min_recall,
            )
            plot_compression_vs_recall(
                comp_data[category],
                title=f"Compression vs Recall — {category}",
                output_path=plots_dir / f"compression_recall_{category}.png",
                min_recall_line=args.min_recall,
            )
            print(f"  Saved plots for {category}")

    # Write CSVs
    summary_csv = results_dir / "summary.csv"
    if summary_rows:
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nSaved summary: {summary_csv}")

    all_thr_csv = results_dir / "all_thresholds.csv"
    if all_threshold_rows:
        with open(all_thr_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_threshold_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_threshold_rows)
        print(f"Saved threshold sweep: {all_thr_csv}")

    # Cross-detector, cross-category combined PR plot
    if pr_data:
        all_curves: dict[str, tuple] = {}
        for det_name in selected_detectors:
            all_recalls: list[float] = []
            all_precisions: list[float] = []
            for cat_curves in pr_data.values():
                if det_name in cat_curves:
                    r, p = cat_curves[det_name]
                    all_recalls.extend(r)
                    all_precisions.extend(p)
            if all_recalls:
                # sort by recall
                pairs = sorted(zip(all_recalls, all_precisions))
                all_curves[det_name] = (
                    [p[0] for p in pairs],
                    [p[1] for p in pairs],
                )
        if all_curves:
            plot_pr_curves(
                all_curves,
                title="PR Curve — All Categories Combined",
                output_path=plots_dir / "pr_curve_all.png",
                min_recall_line=args.min_recall,
            )
            print(f"Saved combined PR curve: {plots_dir / 'pr_curve_all.png'}")

    print(f"\nAll results saved to: {results_dir}/")
    print("Done.\n")


if __name__ == "__main__":
    main()
