"""
Frame-level evaluation metrics for the motion-based frame reduction task.

The task is framed as binary classification at the frame level:
  - Positive  (label=1) : frame "has motion" -> should be KEPT
  - Negative  (label=0) : frame is static   -> safe to DROP

Given a list of (gt_label, detector_score) pairs and a decision threshold,
we compute standard binary classification metrics plus compression ratio.

Definitions
-----------
TP : kept a motion frame      (correct keep)
FN : dropped a motion frame   (missed motion - hard failure)
TN : dropped a static frame   (correct drop)
FP : kept a static frame      (wasted budget, but safe)

Recall    = TP / (TP + FN)  <- most important; dropping motion is costly
Precision = TP / (TP + FP)
F1        = 2 * P * R / (P + R)
Compression Ratio = TN / (TN + FP + TP + FN)  <- fraction of frames dropped
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class FrameMetrics:
    """Metrics at a single threshold operating point."""
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int

    recall: float = field(init=False)
    precision: float = field(init=False)
    f1: float = field(init=False)
    compression_ratio: float = field(init=False)
    specificity: float = field(init=False)

    def __post_init__(self) -> None:
        total = self.tp + self.fp + self.tn + self.fn
        self.recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        self.precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        denom = 2 * self.tp + self.fp + self.fn
        self.f1 = (2 * self.tp) / denom if denom > 0 else 0.0
        self.compression_ratio = self.tn / total if total > 0 else 0.0
        self.specificity = self.tn / (self.tn + self.fp) if (self.tn + self.fp) > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "recall": self.recall,
            "precision": self.precision,
            "f1": self.f1,
            "compression_ratio": self.compression_ratio,
            "specificity": self.specificity,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
        }


def compute_metrics(
    gt_labels: Sequence[int],
    scores: Sequence[float],
    threshold: float,
) -> FrameMetrics:
    """
    Compute frame-level metrics at a single threshold.

    Parameters
    ----------
    gt_labels : sequence of int
        Ground-truth binary labels (1 = has motion, 0 = static).
    scores : sequence of float
        Detector motion scores in [0, 1].
    threshold : float
        Decision threshold. Frame is kept (predicted positive) if score >= threshold.

    Returns
    -------
    FrameMetrics
    """
    gt = np.asarray(gt_labels, dtype=int)
    sc = np.asarray(scores, dtype=float)
    pred = (sc >= threshold).astype(int)

    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())

    return FrameMetrics(
        threshold=threshold,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


def threshold_sweep(
    gt_labels: Sequence[int],
    scores: Sequence[float],
    n_thresholds: int = 100,
) -> List[FrameMetrics]:
    """
    Sweep thresholds from 0 to 1 and return a FrameMetrics for each.

    The thresholds are chosen uniformly between the min and max observed
    scores to avoid degenerate all-positive or all-negative regions.
    """
    sc = np.asarray(scores, dtype=float)
    lo, hi = float(sc.min()), float(sc.max())
    if lo == hi:
        # all scores identical; return two extreme points
        return [
            compute_metrics(gt_labels, scores, lo),
            compute_metrics(gt_labels, scores, hi + 1e-9),
        ]
    thresholds = np.linspace(lo, hi, n_thresholds)
    return [compute_metrics(gt_labels, scores, t) for t in thresholds]


def best_operating_point(
    sweep_results: List[FrameMetrics],
    min_recall: float = 0.95,
) -> FrameMetrics:
    """
    Among all operating points with recall >= min_recall,
    return the one with the highest compression ratio.

    Falls back to the point with highest recall if no point meets the target.
    """
    candidates = [m for m in sweep_results if m.recall >= min_recall]
    if not candidates:
        return max(sweep_results, key=lambda m: m.recall)
    return max(candidates, key=lambda m: m.compression_ratio)


def aggregate_metrics(
    per_sequence: List[FrameMetrics],
) -> dict:
    """
    Aggregate a list of per-sequence FrameMetrics into macro-averaged summary.

    Returns a dict of mean values across sequences.
    """
    keys = ["recall", "precision", "f1", "compression_ratio", "specificity"]
    result = {}
    for k in keys:
        vals = [getattr(m, k) for m in per_sequence]
        result[f"mean_{k}"] = float(np.mean(vals))
        result[f"std_{k}"] = float(np.std(vals))
    result["n_sequences"] = len(per_sequence)
    return result
