"""
Precision-Recall curve generation and plotting.

For each detector x category combination, we sweep decision thresholds
and plot the resulting PR curve. The area under the PR curve (AP) is
also reported as a single-number summary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend safe for scripts
import matplotlib.pyplot as plt
import numpy as np

from .metrics import FrameMetrics, threshold_sweep


def pr_curve_from_sweep(
    sweep: List[FrameMetrics],
) -> Tuple[List[float], List[float], List[float]]:
    """
    Extract (recall, precision, threshold) lists from a threshold sweep.

    Sorted by increasing recall for clean plotting.
    """
    pts = sorted(sweep, key=lambda m: m.recall)
    recalls = [m.recall for m in pts]
    precisions = [m.precision for m in pts]
    thresholds = [m.threshold for m in pts]
    return recalls, precisions, thresholds


def average_precision(recalls: Sequence[float], precisions: Sequence[float]) -> float:
    """
    Compute Average Precision (area under PR curve) using the trapezoidal rule.
    """
    r = np.asarray(recalls, dtype=float)
    p = np.asarray(precisions, dtype=float)
    # sort by recall
    idx = np.argsort(r)
    r, p = r[idx], p[idx]
    return float(np.trapezoid(p, r))


def plot_pr_curves(
    curves: Dict[str, Tuple[List[float], List[float]]],
    title: str = "Precision-Recall Curve",
    output_path: Optional[Path] = None,
    min_recall_line: float = 0.95,
    figsize: Tuple[int, int] = (7, 5),
) -> plt.Figure:
    """
    Plot multiple PR curves on one axes.

    Parameters
    ----------
    curves : dict
        Mapping of detector_name -> (recalls, precisions).
    title : str
        Plot title.
    output_path : Path | None
        If provided, save figure to this path.
    min_recall_line : float
        Draw a vertical dashed line at this recall value (operating point marker).
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    for detector_name, (recalls, precisions) in curves.items():
        ap = average_precision(recalls, precisions)
        ax.plot(
            recalls,
            precisions,
            marker=".",
            markersize=3,
            linewidth=1.5,
            label=f"{detector_name} (AP={ap:.3f})",
        )

    ax.axvline(
        x=min_recall_line,
        color="grey",
        linestyle="--",
        linewidth=1.0,
        label=f"Target recall={min_recall_line}",
    )

    ax.set_xlabel("Recall (Frame)", fontsize=12)
    ax.set_ylabel("Precision (Frame)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=120)

    return fig


def plot_compression_vs_recall(
    sweep_data: Dict[str, List[FrameMetrics]],
    title: str = "Compression Ratio vs Recall",
    output_path: Optional[Path] = None,
    min_recall_line: float = 0.95,
    figsize: Tuple[int, int] = (7, 5),
) -> plt.Figure:
    """
    Plot compression ratio vs recall for each detector.

    This shows the operating trade-off clearly: how much can we compress
    while maintaining a target recall level.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for detector_name, sweep in sweep_data.items():
        pts = sorted(sweep, key=lambda m: m.recall)
        recalls = [m.recall for m in pts]
        compressions = [m.compression_ratio for m in pts]
        ax.plot(
            recalls,
            compressions,
            marker=".",
            markersize=3,
            linewidth=1.5,
            label=detector_name,
        )

    ax.axvline(
        x=min_recall_line,
        color="grey",
        linestyle="--",
        linewidth=1.0,
        label=f"Target recall={min_recall_line}",
    )

    ax.set_xlabel("Recall (Frame)", fontsize=12)
    ax.set_ylabel("Compression Ratio (frames dropped / total)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=120)

    return fig
