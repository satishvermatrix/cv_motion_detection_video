"""
CDnet 2014 dataset loader.

Handles loading sequences, ground-truth masks, ROI masks, and deriving
frame-level binary motion labels from pixel-level annotations.

CDnet GT pixel label values:
  0   -> static (background)
  50  -> shadow  (treated as background in metrics)
  85  -> outside ROI  (ignored)
  170 -> unknown / motion-blurred  (ignored)
  255 -> foreground (motion)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np

LABEL_STATIC = 0
LABEL_SHADOW = 50
LABEL_NONROI = 85
LABEL_UNKNOWN = 170
LABEL_FOREGROUND = 255

# Pixels considered "valid" for evaluation (not ignored)
VALID_LABELS = {LABEL_STATIC, LABEL_SHADOW, LABEL_FOREGROUND}


@dataclass
class FrameRecord:
    """One frame worth of data from a CDnet sequence."""
    frame_idx: int          # 1-based frame number (matches filename)
    input_path: Path
    gt_path: Optional[Path]
    in_temporal_roi: bool   # False for warmup frames outside evaluation window

    # Lazily loaded
    _input: Optional[np.ndarray] = field(default=None, repr=False)
    _gt_mask: Optional[np.ndarray] = field(default=None, repr=False)

    def load_input(self) -> np.ndarray:
        if self._input is None:
            self._input = cv2.imread(str(self.input_path))
            if self._input is None:
                raise FileNotFoundError(f"Cannot read input: {self.input_path}")
        return self._input

    def load_gt_mask(self) -> Optional[np.ndarray]:
        """Returns raw uint8 GT mask or None if no GT is available."""
        if self._gt_mask is None and self.gt_path is not None:
            self._gt_mask = cv2.imread(str(self.gt_path), cv2.IMREAD_GRAYSCALE)
            if self._gt_mask is None:
                raise FileNotFoundError(f"Cannot read GT: {self.gt_path}")
        return self._gt_mask


@dataclass
class CDnetSequence:
    """
    One video sequence from a CDnet category.

    Attributes
    ----------
    category : str
        e.g. "baseline"
    name : str
        e.g. "highway"
    root : Path
        Path to sequence root (contains input/, groundtruth/, ROI.bmp, ...)
    temporal_roi : Tuple[int, int]
        (start_frame, end_frame) inclusive, 1-based
    roi_mask : np.ndarray | None
        Binary spatial ROI mask (H x W, uint8, 255 = inside ROI)
    """
    category: str
    name: str
    root: Path
    temporal_roi: Tuple[int, int]
    roi_mask: Optional[np.ndarray]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, seq_root: Path, category: str = "") -> "CDnetSequence":
        seq_root = Path(seq_root)
        if not seq_root.exists():
            raise FileNotFoundError(f"Sequence root not found: {seq_root}")

        # temporal ROI
        troi_path = seq_root / "temporalROI.txt"
        if troi_path.exists():
            parts = troi_path.read_text().strip().split()
            temporal_roi = (int(parts[0]), int(parts[1]))
        else:
            temporal_roi = (1, 999_999)

        # spatial ROI
        roi_path = seq_root / "ROI.bmp"
        roi_mask = None
        if roi_path.exists():
            roi_mask = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)

        return cls(
            category=category or seq_root.parent.name,
            name=seq_root.name,
            root=seq_root,
            temporal_roi=temporal_roi,
            roi_mask=roi_mask,
        )

    # ------------------------------------------------------------------
    # Frame iteration
    # ------------------------------------------------------------------
    def _input_frames(self) -> List[Path]:
        input_dir = self.root / "input"
        frames = sorted(input_dir.glob("in*.jpg"))
        if not frames:
            frames = sorted(input_dir.glob("in*.png"))
        return frames

    def _gt_for_idx(self, idx: int) -> Optional[Path]:
        gt_dir = self.root / "groundtruth"
        p = gt_dir / f"gt{idx:06d}.png"
        return p if p.exists() else None

    def _frame_idx_from_path(self, p: Path) -> int:
        m = re.search(r"(\d+)", p.stem)
        return int(m.group(1)) if m else 0

    def iter_frames(
        self,
        eval_only: bool = True,
    ) -> Iterator[FrameRecord]:
        """
        Iterate over frames in the sequence.

        Parameters
        ----------
        eval_only : bool
            If True (default), skip frames outside the temporal ROI window.
        """
        start, end = self.temporal_roi
        for input_path in self._input_frames():
            idx = self._frame_idx_from_path(input_path)
            in_roi = start <= idx <= end
            if eval_only and not in_roi:
                continue
            yield FrameRecord(
                frame_idx=idx,
                input_path=input_path,
                gt_path=self._gt_for_idx(idx),
                in_temporal_roi=in_roi,
            )

    def __len__(self) -> int:
        start, end = self.temporal_roi
        return end - start + 1

    def __repr__(self) -> str:
        return (
            f"CDnetSequence({self.category}/{self.name}, "
            f"temporal_roi={self.temporal_roi})"
        )


# ---------------------------------------------------------------------------
# Ground-truth helpers
# ---------------------------------------------------------------------------

def _align_roi(roi_mask: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Resize ROI mask to match target shape if they differ."""
    if roi_mask.shape == target.shape:
        return roi_mask
    return cv2.resize(
        roi_mask,
        (target.shape[1], target.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )


def gt_foreground_count(
    gt_mask: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
) -> int:
    """Number of foreground (motion) pixels inside the ROI."""
    fg = gt_mask == LABEL_FOREGROUND
    if roi_mask is not None:
        roi = _align_roi(roi_mask, gt_mask)
        fg = fg & (roi > 0)
    return int(fg.sum())


def gt_valid_pixel_count(
    gt_mask: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
) -> int:
    """Number of pixels that are valid for evaluation (not ignored)."""
    valid = np.isin(gt_mask, list(VALID_LABELS))
    if roi_mask is not None:
        roi = _align_roi(roi_mask, gt_mask)
        valid = valid & (roi > 0)
    return int(valid.sum())


def derive_frame_label(
    gt_mask: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    min_foreground_pixels: int = 200,
) -> bool:
    """
    Returns True if the frame is considered to "have motion".

    A frame has motion if at least `min_foreground_pixels` foreground pixels
    exist inside the ROI.
    """
    return gt_foreground_count(gt_mask, roi_mask) >= min_foreground_pixels


# ---------------------------------------------------------------------------
# Dataset-level loader
# ---------------------------------------------------------------------------

def load_category(
    dataset_root: Path,
    category: str,
) -> List[CDnetSequence]:
    """Load all sequences from one CDnet category."""
    cat_path = Path(dataset_root) / category
    if not cat_path.exists():
        raise FileNotFoundError(f"Category not found: {cat_path}")
    sequences = []
    for seq_dir in sorted(cat_path.iterdir()):
        if seq_dir.is_dir() and (seq_dir / "input").exists():
            sequences.append(CDnetSequence.load(seq_dir, category=category))
    return sequences


def load_dataset(
    dataset_root: Path,
    categories: Optional[List[str]] = None,
) -> dict[str, List[CDnetSequence]]:
    """
    Load sequences grouped by category.

    Parameters
    ----------
    dataset_root : Path
        Root of the CDnet dataset (contains category folders).
    categories : list[str] | None
        Specific categories to load. If None, loads all available.

    Returns
    -------
    dict mapping category name -> list of CDnetSequence
    """
    dataset_root = Path(dataset_root)
    if categories is None:
        categories = [
            d.name for d in sorted(dataset_root.iterdir())
            if d.is_dir()
        ]
    result = {}
    for cat in categories:
        try:
            result[cat] = load_category(dataset_root, cat)
        except FileNotFoundError:
            pass
    return result
