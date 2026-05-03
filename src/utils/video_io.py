"""
Video I/O helpers.

Provides frame reading from image sequences and video files,
plus a consistent BGR -> grayscale conversion used across detectors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np


def to_gray(frame: np.ndarray) -> np.ndarray:
    """Convert BGR frame to grayscale. Passes through if already 2-D."""
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def read_frame(path: Path) -> np.ndarray:
    """Load a single frame from disk (BGR uint8)."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read frame: {path}")
    return img


def iter_image_sequence(
    input_dir: Path,
    pattern: str = "in*.jpg",
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Yield (frame_idx, bgr_frame) pairs from a directory of images.

    Frames are yielded in sorted filename order.
    frame_idx is the integer extracted from the filename (1-based).
    """
    import re
    paths = sorted(Path(input_dir).glob(pattern))
    if not paths and pattern.endswith(".jpg"):
        paths = sorted(Path(input_dir).glob("in*.png"))
    for p in paths:
        m = re.search(r"(\d+)", p.stem)
        idx = int(m.group(1)) if m else 0
        frame = cv2.imread(str(p))
        if frame is not None:
            yield idx, frame


def resize_frame(
    frame: np.ndarray,
    max_dim: int = 640,
) -> np.ndarray:
    """Resize frame so that neither dimension exceeds max_dim, preserving aspect ratio."""
    h, w = frame.shape[:2]
    if max(h, w) <= max_dim:
        return frame
    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def frames_to_video(
    frames: List[np.ndarray],
    output_path: Path,
    fps: float = 10.0,
) -> None:
    """Write a list of BGR frames to an mp4 video file."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
