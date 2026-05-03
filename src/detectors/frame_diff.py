"""
Frame-differencing motion detectors.

Two variants:
  - RawFrameDiff   : absolute grayscale difference between consecutive frames
  - BlurredFrameDiff: same but applies Gaussian blur before differencing
                      to suppress noise / compression artifacts

Both return a motion_score in [0, 1]: the fraction of pixels whose
inter-frame difference exceeds `pixel_threshold`.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .base import BaseDetector
from ..utils.video_io import to_gray


class RawFrameDiff(BaseDetector):
    """
    Motion score = fraction of pixels with |frame[t] - frame[t-1]| > pixel_threshold.

    Parameters
    ----------
    pixel_threshold : int
        Per-pixel absolute difference threshold (0–255).
        Lower → more sensitive; higher → less sensitive to noise.
    """

    name = "frame_diff_raw"

    def __init__(self, pixel_threshold: int = 25) -> None:
        self.pixel_threshold = pixel_threshold
        self._prev_gray: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev_gray = None

    def compute_score(self, frame: np.ndarray) -> float:
        gray = to_gray(frame)
        if self._prev_gray is None:
            self._prev_gray = gray
            return 0.0

        diff = cv2.absdiff(gray, self._prev_gray)
        motion_mask = diff > self.pixel_threshold
        score = float(motion_mask.mean())

        self._prev_gray = gray
        return score

    def get_diff_frame(self, frame: np.ndarray) -> np.ndarray:
        """Return the raw difference image (for visualisation)."""
        gray = to_gray(frame)
        if self._prev_gray is None:
            return np.zeros_like(gray)
        diff = cv2.absdiff(gray, self._prev_gray)
        return diff


class BlurredFrameDiff(BaseDetector):
    """
    Same as RawFrameDiff but applies a Gaussian blur to each frame before
    differencing. Blur reduces high-frequency noise and JPEG compression
    artifacts, giving a cleaner motion signal.

    Parameters
    ----------
    pixel_threshold : int
        Per-pixel absolute difference threshold after blurring.
    blur_ksize : int
        Gaussian kernel size (must be odd). Default 5.
    """

    name = "frame_diff_blurred"

    def __init__(
        self,
        pixel_threshold: int = 20,
        blur_ksize: int = 5,
    ) -> None:
        self.pixel_threshold = pixel_threshold
        self.blur_ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        self._prev_blurred: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev_blurred = None

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = to_gray(frame)
        return cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)

    def compute_score(self, frame: np.ndarray) -> float:
        blurred = self._preprocess(frame)

        if self._prev_blurred is None:
            self._prev_blurred = blurred
            return 0.0

        diff = cv2.absdiff(blurred, self._prev_blurred)
        motion_mask = diff > self.pixel_threshold
        score = float(motion_mask.mean())

        self._prev_blurred = blurred
        return score
