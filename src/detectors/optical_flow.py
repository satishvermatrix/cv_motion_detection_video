"""
Optical-flow based motion detector (Farneback dense flow).

Unlike frame differencing, optical flow estimates the actual motion
direction and magnitude at each pixel, making it more robust to
uniform illumination changes (which cause large diffs but zero flow).

Motion score = mean flow magnitude across all pixels, normalized to [0, 1]
by a configurable `max_magnitude` parameter (default 20 px/frame).
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .base import BaseDetector
from ..utils.video_io import to_gray


class FarnebackDetector(BaseDetector):
    """
    Dense optical flow (Farneback) motion detector.

    Parameters
    ----------
    pyr_scale : float
        Image scale (<1) to build pyramids. 0.5 means a classical pyramid.
    levels : int
        Number of pyramid layers.
    winsize : int
        Averaging window size. Larger = smoother but less detailed flow.
    iterations : int
        Number of iterations per pyramid level.
    poly_n : int
        Size of pixel neighbourhood for polynomial expansion (5 or 7).
    poly_sigma : float
        Standard deviation of Gaussian for polynomial smoothing.
    max_magnitude : float
        Flow magnitude value considered "full motion" (used for normalisation).
        Pixels with magnitude >= max_magnitude contribute 1.0 to the score.
    score_mode : str
        "mean_norm"  : mean of per-pixel magnitude / max_magnitude (default)
        "frac_above" : fraction of pixels with magnitude > motion_threshold
    motion_threshold : float
        Used only when score_mode="frac_above". Minimum flow magnitude to
        count a pixel as moving.
    """

    name = "optical_flow_farneback"

    def __init__(
        self,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
        max_magnitude: float = 20.0,
        score_mode: str = "mean_norm",
        motion_threshold: float = 1.0,
    ) -> None:
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.max_magnitude = max_magnitude
        self.score_mode = score_mode
        self.motion_threshold = motion_threshold
        self._prev_gray: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev_gray = None

    def _compute_flow(
        self, prev: np.ndarray, curr: np.ndarray
    ) -> np.ndarray:
        flow = cv2.calcOpticalFlowFarneback(
            prev,
            curr,
            None,
            self.pyr_scale,
            self.levels,
            self.winsize,
            self.iterations,
            self.poly_n,
            self.poly_sigma,
            0,
        )
        return flow  # shape (H, W, 2), dtype float32

    def compute_score(self, frame: np.ndarray) -> float:
        gray = to_gray(frame)

        if self._prev_gray is None:
            self._prev_gray = gray
            return 0.0

        flow = self._compute_flow(self._prev_gray, gray)
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

        if self.score_mode == "frac_above":
            score = float((magnitude > self.motion_threshold).mean())
        else:  # mean_norm
            score = float(np.clip(magnitude / self.max_magnitude, 0.0, 1.0).mean())

        self._prev_gray = gray
        return score

    def get_flow_magnitude(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Return a float32 magnitude map for the current frame.
        Returns None on the first frame (no previous frame available).
        """
        gray = to_gray(frame)
        if self._prev_gray is None:
            self._prev_gray = gray
            return None
        flow = self._compute_flow(self._prev_gray, gray)
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        self._prev_gray = gray
        return magnitude

    def get_flow_hsv_vis(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Return a BGR visualisation of optical flow using HSV colour coding.
        Hue encodes direction, value encodes magnitude.
        Returns None on the first frame.
        """
        gray = to_gray(frame)
        if self._prev_gray is None:
            self._prev_gray = gray
            return None

        flow = self._compute_flow(self._prev_gray, gray)
        self._prev_gray = gray

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((*gray.shape, 3), dtype=np.uint8)
        hsv[..., 0] = angle * 180 / np.pi / 2   # hue: direction
        hsv[..., 1] = 255                         # saturation: full
        hsv[..., 2] = cv2.normalize(             # value: magnitude
            magnitude, None, 0, 255, cv2.NORM_MINMAX
        )
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
