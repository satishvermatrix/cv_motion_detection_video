"""
Background-subtraction motion detectors.

Wraps OpenCV's MOG2 and KNN background subtractors with the same
interface as the frame-differencing detectors.

These models maintain a statistical background model that adapts over
time, making them more robust to gradual lighting changes and
camera noise than frame differencing.

Motion score = fraction of foreground pixels in the subtractor's output mask.
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import BaseDetector


class MOG2Detector(BaseDetector):
    """
    Gaussian Mixture Model background subtractor (MOG2).

    Automatically selects the number of Gaussian components per pixel and
    handles shadows (marks them as grey in the foreground mask).

    Parameters
    ----------
    history : int
        Number of frames used to build the background model.
    var_threshold : float
        Threshold on the squared Mahalanobis distance. Higher = less sensitive.
    detect_shadows : bool
        If True, shadows are detected and excluded from the motion score.
    learning_rate : float
        Rate at which the background model is updated (-1 = auto).
    """

    name = "mog2"

    def __init__(
        self,
        history: int = 500,
        var_threshold: float = 16.0,
        detect_shadows: bool = True,
        learning_rate: float = -1.0,
    ) -> None:
        self.history = history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
        self.learning_rate = learning_rate
        self._subtractor = self._make_subtractor()

    def _make_subtractor(self) -> cv2.BackgroundSubtractor:
        return cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=self.detect_shadows,
        )

    def reset(self) -> None:
        self._subtractor = self._make_subtractor()

    def compute_score(self, frame: np.ndarray) -> float:
        fg_mask = self._subtractor.apply(frame, learningRate=self.learning_rate)
        # fg_mask values: 255 = foreground, 127 = shadow, 0 = background
        # Shadows are excluded from the motion score
        motion_pixels = fg_mask == 255
        return float(motion_pixels.mean())

    def get_fg_mask(self, frame: np.ndarray) -> np.ndarray:
        """Return the raw foreground mask (for visualisation)."""
        return self._subtractor.apply(frame, learningRate=self.learning_rate)


class KNNDetector(BaseDetector):
    """
    K-Nearest Neighbours background subtractor (KNN).

    More efficient than MOG2 when there are many background pixels that
    are not changing. Generally more sensitive to sudden changes.

    Parameters
    ----------
    history : int
        Number of frames used to build the background model.
    dist2_threshold : float
        Threshold on the squared distance. Higher = less sensitive.
    detect_shadows : bool
        If True, shadows are detected and excluded from the motion score.
    learning_rate : float
        Rate at which the background model is updated (-1 = auto).
    """

    name = "knn"

    def __init__(
        self,
        history: int = 500,
        dist2_threshold: float = 400.0,
        detect_shadows: bool = True,
        learning_rate: float = -1.0,
    ) -> None:
        self.history = history
        self.dist2_threshold = dist2_threshold
        self.detect_shadows = detect_shadows
        self.learning_rate = learning_rate
        self._subtractor = self._make_subtractor()

    def _make_subtractor(self) -> cv2.BackgroundSubtractor:
        return cv2.createBackgroundSubtractorKNN(
            history=self.history,
            dist2Threshold=self.dist2_threshold,
            detectShadows=self.detect_shadows,
        )

    def reset(self) -> None:
        self._subtractor = self._make_subtractor()

    def compute_score(self, frame: np.ndarray) -> float:
        fg_mask = self._subtractor.apply(frame, learningRate=self.learning_rate)
        motion_pixels = fg_mask == 255
        return float(motion_pixels.mean())

    def get_fg_mask(self, frame: np.ndarray) -> np.ndarray:
        return self._subtractor.apply(frame, learningRate=self.learning_rate)
