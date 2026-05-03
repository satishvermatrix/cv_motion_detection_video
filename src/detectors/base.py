"""
Base class for all motion detectors.

Every detector implements the same interface so the evaluation harness
can treat them interchangeably.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseDetector(ABC):
    """
    Interface contract for motion detectors.

    Detectors are stateful (they may maintain background models or
    previous frames). Call `reset()` at the start of each new sequence.
    """

    name: str = "base"

    @abstractmethod
    def compute_score(self, frame: np.ndarray) -> float:
        """
        Given a BGR frame, return a motion score in [0, 1].

        0 = no motion detected, 1 = entire frame is in motion.
        The caller applies a threshold to decide keep vs. drop.
        """
        ...

    def reset(self) -> None:
        """Reset any internal state (previous frame, background model, etc.)."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
