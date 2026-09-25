"""Largest-residual rule: name the loudest node. The naive victim-style localizer FSI must beat on propagation."""
from __future__ import annotations
import numpy as np
from ..localize.base import Localizer, Prediction

class LargestResidual(Localizer):
    name = "largest_residual"

    def predict(self, episode, graph, operator=None) -> Prediction:
        e = episode.energy().astype(float)
        return Prediction(scores=e.copy(), meta={"kind": "victim"})
