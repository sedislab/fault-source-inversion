"""The one interface every localizer implements, so FSI and all baselines score through the same harness."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
from ..core.graph import StructuralGraph
from ..core.types import Episode, Operator

@dataclass
class Prediction:
    """A localizer's output for one episode: a fault score per node (higher = more likely the source),
    with optional recovered magnitudes and a (possibly conformal) candidate set."""
    scores: np.ndarray
    magnitudes: np.ndarray | None = None
    candidate_set: list[int] | None = None
    meta: dict = field(default_factory=dict)

    def ranking(self, candidates: list[int] | None = None) -> list[int]:
        idx = np.array(candidates if candidates is not None else range(len(self.scores)))
        return list(idx[np.argsort(-self.scores[idx], kind="stable")])

    def top(self, k: int = 1, candidates: list[int] | None = None) -> list[int]:
        return self.ranking(candidates)[:k]

class Localizer(ABC):
    """Fit on training episodes (fault labels available in sim), then score any episode's nodes."""
    name: str = "localizer"
    needs_operator: bool = False
    needs_series: bool = False

    def fit(self, episodes: list[Episode], graph: StructuralGraph, operator: Operator | None = None) -> "Localizer":
        return self

    @abstractmethod
    def predict(self, episode: Episode, graph: StructuralGraph, operator: Operator | None = None) -> Prediction:
        ...

    def predict_many(self, episodes, graph, operator=None) -> list[Prediction]:
        return [self.predict(e, graph, operator) for e in episodes]
