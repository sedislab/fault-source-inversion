"""Degenerate reference scorers. These carry no information about the source and exist to pin the floor of every
metric in the results table.

They are not optional. The metric they replaced (CvV) was retired precisely because nobody had measured what a
degenerate scorer got: ranking nodes by NEGATIVE residual energy scored CvV 1.000 while localizing almost nothing,
and a constant scorer scored exactly 0.500 -- above the proposed method's 0.505. A reader cannot tell a real
margin from a do-nothing floor unless the floor is in the table, so these rows always run."""
from __future__ import annotations
import numpy as np
from ..localize.base import Localizer, Prediction

class ConstantScore(Localizer):
    """Every node scored identically: the pure no-information floor. Ties are broken by the harness's stable
    argsort, so this reports whatever accuracy naming the lowest-indexed candidate happens to get."""
    name = "zz_constant"

    def predict(self, episode, graph, operator=None) -> Prediction:
        return Prediction(scores=np.zeros(episode.n), meta={"kind": "reference"})

class AntiEnergy(Localizer):
    """Rank by NEGATIVE residual energy -- name the quietest node. The scorer that broke CvV; kept as a standing
    check that no future metric can be won by simply inverting the energy ranking."""
    name = "zz_anti_energy"

    def predict(self, episode, graph, operator=None) -> Prediction:
        return Prediction(scores=-episode.energy().astype(float), meta={"kind": "reference"})

class RandomScore(Localizer):
    """Uniform random scores at a fixed seed: the empirical chance level, including any candidate-set effects."""
    name = "zz_random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def predict(self, episode, graph, operator=None) -> Prediction:
        return Prediction(scores=self.rng.random(episode.n), meta={"kind": "reference"})
