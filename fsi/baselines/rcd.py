"""Causal RCA (RCD / CausalRCA family, via RCAEval): the fault is an intervention that changes the generative
mechanism of one variable, so the root cause is the channel whose distribution shifts most between a NORMAL and
the ABNORMAL window, after discounting reactive victims whose shift is explained by their graph neighbors.

Tractable reimplementation of the RCD/CausalRCA idea, not the full pipeline: the per-variable change score is a
standardized distributional shift (residual RMS vs a normal reference) and the causal adjustment is a single
graph-adjacency parent-discount, in place of the Psi-PC localized causal discovery (RCD) or the DAG-GNN +
PageRank structure learning (CausalRCA). Unsupervised: the normal reference is learned from normal windows only.
"""
from __future__ import annotations
import numpy as np
from ..localize.base import Localizer, Prediction

class RCD(Localizer):
    name = "rcd"
    needs_series = True

    def __init__(self, gamma: float = 0.6, floor: float = 1e-3):
        self.gamma = gamma
        self.floor = floor
        self.ref_ = None
        self.A_ = None

    def _residual(self, ep):
        if ep.y is None or ep.yhat is None:
            raise ValueError("RCD needs time-series: sample episodes with keep_series=True")
        return ep.y - ep.yhat

    def _onset(self, ep) -> int:
        return int(ep.meta.get("onset_step", ep.y.shape[0] // 2))

    def fit(self, episodes, graph, operator=None) -> "RCD":
        stds = []
        for ep in episodes:
            r = self._residual(ep)
            pre = r[: self._onset(ep)]
            if len(pre) > 1:
                stds.append(pre.std(0))
        if not stds:
            raise ValueError("RCD.fit needs episodes with a normal (pre-onset) window")
        self.ref_ = np.maximum(np.mean(stds, 0), self.floor)
        self.A_ = graph.adjacency()
        return self

    def _change_score(self, ep) -> np.ndarray:
        ref = self.ref_ if self.ref_ is not None else self.floor
        ab = self._residual(ep)[self._onset(ep):]
        rms = np.sqrt((ab ** 2).mean(0))
        return np.clip(rms / (ref + 1e-9) - 1.0, 0, None)

    def predict(self, episode, graph, operator=None) -> Prediction:
        A = self.A_ if self.A_ is not None else graph.adjacency()
        c = self._change_score(episode)
        neigh = A @ c / (A.sum(1) + 1e-9)
        s = np.clip(c - self.gamma * neigh, 0, None)
        return Prediction(scores=s, meta={"kind": "causal_rca", "gamma": self.gamma})
