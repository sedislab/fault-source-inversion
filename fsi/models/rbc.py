"""Reconstruction-Based Contribution: rank each node by how much of the residual field is explained by ITS OWN
operator column, rather than by how loud the node is. Ranking by residual magnitude is a contribution plot, which
provably smears a fault onto correlated healthy nodes; RBC has a Cauchy-Schwarz guarantee that the faulty node
scores highest for a sufficiently large single fault (Alcala & Qin, Automatica 2009).

    RBC_j = (g_j^T M r)^2 / (g_j^T M g_j)

with g_j the j-th operator column and M the inverse residual covariance. The field is already whitened by
standardize_fields, so M = I unless healthy episodes are supplied to estimate a diagonal M. Training-free."""
from __future__ import annotations
import numpy as np
from ..localize.base import Localizer, Prediction

def _stack(op, r):
    """Stacked operator columns and field over the fidelity channels: (nC, n), (nC,)."""
    if getattr(op, "G_multi", None) is not None:
        Gm = op.G_multi
        C = Gm.shape[0]
        g = np.concatenate([Gm[c] for c in range(C)], 0)
        v = np.concatenate([r[:, 1 + c] for c in range(C)], 0)
        return g, v
    return op.G, (r[:, 0] if r.ndim == 2 else r)

class RBCLocalizer(Localizer):
    name = "fsi_rbc"
    needs_operator = True

    def __init__(self, whiten=True, eps=1e-9):
        self.whiten = whiten
        self.eps = eps
        self.minv = None

    def fit(self, episodes, graph, operator=None):
        if self.whiten:
            H = [e for e in episodes if e.is_healthy]
            if H:
                V = np.stack([_stack(operator, e.r)[1] for e in H])
                self.minv = 1.0 / (V.std(0) + 1e-3)          # diagonal M^(1/2): whiten by healthy variability
        return self

    def predict(self, episode, graph, operator=None) -> Prediction:
        g, v = _stack(operator, episode.r)
        if self.minv is not None:
            g = g * self.minv[:, None]
            v = v * self.minv
        num = (g.T @ v) ** 2
        den = (g * g).sum(0) + self.eps
        scores = num / den
        dead = (g * g).sum(0) < self.eps * 10                # nodes with no fingerprint cannot be blamed
        # Finite sentinel, not -inf: the harness rejects non-finite scores (that check is what catches a baseline
        # silently emitting NaN), and ranking last only needs a value below every real score.
        scores[dead] = scores[~dead].min() - 1.0 if (~dead).any() else 0.0
        return Prediction(scores=scores.astype(float), meta={"kind": "rbc"})
