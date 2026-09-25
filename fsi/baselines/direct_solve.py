"""Direct sparse solve against our own operator G: the critical non-learned ablation. Same G as FSI, same
inverse-source objective, but solved iteratively per episode instead of amortized. FISTA by default (fast,
scalable); cvxpy exact optional. This isolates the value of amortization and of learning the nonlinear inverse."""
from __future__ import annotations
import numpy as np
from ..localize.base import Localizer, Prediction

def _fista(G, r, lam, iters=200):
    """Nonnegative sparse recovery: min 0.5||r-Gs||^2 + lam||s||_1, s>=0."""
    n = G.shape[1]
    L = float(np.linalg.norm(G, 2) ** 2) + 1e-8
    step = 1.0 / L
    s = np.zeros(n); z = s.copy(); t = 1.0
    for _ in range(iters):
        grad = G.T @ (G @ z - r)
        s_new = np.clip(z - step * grad - step * lam, 0, None)
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        z = s_new + ((t - 1) / t_new) * (s_new - s)
        s, t = s_new, t_new
    return s

class DirectSolve(Localizer):
    needs_operator = True

    def __init__(self, lam=0.05, iters=200, mode="fista"):
        self.lam = lam; self.iters = iters; self.mode = mode
        self.name = f"direct_{'l1' if lam > 0 else 'nnls'}"

    def predict(self, episode, graph, operator=None) -> Prediction:
        assert operator is not None, "DirectSolve needs the operator G"
        G, r = operator.G, episode.energy().astype(float)
        if self.mode == "cvxpy":
            s = self._cvxpy(G, r)
        else:
            s = _fista(G, r, self.lam, self.iters)
        return Prediction(scores=s, magnitudes=s, meta={"kind": "inverse", "solver": self.mode})

    def _cvxpy(self, G, r):
        import cvxpy as cp
        s = cp.Variable(G.shape[1], nonneg=True)
        prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(G @ s - r) + self.lam * cp.norm1(s)))
        prob.solve(solver=cp.OSQP)
        return np.clip(np.asarray(s.value).ravel(), 0, None) if s.value is not None else np.zeros(G.shape[1])
