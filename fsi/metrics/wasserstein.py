"""1-Wasserstein distance between two node distributions under the graph hop metric, via a transport LP.

Dependency-free (scipy highs). Small graphs, so the dense LP is fine and exact."""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog

def graph_wasserstein(p: np.ndarray, q: np.ndarray, D: np.ndarray) -> float:
    """min_T <T, D> s.t. T>=0, T1 = p, T^T1 = q. p, q are distributions on the same nodes; D is the hop cost."""
    p = np.asarray(p, float); q = np.asarray(q, float); D = np.asarray(D, float)
    n = len(p)
    if abs(p.sum() - q.sum()) > 1e-6:
        p = p / (p.sum() + 1e-12); q = q / (q.sum() + 1e-12)
    supp_p = np.where(p > 1e-12)[0]
    supp_q = np.where(q > 1e-12)[0]
    if len(supp_p) == 0 or len(supp_q) == 0:
        return float("nan")
    cost = D[np.ix_(supp_p, supp_q)].ravel()
    a, b = p[supp_p], q[supp_q]
    na, nb = len(a), len(b)
    A_eq, b_eq = [], []
    for i in range(na):
        row = np.zeros(na * nb); row[i * nb:(i + 1) * nb] = 1.0
        A_eq.append(row); b_eq.append(a[i])
    for j in range(nb):
        row = np.zeros(na * nb); row[j::nb] = 1.0
        A_eq.append(row); b_eq.append(b[j])
    res = linprog(cost, A_eq=np.asarray(A_eq), b_eq=np.asarray(b_eq), bounds=(0, None), method="highs")
    return float(res.fun) if res.success else float(cost.min())
