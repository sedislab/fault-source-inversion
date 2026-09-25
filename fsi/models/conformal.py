"""Distribution-free conformal support sets. APS gives a set that contains the single true source with
probability >= 1-alpha; Conformal Risk Control bounds the expected miss-rate (FNR) at alpha for k>=1. The
calibration unit is one episode (never a timestep). Mondrian groups (embodiment x fault-type) get per-group
thresholds so coverage holds within each group, not just marginally."""
from __future__ import annotations
import numpy as np
from ..localize.base import Localizer

def _dist(scores, fail):
    p = np.clip(scores[fail], 0, None)
    return p / (p.sum() + 1e-12)

def aps_calibrate(preds, episodes, fail, alpha):
    fail = np.asarray(fail)
    scores = []
    for pred, ep in zip(preds, episodes):
        if ep.is_healthy:
            continue
        d = _dist(pred.scores, fail)
        order = np.argsort(-d)
        pos = int(np.where(fail[order] == ep.support()[0])[0][0])
        scores.append(float(d[order][:pos + 1].sum()))
    n = len(scores)
    q = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, q, method="higher"))

def aps_set(pred, fail, qhat):
    fail = np.asarray(fail)
    d = _dist(pred.scores, fail)
    order = np.argsort(-d)
    k = int(np.searchsorted(np.cumsum(d[order]), qhat) + 1)
    return list(fail[order][:min(k, len(fail))])

def crc_calibrate(preds, episodes, fail, alpha, grid=64):
    """Smallest threshold-set with E[FNR] bounded by alpha (Angelopoulos et al. conformal risk control)."""
    fail = np.asarray(fail)
    P, S = [], []
    for pred, ep in zip(preds, episodes):
        if ep.is_healthy:
            continue
        P.append(np.clip(pred.scores[fail], 0, None))
        S.append([int(np.where(fail == v)[0][0]) for v in ep.support()])
    P = np.asarray(P); n = len(P)
    hi = float(P.max()) if n else 1.0
    for thr in np.linspace(hi, 0, grid):
        risk = np.mean([sum(1 for v in s if p[v] < thr) / len(s) for p, s in zip(P, S)])
        if (n / (n + 1)) * risk + 1 / (n + 1) <= alpha:
            return float(thr)
    return 0.0

def crc_set(pred, fail, thr):
    fail = np.asarray(fail)
    sel = fail[np.clip(pred.scores[fail], 0, None) >= thr]
    return list(sel) if len(sel) else [int(fail[np.argmax(pred.scores[fail])])]

class ConformalSupport:
    """Calibrate on held-out episodes, then emit a guaranteed candidate set per episode. mode 'aps' (k=1
    coverage) or 'crc' (k>=1 FNR control). group_by yields Mondrian per-group thresholds."""
    def __init__(self, localizer: Localizer, mode="crc", alpha=0.1, group_by=None):
        self.loc = localizer; self.mode = mode; self.alpha = alpha; self.group_by = group_by
        self.thr = {}

    def _key(self, ep):
        return self.group_by(ep) if self.group_by else "_all"

    def calibrate(self, episodes, graph, operator=None):
        preds = self.loc.predict_many(episodes, graph, operator)
        fail = graph.fail_nodes
        groups = {}
        for pred, ep in zip(preds, episodes):
            groups.setdefault(self._key(ep), ([], []))
            groups[self._key(ep)][0].append(pred); groups[self._key(ep)][1].append(ep)
        for key, (ps, es) in groups.items():
            if self.mode == "aps":
                self.thr[key] = aps_calibrate(ps, es, fail, self.alpha)
            else:
                self.thr[key] = crc_calibrate(ps, es, fail, self.alpha)
        self._fail = fail
        return self

    def predict_set(self, ep, graph, operator=None):
        pred = self.loc.predict(ep, graph, operator)
        thr = self.thr.get(self._key(ep), next(iter(self.thr.values())))
        return (aps_set if self.mode == "aps" else crc_set)(pred, self._fail, thr)

    def evaluate(self, episodes, graph, operator=None):
        cov, size, miss = [], [], []
        for ep in episodes:
            if ep.is_healthy:
                continue
            S = set(self.predict_set(ep, graph, operator)); supp = set(ep.support())
            cov.append(float(supp <= S) if self.mode == "aps" else None)
            miss.append(len(supp - S) / len(supp))
            size.append(len(S))
        out = {"avg_set_size": float(np.mean(size)), "fnr": float(np.mean(miss)),
               "coverage": 1 - float(np.mean(miss))}
        c = [x for x in cov if x is not None]
        if c:
            out["set_coverage"] = float(np.mean(c))
        return out
