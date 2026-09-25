"""Localization metrics. CvV is the primary "did we find the cause, not the victim" score; the rest are the
graph-Wasserstein error and the ranking canon, all scored over failable nodes through one harness."""
from __future__ import annotations
import numpy as np
from ..core.graph import StructuralGraph
from ..core.types import Episode, Operator
from ..localize.base import Prediction
from .wasserstein import graph_wasserstein

def _cand(graph: StructuralGraph, candidates=None) -> np.ndarray:
    return np.asarray(candidates if candidates is not None else graph.fail_nodes)

def _ranking(pred: Prediction, cand: np.ndarray) -> list[int]:
    return list(cand[np.argsort(-pred.scores[cand], kind="stable")])

def top1(pred, ep, graph, candidates=None) -> float | None:
    if ep.is_healthy:
        return None
    rank = _ranking(pred, _cand(graph, candidates))
    return float(rank[0] in set(ep.support()))

def hit_at_k(pred, ep, graph, k=3, candidates=None) -> float | None:
    if ep.is_healthy:
        return None
    rank = _ranking(pred, _cand(graph, candidates))
    return float(bool(set(rank[:k]) & set(ep.support())))

def mrr(pred, ep, graph, candidates=None) -> float | None:
    if ep.is_healthy:
        return None
    rank = _ranking(pred, _cand(graph, candidates))
    supp = set(ep.support())
    for i, v in enumerate(rank):
        if v in supp:
            return 1.0 / (i + 1)
    return 0.0

def ac_at_k(pred, ep, graph, k=3, candidates=None) -> float | None:
    """Recall of the true support within the top-k (RCA canon)."""
    if ep.is_healthy:
        return None
    rank = _ranking(pred, _cand(graph, candidates))
    supp = set(ep.support())
    return len(set(rank[:k]) & supp) / len(supp)

def avg_at_k(pred, ep, graph, k=5, candidates=None) -> float | None:
    if ep.is_healthy:
        return None
    return float(np.mean([ac_at_k(pred, ep, graph, j, candidates) for j in range(1, k + 1)]))

def map_at_k(pred, ep, graph, k=5, candidates=None) -> float | None:
    if ep.is_healthy:
        return None
    rank = _ranking(pred, _cand(graph, candidates))
    supp = set(ep.support())
    hits, precs = 0, []
    for i, v in enumerate(rank[:k]):
        if v in supp:
            hits += 1; precs.append(hits / (i + 1))
    return float(np.mean(precs)) if precs else 0.0

def hop_hit_at_k(pred, ep, graph, k=1, lam=1.0, candidates=None) -> float | None:
    """Hop-discounted Hit@k: credit near-misses by graph distance. lam->0 recovers exact AC@1."""
    if ep.is_healthy:
        return None
    D = graph.hop_distance()
    rank = _ranking(pred, _cand(graph, candidates))
    supp = ep.support()
    return float(max(np.exp(-min(D[v, s] for s in supp) / lam) for v in rank[:k]))

def graph_wasserstein_error(pred, ep, graph, candidates=None) -> float | None:
    """Optimal-transport distance (in hops) from the predicted source distribution to the true one."""
    if ep.is_healthy:
        return None
    cand = _cand(graph, candidates)
    p = np.zeros(graph.n)
    sc = np.clip(pred.scores[cand], 0, None)
    p[cand] = sc / (sc.sum() + 1e-12) if sc.sum() > 0 else 1.0 / len(cand)
    q = ep.source_vector().astype(float); q = q / (q.sum() + 1e-12)
    return graph_wasserstein(p, q, graph.hop_distance())

def support_f1(pred, ep, graph, candidates=None) -> float | None:
    """Predict the top-|support| nodes and F1 against the true support (multi-fault recovery)."""
    if ep.is_healthy:
        return None
    supp = set(ep.support())
    pset = set(_ranking(pred, _cand(graph, candidates))[:len(supp)])
    tp = len(pset & supp)
    prec = tp / len(pset) if pset else 0.0
    rec = tp / len(supp)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

def is_hard(ep, graph, candidates=None) -> bool:
    """A cause-over-victim episode: the loudest candidate node is NOT the source, so naming the loudest node is
    guaranteed wrong and only a method that actually inverts propagation can succeed."""
    cand = _cand(graph, candidates)
    return int(cand[np.argmax(ep.energy()[cand])]) not in set(ep.support())

def hard_top1(preds, episodes, graph, candidates=None) -> dict:
    """Top-1 accuracy restricted to cause-over-victim episodes. This replaces the earlier CvV score, which was
    not a valid metric: CvV selected its pairs with `energy[victim] >= energy[source]` and then asked whether the
    source outranked the victim, so it was a pure inversion of the energy ranking on exactly the pairs where
    inverting wins. Ranking nodes by NEGATIVE residual energy scored CvV 1.000 while localizing almost nothing
    (top-1 0.065), a constant scorer scored exactly 0.500, and the loudest-node rule scored 0.000 rather than the
    <=0.5 its docstring claimed -- so a method sitting at 0.505 was at the do-nothing floor, not mid-field.

    Restricting top-1 to hard episodes keeps the intended semantics and cannot be gamed: the localizer must still
    name the source among ALL candidates, so chance is 1/|candidates|, the loudest-node rule scores 0 by
    construction, anti-energy scores near 0, and only genuine cause-finding scores high."""
    cand = _cand(graph, candidates)
    vals = [top1(p, e, graph, candidates) for p, e in zip(preds, episodes)
            if not e.is_healthy and is_hard(e, graph, candidates)]
    n_fault = sum(1 for e in episodes if not e.is_healthy)
    return {"hard_top1": float(np.mean(vals)) if vals else float("nan"), "n_hard": len(vals),
            "hard_frac": len(vals) / n_fault if n_fault else float("nan"),
            "n_fault": n_fault, "chance": 1.0 / len(cand)}

_FAMILY_PREFIX = (("enc_", "encoder"), ("imu_", "imu"), ("contact_", "foot"), ("sensor_", "sensor"))

def fault_family(ep) -> str | None:
    """Which family a fault episode belongs to. Read from the label when the generator recorded one, else from
    the kind's prefix. The families answer different questions and must never be pooled into one accuracy: an
    observation-layer fault (imu, foot contact) does not propagate, so its source IS the loudest node and it is
    EASY by construction, while an in-loop fault (actuator, encoder) has to be inverted. Averaging them reports
    a number that moves when the mix changes rather than when a method improves."""
    if ep.is_healthy:
        return None
    fam = ep.meta.get("family")
    if fam:
        return str(fam)
    k = ep.faults[0].fault_type
    return next((name for pre, name in _FAMILY_PREFIX if k.startswith(pre)), "actuator")

def per_family(preds, episodes, graph, candidates=None) -> dict:
    """top1 / hard_top1 / episode counts per fault family, flattened into `top1_<family>` style keys."""
    out = {}
    fams = sorted({f for e in episodes if (f := fault_family(e)) is not None})
    for fam in fams:
        sel = [(p, e) for p, e in zip(preds, episodes) if fault_family(e) == fam]
        vals = [top1(p, e, graph, candidates) for p, e in sel]
        hard = [top1(p, e, graph, candidates) for p, e in sel if is_hard(e, graph, candidates)]
        out[f"top1_{fam}"] = float(np.mean(vals)) if vals else float("nan")
        out[f"hard_top1_{fam}"] = float(np.mean(hard)) if hard else float("nan")
        out[f"n_{fam}"] = len(sel)
    return out

def evaluate_localization(preds, episodes, graph, operator=None, ks=(1, 3, 5), lam=1.0, candidates=None,
                          families=True) -> dict:
    out = {}
    per = {"top1": top1, "mrr": mrr, "w1g": graph_wasserstein_error, "support_f1": support_f1}
    for name, fn in per.items():
        vals = [fn(p, e, graph, candidates) for p, e in zip(preds, episodes)]
        vals = [v for v in vals if v is not None]
        out[name] = float(np.mean(vals)) if vals else float("nan")
    for k in ks:
        for name, fn in (("hit", hit_at_k), ("ac", ac_at_k)):
            vals = [fn(p, e, graph, k, candidates) for p, e in zip(preds, episodes)]
            vals = [v for v in vals if v is not None]
            out[f"{name}@{k}"] = float(np.mean(vals)) if vals else float("nan")
    hh = [hop_hit_at_k(p, e, graph, 1, lam, candidates) for p, e in zip(preds, episodes)]
    hh = [v for v in hh if v is not None]
    out["hop_hit@1"] = float(np.mean(hh)) if hh else float("nan")
    out.update(hard_top1(preds, episodes, graph, candidates))
    if families:
        out.update(per_family(preds, episodes, graph, candidates))
    return out
