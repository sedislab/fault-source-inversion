"""Detection scored honestly: threshold calibrated on held-out healthy data only, event-wise F1 and AUPRC
without point adjustment, and a false-alarm rate. Timestep delay is added where per-step residuals exist."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score
from ..core.types import Episode

def detection_score(ep: Episode) -> float:
    return float(np.linalg.norm(ep.energy()))

def calibrate_threshold(healthy_scores, target_far=0.01) -> float:
    return float(np.quantile(np.asarray(healthy_scores), 1 - target_far))

def event_f1(scores, labels, thresh) -> dict:
    scores = np.asarray(scores); labels = np.asarray(labels).astype(int)
    pred = (scores > thresh).astype(int)
    tp = int((pred & labels).sum()); fp = int((pred & (1 - labels)).sum()); fn = int(((1 - pred) & labels).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}

def auprc(scores, labels) -> float:
    labels = np.asarray(labels).astype(int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return float(average_precision_score(labels, np.asarray(scores)))

def false_alarm_rate(healthy_scores, thresh) -> float:
    return float(np.mean(np.asarray(healthy_scores) > thresh))

def detection_delay(ep: Episode, thresh: float, per_node_thresh: bool = False):
    """Control steps from fault onset to first alarm, if per-step residuals were kept. None if never fires."""
    if ep.y is None or ep.yhat is None:
        return None
    resid = np.linalg.norm(ep.y - ep.yhat, axis=1)
    onset = ep.meta.get("onset_step", 0)
    fired = np.where(resid[onset:] > thresh)[0]
    return int(fired[0]) if len(fired) else None

def evaluate_detection(test_episodes, healthy_val_episodes, target_far=0.01) -> dict:
    hv = [detection_score(e) for e in healthy_val_episodes]
    thr = calibrate_threshold(hv, target_far)
    scores = [detection_score(e) for e in test_episodes]
    labels = [0 if e.is_healthy else 1 for e in test_episodes]
    out = event_f1(scores, labels, thr)
    out["auprc"] = auprc(scores, labels)
    out["far"] = false_alarm_rate([s for s, l in zip(scores, labels) if l == 0], thr)
    delays = [d for d in (detection_delay(e, thr) for e in test_episodes if not e.is_healthy) if d is not None]
    out["detection_delay"] = float(np.mean(delays)) if delays else float("nan")
    out["threshold"] = thr
    return out
