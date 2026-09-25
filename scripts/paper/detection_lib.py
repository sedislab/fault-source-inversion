"""Shared analysis for the detection + conformal paper figures.

Everything here is EPISODE-LEVEL. One score per episode, one binary label per episode, no point adjustment
anywhere (a random per-timestep score reaches SOTA F1 under point adjust, so the whole comparison would be
meaningless). Verified by inspection: `grep -rni "point.adjust" fsi/ scripts/` returns only the three docstrings
that state it is NOT used; no metric in the repo expands an alarm to cover a ground-truth event window.

Three episode-level detection statistics are compared. `zmax` and `cusum` read the SAME per-step residual
stream, so the only difference between them is the accumulation rule:

  field_energy  ||r[:, 0]||_2 -- the statistic the paper currently reports
                (fsi.metrics.detection.detection_score), the L2 norm over nodes of the reduced residual field's
                energy channel. It is a whole-episode reduction and has no time resolution.
  zmax          max_t e(t) with e(t) = RMS over nodes of the standardized per-step residual magnitude.
                Instantaneous control: same stream as CUSUM, no accumulation.
  cusum         max_{i,t} g_i(t), g_i(t) = max(0, g_i(t-1) + |z_i(t)| - nu). Per-node CUSUM.

|z_i(t)| is the RMS across node i's signal channels of the per-channel standardized 1-step residual
z_ic(t) = (y_ic(t) - yhat_ic(t) - mu_ic) / s_ic, with mu_ic, s_ic estimated on HEALTHY TRAIN episodes only.
Under healthy data E|z_i| ~= 1, which is why nu is swept over [0.5, 4].
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, roc_curve

HIST = 8  # ForwardModel.hist: the first HIST steps carry no prediction (yhat == y there, residual identically 0)

FAR_GRID = np.round(np.arange(0.01, 0.2001, 0.01), 3)
NU_GRID = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)


# ---------------------------------------------------------------- per-step residual stream

def healthy_residual_stats(healthy_train, hist=HIST):
    R = np.concatenate([(e.y - e.yhat)[hist:] for e in healthy_train], 0)   # (N*(T-hist), n, C)
    return R.mean(0), R.std(0) + 1e-8                                       # (n, C)


def stream(episodes, mu, sd, hist=HIST):
    """|z_i(t)| for a list of episodes -> (E, T-hist, n) float32."""
    out = np.empty((len(episodes), episodes[0].y.shape[0] - hist, episodes[0].n), np.float32)
    for k, e in enumerate(episodes):
        z = ((e.y - e.yhat)[hist:] - mu) / sd
        out[k] = np.sqrt((z ** 2).mean(-1))
    return out


def cusum_batch(Z, nu):
    """g_i(t) = max(0, g_i(t-1) + |z_i(t)| - nu), vectorised over episodes. Z:(E,T,n) -> g:(E,T,n)."""
    g = np.empty_like(Z)
    prev = np.zeros(Z.shape[::2], Z.dtype)          # (E, n)
    for t in range(Z.shape[1]):
        prev = np.maximum(0.0, prev + Z[:, t] - nu)
        g[:, t] = prev
    return g


def first_alarm(sig, h, hist=HIST):
    """sig:(E,T) episode-level running statistic. Returns absolute first-alarm step per episode (or -1)."""
    fired = sig > h
    any_f = fired.any(1)
    k = np.where(any_f, fired.argmax(1) + hist, -1)
    return k


# ---------------------------------------------------------------- episode-level scoring

def auc_pair(scores, labels):
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    if y.sum() in (0, len(y)):
        return float("nan"), float("nan")
    return float(average_precision_score(y, s)), float(roc_auc_score(y, s))


def curve_rows(scores, labels, npts=200):
    """Sampled PR and ROC points. Returns list of dicts (curve, x_*, y_*, threshold)."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    rows = []
    prec, rec, thr = precision_recall_curve(y, s)
    thr = np.append(thr, np.nan)          # precision_recall_curve returns len(thr) == len(prec) - 1
    idx = np.unique(np.linspace(0, len(rec) - 1, min(npts, len(rec))).astype(int))
    for j in idx:
        rows.append({"curve": "pr", "x_recall": float(rec[j]), "y_precision": float(prec[j]),
                     "x_fpr": float("nan"), "y_tpr": float("nan"), "threshold": float(thr[j])})
    fpr, tpr, thr2 = roc_curve(y, s)
    idx = np.unique(np.linspace(0, len(fpr) - 1, min(npts, len(fpr))).astype(int))
    for j in idx:
        rows.append({"curve": "roc", "x_recall": float("nan"), "y_precision": float("nan"),
                     "x_fpr": float(fpr[j]), "y_tpr": float(tpr[j]), "threshold": float(thr2[j])})
    return rows


def calibrate(healthy_cal_scores, target_far):
    """Threshold at the (1 - target_far) quantile of the HEALTHY CALIBRATION scores. Identical rule to
    fsi.metrics.detection.calibrate_threshold, so the operating point is comparable to the published one."""
    return float(np.quantile(np.asarray(healthy_cal_scores, float), 1 - target_far))


def operating_point(scores, labels, thr):
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    pred = (s > thr).astype(int)
    tp = int((pred & y).sum()); fp = int((pred & (1 - y)).sum()); fn = int(((1 - pred) & y).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return {"precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0,
            "realized_far": float(np.mean(pred[y == 0])) if (y == 0).any() else float("nan"),
            "tp": tp, "fp": fp, "fn": fn}


# ---------------------------------------------------------------- the full Part-1 analysis

def analyze_detection(splits, meta, nu_star=2.0, npts=200, verbose=True):
    """Run every detection statistic for one (embodiment, forward-model seed); return tidy row lists.

    `splits` is the output of fsi.data.make_splits. Thresholds are calibrated ONLY on healthy episodes from the
    `cal` split; every reported rate is measured on `test`. `meta` is stamped onto every emitted row.
    """
    from fsi.metrics.detection import detection_score

    cal_h = [e for e in splits["cal"] if e.is_healthy]
    test = list(splits["test"])
    tr_h = [e for e in splits["train"] if e.is_healthy]
    lab = np.array([0 if e.is_healthy else 1 for e in test], int)
    fault_idx = np.where(lab == 1)[0]
    onset = np.array([int(e.meta.get("onset_step", 0)) for e in test], int)

    # Reference for the onset estimate: predicting a CONSTANT onset (the training-set mean) with no data at all.
    # Any onset estimator that does not beat this is not estimating anything.
    tr_on = np.array([int(e.meta.get("onset_step", 0)) for e in splits["train"] if not e.is_healthy], float)
    const_onset = float(tr_on.mean()) if len(tr_on) else float("nan")

    mu, sd = healthy_residual_stats(tr_h)
    Zc, Zt = stream(cal_h, mu, sd), stream(test, mu, sd)
    if verbose:
        print(f"  streams: cal_h {Zc.shape} test {Zt.shape} | E|z| healthy-train "
              f"{float(Zc.mean()):.3f}", flush=True)

    # --- episode-level scores + per-step running statistics ---------------------------------
    ec, et = np.sqrt((Zc ** 2).mean(2)), np.sqrt((Zt ** 2).mean(2))          # (E, T) zmax stream
    scores = {("field_energy", np.nan): ([detection_score(e) for e in cal_h],
                                         [detection_score(e) for e in test]),
              ("zmax", np.nan): (ec.max(1).tolist(), et.max(1).tolist())}
    run_t = {("zmax", np.nan): et}                                            # running stat on TEST only
    gnode_t = {}
    for nu in NU_GRID:
        gc, gt = cusum_batch(Zc, nu), cusum_batch(Zt, nu)
        scores[("cusum", nu)] = (gc.max((1, 2)).tolist(), gt.max((1, 2)).tolist())
        run_t[("cusum", nu)] = gt.max(2)
        gnode_t[nu] = gt
        del gc

    n_h, n_f = int((lab == 0).sum()), int((lab == 1).sum())
    pr_rows, far_rows, cus_rows, onset_rows = [], [], [], []
    aucs = {}

    for key, (cal_s, te_s) in scores.items():
        stat, nu = key
        ap, au = auc_pair(te_s, lab)
        aucs[key] = (ap, au)
        for r in curve_rows(te_s, lab, npts):
            pr_rows.append({**meta, "statistic": stat, "nu": nu, "auprc": ap, "auroc": au,
                            "n_test_healthy": n_h, "n_test_fault": n_f, **r})
        for far in FAR_GRID:
            h = calibrate(cal_s, far)
            op = operating_point(te_s, lab, h)
            far_rows.append({**meta, "statistic": stat, "nu": float(nu), "target_far": float(far),
                             "threshold": h, "auprc": ap, "auroc": au, "n_cal_healthy": len(cal_s),
                             "n_test_healthy": n_h, "n_test_fault": n_f, **op})

    # --- matched-FAR comparison with detection delay ----------------------------------------
    for far in FAR_GRID:
        for key, (cal_s, te_s) in scores.items():
            stat, nu = key
            h = calibrate(cal_s, far)
            op = operating_point(te_s, lab, h)
            ap, au = aucs[key]
            # The cal-healthy split is small (146-243 episodes), so the REALIZED far at a given target wobbles by
            # more than the recall differences under test. `recall_at_matched_test_far` re-thresholds each
            # statistic on the TEST healthy scores so every statistic sits at exactly the same realized FAR --
            # an oracle threshold, reported alongside (never instead of) the honestly calibrated one, because it
            # is the only way to read the comparison as apples-to-apples. It is the ROC read at FPR = far.
            h_or = calibrate(np.asarray(te_s)[lab == 0], far)
            op_or = operating_point(te_s, lab, h_or)
            row = {**meta, "statistic": stat, "nu": float(nu), "target_far": float(far), "threshold": h,
                   "auprc": ap, "auroc": au, "n_cal_healthy": len(cal_s),
                   "n_test_healthy": n_h, "n_test_fault": n_f, **op,
                   "threshold_matched": h_or, "recall_at_matched_test_far": op_or["recall"],
                   "realized_far_matched": op_or["realized_far"]}
            if stat == "field_energy":
                # The deployed field statistic is a whole-episode reduction: one number per episode, no time
                # index, so it HAS no detection delay. `zmax` is its streaming analogue and carries the delay.
                blank = {"mean_delay_steps": np.nan, "median_delay_steps": np.nan, "n_delay": 0,
                         "frac_alarm_before_onset": np.nan, "mean_abs_onset_err_steps": np.nan,
                         "median_abs_onset_err_steps": np.nan, "n_onset": 0,
                         "mean_abs_onset_err_const_steps": np.nan}
                row.update(blank)
                row.update({f"{k}_matched": v for k, v in blank.items()})
            else:
                sig = run_t[key][fault_idx]                   # (F, T) running statistic on fault episodes
                o = onset[fault_idx]
                gn = gnode_t[nu][fault_idx] if stat == "cusum" else None
                row.update(_timing_stats(sig, gn, o, h, const_onset))
                row.update({f"{k}_matched": v for k, v in
                            _timing_stats(sig, gn, o, h_or, const_onset).items()})
            cus_rows.append(row)
        if verbose:
            print(f"  FAR {far:.2f} done", flush=True)

    # --- per-episode onset estimates at nu_star, FAR 0.05 -----------------------------------
    h = calibrate(scores[("cusum", nu_star)][0], 0.05)
    G = gnode_t[nu_star]
    est_onset = np.full(len(test), -1, int)
    for j, i in enumerate(fault_idx):
        e = test[i]
        oh, alarm = _one_onset(G[i], h)
        est_onset[i] = oh
        onset_rows.append({**meta, "nu": float(nu_star), "target_far": 0.05, "threshold": h,
                           "test_episode_index": int(i),
                           "fault_type": e.faults[0].fault_type, "source_node": int(e.faults[0].node),
                           "severity": float(e.faults[0].magnitude), "onset_kind": e.faults[0].onset.value,
                           "true_onset_step": int(onset[i]),
                           "alarm_step": int(alarm) if alarm >= 0 else None,
                           "est_onset_step": int(oh) if oh >= 0 else None,
                           "signed_onset_err_steps": int(oh - onset[i]) if oh >= 0 else None,
                           "abs_onset_err_steps": int(abs(oh - onset[i])) if oh >= 0 else None,
                           "delay_steps": int(alarm - onset[i]) if alarm >= 0 else None,
                           "detected": int(alarm >= 0),
                           "const_onset_pred_step": const_onset,
                           "abs_onset_err_const_steps": abs(const_onset - onset[i])})
    return {"pr_roc": pr_rows, "far_sweep": far_rows, "cusum": cus_rows, "onset": onset_rows,
            "est_onset": est_onset, "onset_threshold": h}


def _timing_stats(sig, gnode, onsets, h, const_onset):
    """Delay and onset-estimate statistics for one running statistic at one threshold.
    delay = (first alarm at or after the true onset) - onset, over the episodes that alarm after onset."""
    k_any = first_alarm(sig, h)
    masked = np.where(np.arange(sig.shape[1])[None, :] + HIST >= onsets[:, None], sig, -np.inf)
    k_post = first_alarm(masked, h)
    d = (k_post - onsets)[k_post >= 0]
    pre = float(np.mean((k_any >= 0) & (k_any < onsets)))
    oe, oe_o = (_onset_errors(gnode, h, onsets) if gnode is not None else (np.array([]), np.array([])))
    return {"mean_delay_steps": float(np.mean(d)) if len(d) else np.nan,
            "median_delay_steps": float(np.median(d)) if len(d) else np.nan,
            "n_delay": int(len(d)), "frac_alarm_before_onset": pre,
            "mean_abs_onset_err_steps": float(np.mean(np.abs(oe))) if len(oe) else np.nan,
            "median_abs_onset_err_steps": float(np.median(np.abs(oe))) if len(oe) else np.nan,
            "n_onset": int(len(oe)),
            "mean_abs_onset_err_const_steps": (float(np.mean(np.abs(oe_o - const_onset)))
                                               if len(oe) else np.nan)}


def _one_onset(g, h):
    """g:(T,n) for one episode. Returns (est_onset_abs, alarm_step_abs), each -1 if no alarm.
    Standard CUSUM change-point estimate: the last time the ALARMING node's accumulator was at zero."""
    gm = g.max(1)
    fired = np.where(gm > h)[0]
    if not len(fired):
        return -1, -1
    k = int(fired[0]); i = int(np.argmax(g[k]))
    zero = np.where(g[:k + 1, i] <= 0)[0]
    return (int(zero[-1] + 1 + HIST) if len(zero) else HIST), k + HIST


def _onset_errors(G, h, onsets):
    """Signed onset-estimate errors and the matching true onsets, over the episodes that alarmed."""
    err, tru = [], []
    for g, o in zip(G, onsets):
        oh, _ = _one_onset(g, h)
        if oh >= 0:
            err.append(oh - o); tru.append(o)
    return np.asarray(err, float), np.asarray(tru, float)
