"""Does operator identifiability predict where the localizers make their mistakes?

For each embodiment we (1) take the pairwise column-coherence matrix of the propagation operator G restricted to
the candidate source nodes, (2) measure the empirical confusion matrix of each localizer over the test episodes
across seeds, and (3) ask whether coherence(i,j) rank-predicts confusion(i,j) -- with graph hop distance as the
control predictor, because on a serial chain "coherent" and "adjacent" are nearly the same statement and the
identifiability claim is only interesting if coherence survives conditioning on hops.

Protocol mirrors scripts/eval_embodiment.py exactly (cap 6000 @ rng 0, make_splits seed 0, candidates =
realizable sources from train+cal, series channel 2 for the time-series baselines) so the hard_top1 we recompute
here can be checked against the headline tables.

Writes only to results/paper/identifiability/.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])  # repository root (code)
ROOT = os.environ.get("FSI_ROOT", REPO)  # data/results root, see docs/CONFIGURATION.md
import sys, time, json, dataclasses, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, REPO)
from fsi.core.types import Operator
from fsi.core.graph import StructuralGraph
from fsi.data import load_episodes, make_splits
from fsi.baselines import (LargestResidual, DirectSolve, GDN, TranAD, GraphSL, RCD, AERCA,
                           ConstantScore, AntiEnergy, RandomScore)
from fsi.models import FSILocalizer, RBCLocalizer
from fsi.metrics.localization import is_hard

OUT = Path(f"{ROOT}/results/paper/identifiability")

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--cap", type=int, default=6000)
ap.add_argument("--perms", type=int, default=2000)
ap.add_argument("--series_channel", type=int, default=2)
ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--quick", action="store_true", help="tiny smoke run")
args = ap.parse_args()
torch.set_num_threads(args.threads)

# ----------------------------------------------------------------------------- stats helpers
from scipy.stats import rankdata

def _rank(x):
    """Average-rank transform (ties shared), so Pearson-on-ranks == Spearman."""
    return rankdata(np.asarray(x, float))

def _pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    da, db = np.sqrt((a * a).sum()), np.sqrt((b * b).sum())
    if da < 1e-12 or db < 1e-12:
        return float("nan")
    return float((a * b).sum() / (da * db))

def spearman(x, y):
    return _pearson(_rank(x), _rank(y))

def partial_spearman(x, y, z):
    """Spearman of x,y controlling for z: linearly residualise the rank vectors on the rank of z."""
    return _partial_fn(x, z)(y)

def _partial_fn(x, z):
    """Returns f(y) = partial Spearman(x, y | z) with everything about x and z precomputed (the permutation
    test reuses it thousands of times with only y changing)."""
    rx, rz = _rank(x), _rank(z)
    Z = np.stack([np.ones_like(rz), rz], 1)
    P = Z @ np.linalg.pinv(Z)
    res = lambda v: v - P @ v
    rxr = res(rx)
    return lambda y: _pearson(rxr, res(_rank(y)))

def _spearman_fn(x):
    rx = _rank(x)
    return lambda y: _pearson(rx, _rank(y))

def mantel_p(stat_fn, C, B, rng):
    """Permutation test for a matrix-vs-matrix rank statistic. The null permutes NODE LABELS of the confusion
    matrix (rows and columns together), which is the only shuffle that respects the fact that off-diagonal pairs
    sharing a node are not independent observations. Two-sided on |rho|."""
    obs = stat_fn(C)
    if not np.isfinite(obs):
        return obs, float("nan"), 0
    K = C.shape[0]
    hits = 0
    for _ in range(B):
        p = rng.permutation(K)
        v = stat_fn(C[np.ix_(p, p)])
        if np.isfinite(v) and abs(v) >= abs(obs) - 1e-12:
            hits += 1
    return obs, (1.0 + hits) / (1.0 + B), B

# ----------------------------------------------------------------------------- data / protocol
def series_view(eps, c):
    return [dataclasses.replace(e, y=(e.y[..., c] if e.y is not None else None),
                                yhat=(e.yhat[..., c] if getattr(e, "yhat", None) is not None
                                      and np.ndim(e.yhat) == 3 else None)) for e in eps]

def coherence_of(G):
    Gn = G / (np.linalg.norm(G, axis=0, keepdims=True) + 1e-12)
    M = np.abs(Gn.T @ Gn)
    np.fill_diagonal(M, 0.0)
    return M

RESID = [("fsi", lambda: FSILocalizer(epochs=80)), ("fsi_rbc", RBCLocalizer), ("graphsl", GraphSL),
         ("direct_l1", lambda: DirectSolve(lam=0.03)), ("largest_residual", LargestResidual),
         ("zz_constant", ConstantScore), ("zz_anti_energy", AntiEnergy), ("zz_random", RandomScore)]
SERIES = [("gdn", GDN), ("tranad", TranAD), ("rcd", RCD), ("aerca", AERCA)]

def predict_nodes(spec, train, test, g, op, cand, seeds):
    """Returns (n_seeds, n_test) predicted node ids, replicating the harness's stable-argsort ranking rule."""
    out, hard1, top1 = [], [], []
    hard_mask = np.array([is_hard(e, g, list(cand)) for e in test])
    truth = np.array([e.support()[0] for e in test])
    for sd in seeds:
        loc = spec() if callable(spec) else spec
        if hasattr(loc, "seed"):
            loc.seed = sd
        torch.manual_seed(sd); np.random.seed(sd)
        loc.fit(train, g, op)
        preds = loc.predict_many(test, g, op)
        bad = sum(1 for p in preds if not np.isfinite(p.scores).all())
        if bad:
            raise ValueError(f"non-finite scores on {bad}/{len(preds)} episodes")
        pn = np.array([cand[np.argsort(-p.scores[cand], kind="stable")[0]] for p in preds])
        out.append(pn)
        top1.append(float((pn == truth).mean()))
        hard1.append(float((pn[hard_mask] == truth[hard_mask]).mean()))
    return np.stack(out), truth, hard_mask, float(np.mean(top1)), float(np.mean(hard1))

def confusion_rates(pred, truth, cand, mask=None):
    """Row-normalised confusion: rate[i, j] = P(predict cand[j] | true source is cand[i]). Averaged over seeds by
    pooling counts, and returned with the per-row episode counts so a caller can see the n behind each row."""
    idx = {v: k for k, v in enumerate(cand)}
    K = len(cand)
    counts = np.zeros((K, K), float)
    sel = np.ones(len(truth), bool) if mask is None else mask
    for s in range(pred.shape[0]):
        for t, p in zip(truth[sel], pred[s][sel]):
            counts[idx[t], idx[p]] += 1
    row_n = counts.sum(1)
    rate = counts / np.maximum(row_n[:, None], 1e-9)
    return rate, counts, row_n

# ----------------------------------------------------------------------------- main
def run(tag, data_dir):
    t0 = time.time()
    eps = load_episodes(Path(data_dir) / "all")
    rng0 = np.random.default_rng(0)
    if len(eps) > args.cap:
        eps = [eps[i] for i in rng0.choice(len(eps), args.cap, replace=False)]
    g = StructuralGraph.load(Path(data_dir) / "graph.json")
    op = Operator.load(Path(data_dir) / "operator_G.npz")
    sp = make_splits(eps, seed=0)
    realizable = sorted({s for e in sp["train"] + sp["cal"] for s in e.support()})
    cands = [v for v in g.fail_nodes if v in set(realizable)]
    cand = np.array(cands if 0 < len(cands) < len(g.fail_nodes) else g.fail_nodes)
    names = g.names()
    train, test = sp["train"], [e for e in sp["test"] if not e.is_healthy and len(e.faults) == 1]
    if args.quick:
        train, test = train[:300], test[:120]
    print(f"[{tag}] n_eps={len(eps)} train={len(train)} test_fault={len(test)} "
          f"candidates={len(cand)} chance={1/len(cand):.3f}", flush=True)

    # --- predictors -----------------------------------------------------------
    Mfull = op.coherence_matrix()
    M = Mfull[np.ix_(cand, cand)]
    hop = g.hop_distance()[np.ix_(cand, cand)].astype(float)
    C = op.G_multi.shape[0] if op.G_multi is not None else 0
    Mc = np.stack([coherence_of(op.G_multi[c])[np.ix_(cand, cand)] for c in range(C)]) if C else None
    np.save(OUT / f"coherence_{tag}.npy", M)
    np.save(OUT / f"hop_{tag}.npy", hop)
    if Mc is not None:
        np.save(OUT / f"coherence_multi_{tag}.npy", Mc)
    np.save(OUT / f"candidates_{tag}.npy", cand)
    (OUT / f"names_{tag}.json").write_text(json.dumps([names[v] for v in cand]))

    preds_ = {"coherence_G": M, "hop_distance": hop}
    if Mc is not None:
        preds_["coherence_multi_mean"] = Mc.mean(0)
        preds_["coherence_multi_max"] = Mc.max(0)

    # --- run every localizer --------------------------------------------------
    seeds = tuple(range(args.seeds))
    got = {}
    for name, spec in RESID:
        try:
            t = time.time()
            got[name] = predict_nodes(spec, train, test, g, op, cand, seeds)
            print(f"  {name:17s} top1={got[name][3]:.3f} hard_top1={got[name][4]:.3f} "
                  f"({time.time()-t:.0f}s)", flush=True)
        except Exception as e:
            print(f"  !! {name} FAILED {type(e).__name__}: {e}"[:200], flush=True)
    tr_s, te_s = series_view(train, args.series_channel), series_view(test, args.series_channel)
    for name, spec in SERIES:
        try:
            t = time.time()
            got[name] = predict_nodes(spec, tr_s, te_s, g, op, cand, seeds)
            print(f"  {name:17s} top1={got[name][3]:.3f} hard_top1={got[name][4]:.3f} "
                  f"({time.time()-t:.0f}s)", flush=True)
        except Exception as e:
            print(f"  !! {name} FAILED {type(e).__name__}: {e}"[:200], flush=True)

    pair_rows, corr_rows, acc_rows = [], [], []
    off = ~np.eye(len(cand), dtype=bool)
    rng = np.random.default_rng(12345)
    for name, (pred, truth, hard_mask, t1, h1) in got.items():
        acc_rows.append({"embodiment": tag, "localizer": name, "top1": t1, "hard_top1": h1,
                         "n_fault_test": len(test), "n_hard": int(hard_mask.sum()), "n_seeds": args.seeds})
        for subset, mask in (("all_fault", None), ("hard_only", hard_mask)):
            rate, counts, row_n = confusion_rates(pred, truth, cand, mask)
            if subset == "all_fault":
                np.save(OUT / f"confusion_{tag}_{name}.npy", rate)
            err = counts.copy(); np.fill_diagonal(err, 0.0)
            err_rate = err / np.maximum(err.sum(1, keepdims=True), 1e-9)   # P(pred=j | true=i, mistake)
            for i in range(len(cand)):
                for j in range(len(cand)):
                    if i == j:
                        continue
                    pair_rows.append({"embodiment": tag, "localizer": name, "episode_subset": subset,
                                      "node_i": int(cand[i]), "node_j": int(cand[j]),
                                      "name_i": names[cand[i]], "name_j": names[cand[j]],
                                      "coherence": float(M[i, j]), "hop_distance": float(hop[i, j]),
                                      "coherence_multi_mean": float(preds_["coherence_multi_mean"][i, j])
                                      if Mc is not None else float("nan"),
                                      "confusion_rate": float(rate[i, j]),
                                      "misroute_rate": float(err_rate[i, j]),
                                      "n_episodes_true_i": float(row_n[i] / args.seeds),
                                      "n_seeds": args.seeds})
            for pname, P in preds_.items():
                f = _spearman_fn(P[off])
                fn = lambda Cm, f=f: f(Cm[off])
                rho, p, B = mantel_p(fn, rate, args.perms, rng)
                corr_rows.append({"embodiment": tag, "localizer": name, "predictor": pname,
                                  "episode_subset": subset, "target": "confusion_rate",
                                  "spearman_rho": rho, "p_perm": p, "n_perm": B, "n_pairs": int(off.sum())})
            # partials: coherence controlling hops, and hops controlling coherence
            for pname, a, b in (("coherence_G|hop", M, hop), ("hop|coherence_G", hop, M)):
                f = _partial_fn(a[off], b[off])
                fn = lambda Cm, f=f: f(Cm[off])
                rho, p, B = mantel_p(fn, rate, args.perms, rng)
                corr_rows.append({"embodiment": tag, "localizer": name, "predictor": pname,
                                  "episode_subset": subset, "target": "confusion_rate",
                                  "spearman_rho": rho, "p_perm": p, "n_perm": B, "n_pairs": int(off.sum())})
            # same battery against the mistake-conditional target
            for pname, P in (("coherence_G", M), ("hop_distance", hop)):
                f = _spearman_fn(P[off])
                fn = lambda Cm, f=f: f(Cm[off])
                rho, p, B = mantel_p(fn, err_rate, args.perms, rng)
                corr_rows.append({"embodiment": tag, "localizer": name, "predictor": pname,
                                  "episode_subset": subset, "target": "misroute_rate",
                                  "spearman_rho": rho, "p_perm": p, "n_perm": B, "n_pairs": int(off.sum())})
    # per-channel coherence: which signal channel's geometry best explains FSI's confusions
    chan_rows = []
    if Mc is not None:
        for name in got:
            rate, _, _ = confusion_rates(got[name][0], got[name][1], cand)
            for c in range(C):
                f = _spearman_fn(Mc[c][off])
                rho, p, B = mantel_p(lambda Cm, f=f: f(Cm[off]), rate, 500, rng)
                chan_rows.append({"embodiment": tag, "localizer": name, "channel": c,
                                  "spearman_rho": rho, "p_perm": p, "n_perm": B, "n_pairs": int(off.sum())})
    print(f"[{tag}] done in {time.time()-t0:.0f}s", flush=True)
    return pair_rows, corr_rows, acc_rows, chan_rows

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    P, R, A, CH = [], [], [], []
    for tag, d in (("franka_v6", f"{ROOT}/data/franka_v6"),
                   ("anymal_v4", f"{ROOT}/data/anymal_v4")):
        p, r, a, ch = run(tag, d)
        P += p; R += r; A += a; CH += ch
        pd.DataFrame(P).to_csv(OUT / "pairs.csv", index=False)
        pd.DataFrame(R).to_csv(OUT / "correlation.csv", index=False)
        pd.DataFrame(A).to_csv(OUT / "accuracy_check.csv", index=False)
        pd.DataFrame(CH).to_csv(OUT / "per_channel_correlation.csv", index=False)
    # top-5 most coherent candidate pairs per embodiment (unordered), with empirical confusion
    dfp = pd.DataFrame(P)
    tops = []
    for tag in dfp.embodiment.unique():
        d = dfp[(dfp.embodiment == tag) & (dfp.episode_subset == "all_fault")]
        key = d.apply(lambda r: tuple(sorted((r.node_i, r.node_j))), axis=1)
        d = d.assign(pair=key)
        base = d[d.localizer == "fsi"].groupby("pair").agg(
            coherence=("coherence", "first"), hop_distance=("hop_distance", "first"),
            confusion_rate_fsi=("confusion_rate", "mean")).reset_index()
        mean_all = d[~d.localizer.str.startswith("zz_")].groupby("pair")["confusion_rate"].mean()
        base["confusion_rate_mean_informative_localizers"] = base["pair"].map(mean_all)
        nm = {v: n for v, n in zip(range(len(dfp)), [])}
        names = {}
        for _, r in d.iterrows():
            names[r.node_i] = r.name_i; names[r.node_j] = r.name_j
        base["node_i"] = base["pair"].map(lambda p: p[0]); base["node_j"] = base["pair"].map(lambda p: p[1])
        base["name_i"] = base["node_i"].map(names); base["name_j"] = base["node_j"].map(names)
        base["embodiment"] = tag
        base = base.sort_values("coherence", ascending=False).head(5).drop(columns=["pair"])
        tops.append(base)
    pd.concat(tops).to_csv(OUT / "top_coherent_pairs.csv", index=False)
    print("saved to", OUT, flush=True)

main()
