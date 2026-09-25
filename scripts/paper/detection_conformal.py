"""Part 2: conformal support sets from the fitted FSI localizer.

Coverage and set size as a function of alpha for both modes ('crc' and 'aps'), both embodiments, marginal and
Mondrian (per-fault-type) calibration, on the released datasets. The composition used here IS
fsi.models.conformal.ConformalSupport -- same aps_calibrate/crc_calibrate/aps_set/crc_set functions, called
directly so that the whole alpha grid reuses ONE set of localizer predictions instead of refitting the
localizer 15 times per configuration. The equivalence is asserted at alpha = 0.1 against ConformalSupport
itself before anything else is written.

Two candidate scopes are reported:
  all_failable  graph.fail_nodes -- what ConformalSupport uses internally, and therefore the denominator behind
                the published "3.62 / 7" (Franka) and "7.24" (ANYmal) numbers. NOTE: the ANYmal figure has been
                quoted as "7.24 / 12"; it is 7.24 out of SEVENTEEN failable nodes, because ConformalSupport
                reads graph.fail_nodes (17) and not the 12 realizable joints the localization tables score over.
  realizable    the 12 (ANYmal) / 7 (Franka) nodes the fault catalogue actually injects at, derived from
                train+cal labels exactly as scripts/eval_embodiment.py derives them. This is the scope every
                localization number in the paper uses, so it is the one the conformal sets should be read in.

Multi-seed over the FSI localizer seed. Seed 4 is included because scripts/eval_embodiment.py keeps the LAST
fitted localizer (seeds 0..4) for conformal, so seed 4 is the one behind the published numbers.
"""
import os, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])  # repository root (code)
ROOT = os.environ.get("FSI_ROOT", REPO)  # data/results root, see docs/CONFIGURATION.md
import sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, REPO)
from fsi.core.graph import StructuralGraph
from fsi.core.types import Operator
from fsi.data import load_episodes, make_splits
from fsi.models import FSILocalizer, ConformalSupport
from fsi.models.conformal import aps_calibrate, aps_set, crc_calibrate, crc_set

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=f"{ROOT}/results/paper/detection")
ap.add_argument("--cap", type=int, default=6000)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
ap.add_argument("--epochs", type=int, default=80)
args = ap.parse_args()

DATASETS = [("franka", f"{ROOT}/data/franka_v6"), ("anymal", f"{ROOT}/data/anymal_v4")]
ALPHAS = np.round(np.arange(0.02, 0.3001, 0.02), 3)
ALPHA_HIST = 0.1
ftype = lambda e: e.faults[0].fault_type if e.faults else "none"


def calib(mode, preds, eps, fail, alpha):
    return (aps_calibrate if mode == "aps" else crc_calibrate)(preds, eps, fail, alpha)


def mkset(mode, pred, fail, thr):
    return (aps_set if mode == "aps" else crc_set)(pred, fail, thr)


def main():
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(16)
    alpha_rows, size_rows, group_rows = [], [], []
    for emb, path in DATASETS:
        eps = load_episodes(Path(path) / "all")
        rng = np.random.default_rng(0)
        if len(eps) > args.cap:
            eps = [eps[i] for i in rng.choice(len(eps), args.cap, replace=False)]
        g = StructuralGraph.load(Path(path) / "graph.json")
        op = Operator.load(Path(path) / "operator_G.npz")
        sp = make_splits(eps, seed=0)
        realizable = sorted({s for e in sp["train"] + sp["cal"] for s in e.support()})
        cands = [v for v in g.fail_nodes if v in set(realizable)]
        scopes = {"all_failable": np.asarray(g.fail_nodes),
                  "realizable": np.asarray(cands if cands else g.fail_nodes)}
        cal_f = [e for e in sp["cal"] if not e.is_healthy]
        te_f = [e for e in sp["test"] if not e.is_healthy]
        print(f"== {emb}: {len(eps)} eps | cal_faults {len(cal_f)} test_faults {len(te_f)} | "
              f"scopes {[(k, len(v)) for k, v in scopes.items()]} | {dev}", flush=True)

        for seed in args.seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            loc = FSILocalizer(epochs=args.epochs, seed=seed, device=dev).fit(sp["train"], g, op)
            p_cal = loc.predict_many(cal_f, g, op)
            p_te = loc.predict_many(te_f, g, op)
            assert all(np.isfinite(p.scores).all() for p in p_cal + p_te), "non-finite localizer scores"

            # Equivalence check against ConformalSupport itself (all_failable scope, Mondrian, crc, alpha .1).
            ref = ConformalSupport(loc, mode="crc", alpha=0.1, group_by=ftype).calibrate(cal_f, g, op)
            ref_ev = ref.evaluate(te_f, g, op)
            print(f"  seed {seed}: ConformalSupport crc a=.1 mondrian -> "
                  f"size {ref_ev['avg_set_size']:.4f} coverage {ref_ev['coverage']:.4f}", flush=True)

            for scope, fail in scopes.items():
                keep = [k for k, e in enumerate(te_f) if e.support()[0] in set(fail.tolist())]
                keep_c = [k for k, e in enumerate(cal_f) if e.support()[0] in set(fail.tolist())]
                for mode in ("crc", "aps"):
                    for grouping in ("marginal", "mondrian_fault_type"):
                        key = (lambda e: "_all") if grouping == "marginal" else ftype
                        gc = {}
                        for k in keep_c:
                            gc.setdefault(key(cal_f[k]), []).append(k)
                        for alpha in ALPHAS:
                            thr = {gk: calib(mode, [p_cal[i] for i in ii], [cal_f[i] for i in ii],
                                             fail, float(alpha)) for gk, ii in gc.items()}
                            recs = []
                            for k in keep:
                                e = te_f[k]
                                t = thr.get(key(e), next(iter(thr.values())))
                                S = set(mkset(mode, p_te[k], fail, t))
                                recs.append((k, ftype(e), int(e.support()[0]),
                                             float(e.faults[0].magnitude), len(S),
                                             int(e.support()[0] in S)))
                            R = pd.DataFrame(recs, columns=["test_episode_index", "fault_type", "source_node",
                                                            "severity", "set_size", "covered"])
                            base = {"embodiment": emb, "dataset": Path(path).name,
                                    "candidate_scope": scope, "n_candidates": int(len(fail)),
                                    "mode": mode, "grouping": grouping, "fsi_seed": seed,
                                    "alpha": float(alpha)}
                            alpha_rows.append({**base, "n_cal": len(keep_c), "n_test": len(R),
                                               "target_coverage": 1 - float(alpha),
                                               "coverage": float(R.covered.mean()),
                                               "coverage_deviation": float(R.covered.mean()) - (1 - float(alpha)),
                                               "mean_set_size": float(R.set_size.mean()),
                                               "sd_set_size": float(R.set_size.std(ddof=1)),
                                               "median_set_size": float(R.set_size.median()),
                                               "p90_set_size": float(R.set_size.quantile(0.9)),
                                               "frac_singleton": float((R.set_size == 1).mean()),
                                               "frac_full_set": float((R.set_size == len(fail)).mean())})
                            for ft, sub in R.groupby("fault_type"):
                                group_rows.append({**base, "fault_type": ft, "n_test": len(sub),
                                                   "target_coverage": 1 - float(alpha),
                                                   "coverage": float(sub.covered.mean()),
                                                   "coverage_deviation": float(sub.covered.mean()) - (1 - float(alpha)),
                                                   "mean_set_size": float(sub.set_size.mean())})
                            if abs(alpha - ALPHA_HIST) < 1e-9:
                                for _, r in R.iterrows():
                                    size_rows.append({**base, **r.to_dict()})
            print(f"  seed {seed} done", flush=True)

    pd.DataFrame(alpha_rows).to_csv(out / "conformal_alpha.csv", index=False)
    pd.DataFrame(size_rows).to_csv(out / "conformal_sizes.csv", index=False)
    pd.DataFrame(group_rows).to_csv(out / "conformal_group.csv", index=False)
    print("wrote conformal_alpha.csv", len(alpha_rows), "| conformal_sizes.csv", len(size_rows),
          "| conformal_group.csv", len(group_rows), flush=True)


main()
