"""Generic evaluation: every localizer through one harness, full metric suite (localization + detection +
conformal + latency). Residual-based methods read the residual field; time-series methods read a raw channel."""
import os, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[1])  # repository root (code)
ROOT = os.environ.get("FSI_ROOT", REPO)  # data/results root, see docs/CONFIGURATION.md
import sys, dataclasses, json, argparse
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, REPO)
from fsi.core.types import Operator
from fsi.core.graph import StructuralGraph
from fsi.data import load_episodes, make_splits
from fsi.baselines import (LargestResidual, DirectSolve, GDN, TranAD, GraphSL, RCD, AERCA,
                           ConstantScore, AntiEnergy, RandomScore)
from fsi.models import FSILocalizer, ConformalSupport, RBCLocalizer
from fsi.eval import run_comparison, save_table
from fsi.eval.stats import characterize
from fsi.metrics import evaluate_detection, detection_score, calibrate_threshold

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tag", default="eval")
ap.add_argument("--cap", type=int, default=6000)
ap.add_argument("--series_channel", type=int, default=2)
ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--heldout_task", action="store_true")
ap.add_argument("--seeds", type=int, default=5, help="repeats per localizer; the table reports mean and sd")
args = ap.parse_args()
torch.set_num_threads(args.threads)

def series_view(eps, c):
    """Single-channel view for the time-series baselines. yhat must be sliced alongside y -- methods that consume
    y - yhat (RCD) otherwise get a (T,n) signal against a (T,n,C) prediction and die on the broadcast."""
    return [dataclasses.replace(e, y=(e.y[..., c] if e.y is not None else None),
                                yhat=(e.yhat[..., c] if getattr(e, "yhat", None) is not None
                                      and np.ndim(e.yhat) == 3 else None)) for e in eps]

def compare(train, test, g, op, out, tag, cands=None):
    import pandas as pd
    # Factories, not instances: a multi-seed run must build each localizer fresh, and the zz_* reference rows
    # pin the floor of every column so a margin can be read as a margin.
    rl = [LargestResidual, lambda: DirectSolve(lam=0.03), GraphSL, RBCLocalizer,
          lambda: FSILocalizer(epochs=80), ConstantScore, AntiEnergy, RandomScore]
    sl = [GDN, TranAD, RCD, AERCA]
    seeds, keep = tuple(range(args.seeds)), {}
    df = pd.concat([run_comparison(rl, train, test, g, op, seeds=seeds, candidates=cands, keep=keep),
                    run_comparison(sl, series_view(train, args.series_channel),
                                   series_view(test, args.series_channel), g, op, seeds=seeds, candidates=cands)],
                   ignore_index=True).sort_values("hard_top1", ascending=False)
    save_table(df, out / f"{tag}", title=f"{tag} localization comparison")
    print(f"--- {tag} ---\n{df.to_string(index=False)}", flush=True)
    return df, keep

def main():
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    eps = load_episodes(data / "all")
    rng = np.random.default_rng(0)
    if len(eps) > args.cap:
        eps = [eps[i] for i in rng.choice(len(eps), args.cap, replace=False)]
    g = StructuralGraph.load(data / "graph.json")
    op = Operator.load(data / "operator_G.npz")
    print("dataset:", {k: characterize(eps, g, op)[k] for k in ("n_episodes", "cause_victim_gap", "coherence")},
          "| candidates:", len(g.fail_nodes), flush=True)
    sp = make_splits(eps, seed=0)
    # Score over the nodes the fault catalogue can actually inject at, derived from the TRAIN/CAL labels only.
    # graph.fail_nodes is what could fail in principle (17 on ANYmal); the generator only ever injects at the 12
    # leg joints, so scoring over 17 both misreports chance (1/17 = 0.059 rather than 1/12 = 0.083) and lets a
    # method waste its argmax on a node that is never the answer.
    realizable = sorted({s for e in sp["train"] + sp["cal"] for s in e.support()})
    cands = [v for v in g.fail_nodes if v in set(realizable)]
    cands = cands if 0 < len(cands) < len(g.fail_nodes) else None
    print(f"candidates: {len(cands) if cands else len(g.fail_nodes)} of {len(g.fail_nodes)} failable "
          f"(chance {1/len(cands or g.fail_nodes):.3f})", flush=True)
    extra = {"candidates": cands, "chance": 1 / len(cands or g.fail_nodes)}
    cal_h = [e for e in sp["cal"] if e.is_healthy]
    test = sp["test"]
    if cal_h:
        det = evaluate_detection(test, cal_h, target_far=0.05)
        extra["detection"] = det
        print("detection:", {k: round(v, 4) for k, v in det.items() if isinstance(v, float)}, flush=True)
        # Algorithm 4: the localizer runs ONLY when detection fires. Scoring localization on undetected
        # episodes measures noise, so the primary protocol is detection-gated; ungated is reported too.
        thr = calibrate_threshold([detection_score(e) for e in cal_h], target_far=0.05)
        faults_te = [e for e in test if not e.is_healthy]
        gated = [e for e in faults_te if detection_score(e) > thr]
        rate = len(gated) / max(1, len(faults_te))
        extra["detection_gate"] = {"threshold": thr, "n_fault_test": len(faults_te),
                                   "n_detected": len(gated), "detected_rate": rate}
        print(f"detection gate: {len(gated)}/{len(faults_te)} fault episodes detected ({rate:.2f})", flush=True)
        if len(gated) >= 50:
            compare(sp["train"], gated, g, op, out, f"{args.tag}_DETECTED", cands)
    df, keep = compare(sp["train"], test, g, op, out, f"{args.tag}_in_distribution", cands)
    try:
        conf = ConformalSupport(keep["fsi"], mode="crc", alpha=0.1,
                                group_by=lambda e: e.faults[0].fault_type if e.faults else "none").calibrate(
                                [e for e in sp["cal"] if not e.is_healthy], g, op)
        cov = conf.evaluate([e for e in sp["test"] if not e.is_healthy], g, op)
        extra["conformal_fsi"] = cov
        print("FSI conformal:", {k: round(v, 4) for k, v in cov.items()}, flush=True)
    except Exception as e:
        print("conformal failed:", type(e).__name__, str(e)[:150], flush=True)
    if args.heldout_task:
        held = {}
        for t in sorted({e.meta.get("task", "?") for e in eps}):
            tr = [e for e in eps if e.meta.get("task") != t]
            te = [e for e in eps if e.meta.get("task") == t]
            if len(te) < 50 or len(tr) < 200:
                continue
            d_t, _ = compare(tr, te, g, op, out, f"{args.tag}_heldout_{t}", cands)
            held[t] = {r["localizer"]: {k: r.get(k) for k in ("top1", "hard_top1", "w1g", "hit@3")} for _, r in d_t.iterrows()}
        extra["heldout_task"] = held
    (out / f"{args.tag}_detection_conformal.json").write_text(json.dumps(extra, indent=2, default=float))
    print("saved to", out, flush=True)

main()
