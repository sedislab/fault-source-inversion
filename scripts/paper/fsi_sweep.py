"""Sweep FSI configurations on a prebuilt dataset through the SAME harness eval_embodiment.py uses, so the
numbers are directly comparable to the headline tables. One row per (dataset, arm, localizer, metric).

The zz_* reference scorers are always included: a configuration that beats another configuration but not the
floor has not found anything, and the floor moves between datasets (it is a property of the candidate set and
the source distribution, not a constant).

    PYTHONPATH=. python scripts/paper/fsi_sweep.py --data data/franka_v7 --out results/paper/fsi_sweep --seeds 3
"""
import os, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])  # repository root (code)
ROOT = os.environ.get("FSI_ROOT", REPO)  # data/results root, see docs/CONFIGURATION.md
import sys, argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, REPO)
from fsi.core.graph import StructuralGraph
from fsi.core.types import Operator
from fsi.data import load_episodes, make_splits
from fsi.models import FSILocalizer, RBCLocalizer
from fsi.baselines import (LargestResidual, DirectSolve, GDN, TranAD, GraphSL, RCD, AERCA,
                           ConstantScore, AntiEnergy, RandomScore)
from fsi.eval import run_comparison

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tag", default=None)
ap.add_argument("--cap", type=int, default=6000)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--series_channel", type=int, default=2)
ap.add_argument("--arms", default="all", help="comma-separated arm names, or 'all'")
ap.add_argument("--baselines", action="store_true", help="also run every non-FSI baseline")
args = ap.parse_args()
torch.set_num_threads(args.threads)

# Each arm is one FSI configuration. `legacy` reproduces the pre-2026-08-06 model: no cross-node readout and no
# listwise loss, i.e. a per-node MLP whose only cross-node input was the unrolled magnitude -- which probe_magw
# measured at std 0.000 on Franka, so effectively none at all.
ARMS = {
    "legacy":       dict(cross=0, rank_w=0.0, sup_w=1.0),
    "rank_only":    dict(cross=0, rank_w=1.0, sup_w=0.3),
    "cross_only":   dict(cross=2, rank_w=0.0, sup_w=1.0),
    "full":         dict(cross=2, rank_w=1.0, sup_w=0.3),
    "full_cross1":  dict(cross=1, rank_w=1.0, sup_w=0.3),
    "full_wide":    dict(cross=2, rank_w=1.0, sup_w=0.3, hidden=128),
    "full_long":    dict(cross=2, rank_w=1.0, sup_w=0.3, epochs=140),
}


def main():
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or data.name
    eps = load_episodes(data / "all")
    rng = np.random.default_rng(0)
    if len(eps) > args.cap:
        eps = [eps[i] for i in rng.choice(len(eps), args.cap, replace=False)]
    g = StructuralGraph.load(data / "graph.json")
    op = Operator.load(data / "operator_G.npz")
    sp = make_splits(eps, seed=0)
    realizable = sorted({s for e in sp["train"] + sp["cal"] for s in e.support()})
    cands = [v for v in g.fail_nodes if v in set(realizable)]
    cands = cands if 0 < len(cands) < len(g.fail_nodes) else None
    K = len(cands or g.fail_nodes)
    print(f"{tag}: {len(eps)} episodes | field {eps[0].r.shape} | {K} candidates (chance {1/K:.3f}) | "
          f"G_multi {None if op.G_multi is None else op.G_multi.shape}", flush=True)

    seeds = tuple(range(args.seeds))
    names = list(ARMS) if args.arms == "all" else args.arms.split(",")
    rows = []
    for name in names:
        cfg = ARMS[name]
        spec = [lambda c=cfg: FSILocalizer(epochs=c.get("epochs", 80), hidden=c.get("hidden", 64),
                                           cross=c["cross"], rank_w=c["rank_w"], sup_w=c["sup_w"])]
        df = run_comparison(spec, sp["train"], sp["test"], g, op, seeds=seeds, candidates=cands)
        df.insert(0, "arm", name)
        rows.append(df)
        print(f"[{name}] {df.iloc[0]['hard_top1']:.3f} hard / {df.iloc[0]['top1']:.3f} top1", flush=True)

    if args.baselines:
        import dataclasses

        def series_view(e_list, c):
            return [dataclasses.replace(e, y=(e.y[..., c] if e.y is not None else None),
                                        yhat=(e.yhat[..., c] if getattr(e, "yhat", None) is not None
                                              and np.ndim(e.yhat) == 3 else None)) for e in e_list]

        rl = [LargestResidual, lambda: DirectSolve(lam=0.03), GraphSL, RBCLocalizer,
              ConstantScore, AntiEnergy, RandomScore]
        d1 = run_comparison(rl, sp["train"], sp["test"], g, op, seeds=seeds, candidates=cands)
        d2 = run_comparison([GDN, TranAD, RCD, AERCA], series_view(sp["train"], args.series_channel),
                            series_view(sp["test"], args.series_channel), g, op, seeds=seeds, candidates=cands)
        for d in (d1, d2):
            d.insert(0, "arm", "baseline")
            rows.append(d)

    full = pd.concat(rows, ignore_index=True)
    full.insert(0, "dataset", tag)
    full["n_candidates"] = K
    full["chance"] = 1.0 / K
    full.to_csv(out / f"{tag}_sweep.csv", index=False)
    show = [c for c in ("arm", "localizer", "top1", "top1_sd", "hard_top1", "hard_top1_sd", "hit@3", "fit_s")
            if c in full.columns]
    fam = sorted(c for c in full.columns if c.startswith("top1_") and c != "top1_sd")
    print(f"\n=== {tag} ===\n{full[show + fam].sort_values('hard_top1', ascending=False).to_string(index=False)}",
          flush=True)
    (out / f"{tag}_sweep.json").write_text(json.dumps(
        {"dataset": tag, "n_candidates": K, "chance": 1 / K, "field_dim": int(eps[0].r.shape[1]),
         "n_episodes": len(eps), "seeds": args.seeds}, indent=2))
    print("saved to", out, flush=True)


main()
