"""Evaluate on the multi-task Franka corpus, two ways: in-distribution (random episode split across tasks) and
HELD-OUT-TASK (train on the other tasks, test on an unseen one). The second is what earns the task-agnostic
claim. Full metric suite: localization (CvV, W1G, Hit@k, MRR, support-F1), detection, conformal, latency."""
import os, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])  # repository root (code)
ROOT = os.environ.get("FSI_ROOT", REPO)  # data/results root, see docs/CONFIGURATION.md
import sys, dataclasses, json
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(32)
sys.path.insert(0, REPO)
from fsi.core.types import Operator
from fsi.core.graph import StructuralGraph
from fsi.data import load_episodes, make_splits
from fsi.baselines import LargestResidual, DirectSolve, GDN, TranAD, GraphSL, RCD, AERCA
from fsi.models import FSILocalizer, ConformalSupport
from fsi.eval import run_comparison, save_table
from fsi.eval.runner import fitted
from fsi.eval.stats import characterize
from fsi.metrics import evaluate_detection

DATA = Path(f"{ROOT}/data/franka_v2")
OUT = Path(f"{ROOT}/results/franka_v2")
CAP = 6000

def torque_view(eps):
    return [dataclasses.replace(e, y=(e.y[..., 2] if e.y is not None else None)) for e in eps]

def residual_locs():
    return [LargestResidual(), DirectSolve(lam=0.03), GraphSL(), FSILocalizer(epochs=80)]

def series_locs():
    return [GDN(), TranAD(), RCD(), AERCA()]

def compare(train, test, g, op, tag):
    rl, sl = residual_locs(), series_locs()
    import pandas as pd
    df = pd.concat([run_comparison(rl, train, test, g, op),
                    run_comparison(sl, torque_view(train), torque_view(test), g, op)],
                   ignore_index=True).sort_values("cvv", ascending=False)
    save_table(df, OUT / f"franka_v2_{tag}", title=f"Franka v2 ({tag})")
    print(f"--- {tag} ---\n{df.to_string(index=False)}", flush=True)
    return df, rl

def main():
    eps = load_episodes(DATA / "all")
    rng = np.random.default_rng(0)
    if len(eps) > CAP:
        eps = [eps[i] for i in rng.choice(len(eps), CAP, replace=False)]
    g = StructuralGraph.load(DATA / "graph.json")
    op = Operator.load(DATA / "operator_G.npz")
    tasks = sorted({e.meta.get("task", "?") for e in eps})
    print("dataset:", {k: characterize(eps, g, op)[k] for k in ("n_episodes", "cause_victim_gap", "coherence")},
          "| tasks:", tasks, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    sp = make_splits(eps, seed=0)
    df_id, rl = compare(sp["train"], sp["test"], g, op, "in_distribution")

    extra = {}
    cal_heal = [e for e in sp["cal"] if e.is_healthy]
    if cal_heal:
        det = evaluate_detection(sp["test"], cal_heal, target_far=0.05)
        extra["detection"] = det
        print("detection:", {k: round(v, 4) for k, v in det.items() if isinstance(v, float)}, flush=True)
    try:
        conf = ConformalSupport(fitted(rl, "fsi"), mode="crc", alpha=0.1,
                                group_by=lambda e: e.faults[0].fault_type if e.faults else "none").calibrate(
                                [e for e in sp["cal"] if not e.is_healthy], g, op)
        cov = conf.evaluate([e for e in sp["test"] if not e.is_healthy], g, op)
        extra["conformal_fsi"] = cov
        print("FSI conformal:", {k: round(v, 4) for k, v in cov.items()}, flush=True)
    except Exception as e:
        print("conformal failed:", type(e).__name__, str(e)[:150], flush=True)

    held = {}
    for t in tasks:
        tr = [e for e in eps if e.meta.get("task") != t]
        te = [e for e in eps if e.meta.get("task") == t]
        if len(te) < 50 or len(tr) < 200:
            continue
        df_t, _ = compare(tr, te, g, op, f"heldout_{t}")
        held[t] = {r["localizer"]: {k: r.get(k) for k in ("top1", "cvv", "w1g", "hit@3")} for _, r in df_t.iterrows()}
    extra["heldout_task"] = held
    (OUT / "franka_v2_detection_conformal_heldout.json").write_text(json.dumps(extra, indent=2, default=float))
    print("saved to", OUT, flush=True)

if __name__ == "__main__":
    main()
