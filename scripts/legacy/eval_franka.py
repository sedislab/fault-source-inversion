"""Evaluate FSI against the baselines on the real Franka dataset. Residual-based localizers (largest-residual,
direct-l1, GraphSL, FSI) read the residual field; time-series localizers (GDN, TranAD, RCD, AERCA) read the raw
joint-torque signal. Same metric harness as everywhere else, so the numbers are directly comparable."""
import os, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])  # repository root (code)
ROOT = os.environ.get("FSI_ROOT", REPO)  # data/results root, see docs/CONFIGURATION.md
import sys, dataclasses
from pathlib import Path
import torch
torch.set_num_threads(8)
sys.path.insert(0, REPO)
from fsi.core.types import Operator
from fsi.core.graph import StructuralGraph
from fsi.data import load_episodes, make_splits
from fsi.baselines import LargestResidual, DirectSolve, GDN, TranAD, GraphSL, RCD, AERCA
from fsi.models import FSILocalizer
from fsi.eval import run_comparison, save_table
from fsi.eval.stats import characterize

DATA = Path(f"{ROOT}/data/franka")
OUT = Path(f"{ROOT}/results/franka")

def torque_view(eps):
    return [dataclasses.replace(e, y=(e.y[..., 2] if e.y is not None else None)) for e in eps]

def main():
    import numpy as np
    eps = load_episodes(DATA / "all")
    if len(eps) > 2600:
        rng = np.random.default_rng(0)
        eps = [eps[i] for i in rng.choice(len(eps), 2600, replace=False)]
    g = StructuralGraph.load(DATA / "graph.json")
    op = Operator.load(DATA / "operator_G.npz")
    sp = make_splits(eps, seed=0)
    train, test = sp["train"], sp["test"]
    print("dataset:", {k: characterize(eps, g, op)[k] for k in ("n_episodes", "cause_victim_gap", "coherence")}, flush=True)
    residual_locs = [LargestResidual(), DirectSolve(lam=0.03), GraphSL(), FSILocalizer(epochs=80)]
    series_locs = [GDN(), TranAD(), RCD(), AERCA()]
    df1 = run_comparison(residual_locs, train, test, g, op)
    df2 = run_comparison(series_locs, torque_view(train), torque_view(test), g, op)
    import pandas as pd, json
    df = pd.concat([df1, df2], ignore_index=True).sort_values("cvv", ascending=False)
    save_table(df, OUT / "franka_localization", title="Franka (real Isaac Lab) localization comparison")
    print(df.to_string(index=False), flush=True)

    from fsi.metrics import evaluate_detection
    from fsi.models import ConformalSupport
    from fsi.eval.runner import fitted
    cal_heal = [e for e in sp["cal"] if e.is_healthy] or [e for e in train if e.is_healthy][:200]
    det = evaluate_detection(test, cal_heal, target_far=0.05)
    print("detection:", {k: round(v, 4) for k, v in det.items() if isinstance(v, float)}, flush=True)

    extra = {"detection": det}
    try:
        fsi = fitted(residual_locs, "fsi")
        cal_f = [e for e in sp["cal"] if not e.is_healthy]
        conf = ConformalSupport(fsi, mode="crc", alpha=0.1,
                                group_by=lambda e: e.faults[0].fault_type if e.faults else "none").calibrate(cal_f, g, op)
        cov = conf.evaluate([e for e in test if not e.is_healthy], g, op)
        print("FSI conformal (crc, alpha=0.1):", {k: round(v, 4) for k, v in cov.items()}, flush=True)
        extra["conformal_fsi"] = cov
    except Exception as e:
        print("conformal failed:", type(e).__name__, str(e)[:200], flush=True)
    (OUT / "franka_detection_conformal.json").write_text(json.dumps(extra, indent=2, default=float))
    print("saved to", OUT, flush=True)

if __name__ == "__main__":
    main()
