"""Part 1 on the RELEASED datasets (data/franka_v6, data/anymal_v4) -- i.e. the exact residual fields the
published operating point was computed from, one forward-model seed. The multi-seed version is
detection_seeds.py, which retrains f_theta from raw. This script exists so the curves can be tied to the
published numbers (Franka recall .467 @ FAR .039; ANYmal recall .303 @ FAR .038) and so the analysis code is
validated before the expensive job.

Splits are reproduced exactly as scripts/eval_embodiment.py builds them: cap 6000 with default_rng(0), then
fsi.data.make_splits(seed=0).
"""
import os, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])  # repository root (code)
ROOT = os.environ.get("FSI_ROOT", REPO)  # data/results root, see docs/CONFIGURATION.md
import sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/scripts/paper")
from fsi.data import load_episodes, make_splits
from fsi.metrics.detection import detection_score, calibrate_threshold, evaluate_detection
import detection_lib as dl

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=f"{ROOT}/results/paper/detection")
ap.add_argument("--cap", type=int, default=6000)
args = ap.parse_args()

DATASETS = [("franka", f"{ROOT}/data/franka_v6"), ("anymal", f"{ROOT}/data/anymal_v4")]


def main():
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    acc = {k: [] for k in ("pr_roc", "far_sweep", "cusum", "onset")}
    for emb, path in DATASETS:
        eps = load_episodes(Path(path) / "all")
        rng = np.random.default_rng(0)
        if len(eps) > args.cap:
            eps = [eps[i] for i in rng.choice(len(eps), args.cap, replace=False)]
        sp = make_splits(eps, seed=0)
        cal_h = [e for e in sp["cal"] if e.is_healthy]
        pub = evaluate_detection(sp["test"], cal_h, target_far=0.05)
        print(f"== {emb}: {len(eps)} eps | published-protocol check "
              f"recall {pub['recall']:.4f} far {pub['far']:.4f} auprc {pub['auprc']:.4f}", flush=True)
        meta = {"embodiment": emb, "dataset": Path(path).name, "fwd_seed": "released",
                "n_nodes": eps[0].n}
        res = dl.analyze_detection(sp, meta)
        for k in acc:
            acc[k].extend(res[k])
    for k, rows in acc.items():
        pd.DataFrame(rows).to_csv(out / f"{k}_released.csv", index=False)
        print("wrote", out / f"{k}_released.csv", len(rows), "rows", flush=True)


main()
