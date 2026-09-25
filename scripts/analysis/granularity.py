"""At what granularity is a fault actually localizable? Score joint-level, limb-level and joint-type-level
accuracy for the same predictions. If limb accuracy is high while joint accuracy is near chance, the fault is
identifiable to a subsystem and within-limb joints are genuinely near-collinear -- a scope result, not a defect."""
import os, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])  # repository root (code)
ROOT = os.environ.get("FSI_ROOT", REPO)  # data/results root, see docs/CONFIGURATION.md
import sys, argparse
import numpy as np
sys.path.insert(0, REPO)
from pathlib import Path
from fsi.core.types import Operator
from fsi.core.graph import StructuralGraph
from fsi.data import load_episodes, make_splits
from fsi.baselines import LargestResidual, TranAD, GraphSL
from fsi.models import FSILocalizer, RBCLocalizer
from fsi.metrics import detection_score, calibrate_threshold

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--cap", type=int, default=4000)
args = ap.parse_args()

data = Path(args.data)
eps = load_episodes(data / "all")
rng = np.random.default_rng(0)
if len(eps) > args.cap:
    eps = [eps[i] for i in rng.choice(len(eps), args.cap, replace=False)]
g = StructuralGraph.load(data / "graph.json")
op = Operator.load(data / "operator_G.npz")
sp = make_splits(eps, seed=0)
cal_h = [e for e in sp["cal"] if e.is_healthy]
thr = calibrate_threshold([detection_score(e) for e in cal_h], 0.05) if cal_h else -np.inf
test = [e for e in sp["test"] if not e.is_healthy and detection_score(e) > thr]
print(f"detected test faults: {len(test)}", flush=True)

names = g.names()
def limb(i):
    n = names[i]
    return n.split("_")[0] if "_" in n else n
def jtype(i):
    n = names[i]
    return n.split("_")[1] if "_" in n else n
fail = np.asarray(g.fail_nodes)
srcs = sorted({e.support()[0] for e in test})
print(f"true source nodes: {[names[s] for s in srcs]}", flush=True)
print(f"chance: joint={1/len(fail):.3f} limb={1/len(set(limb(i) for i in fail)):.3f} "
      f"type={1/len(set(jtype(i) for i in fail)):.3f}", flush=True)

for loc in [LargestResidual(), GraphSL(), TranAD(), RBCLocalizer(), FSILocalizer(epochs=60)]:
    try:
        loc.fit(sp["train"], g, op)
        preds = loc.predict_many(test, g, op)
        j = l = t = 0
        for p, e in zip(preds, test):
            top = fail[int(np.argmax(p.scores[fail]))]
            s = e.support()[0]
            j += top == s
            l += limb(top) == limb(s)
            t += jtype(top) == jtype(s)
        n = len(test)
        print(f"  {loc.name:16s} joint={j/n:.3f}  limb={l/n:.3f}  jointtype={t/n:.3f}", flush=True)
    except Exception as ex:
        print(f"  {loc.name:16s} FAILED {type(ex).__name__}: {str(ex)[:90]}", flush=True)
