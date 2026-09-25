"""Settle whether mag_w actually controls the operator-inversion branch.

I claimed earlier that mag_w=1 left the unrolled proximal-gradient path dead (std of mag_skip*s = 0.001 vs 1.433
for the full logit) and that raising it to 50 revived FSI. The ablation then measured mag_w=50 vs 1.0 at
delta +0.002 (z=+0.4) on ANYmal and +0.003 (z=+0.1) on Franka, on an IDENTICAL dataset build -- i.e. no effect.
Both cannot be right. This probe measures the branch directly at each mag_w instead of inferring it."""
import os, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])  # repository root (code)
ROOT = os.environ.get("FSI_ROOT", REPO)  # data/results root, see docs/CONFIGURATION.md
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, REPO)
from fsi.core.graph import StructuralGraph
from fsi.core.types import Operator
from fsi.data import load_episodes, make_splits
from fsi.models import FSILocalizer
from fsi.models.inverse import fidelity, norm_adj
from fsi.metrics.localization import hard_top1

for name in ("anymal_v4", "franka_v6"):
    d = Path(f"{ROOT}/data/{name}")
    eps = load_episodes(d / "all")
    g = StructuralGraph.load(d / "graph.json"); op = Operator.load(d / "operator_G.npz")
    sp = make_splits(eps, seed=0); tr, te = sp["train"], sp["test"]
    cands = sorted({s for e in tr for s in e.support()})
    cands = [v for v in g.fail_nodes if v in set(cands)]
    cands = cands if 0 < len(cands) < len(g.fail_nodes) else None
    print(f"\n=== {name} ===", flush=True)
    for mag_w in (1.0, 50.0):
        for deep in (0.3, 0.0):
            torch.manual_seed(0); np.random.seed(0)
            loc = FSILocalizer(epochs=80, mag_w=mag_w)
            # deep supervision is not exposed on the localizer; patch it for this probe only
            import fsi.models.inverse as inv
            orig = inv.train_inverse
            inv.train_inverse = lambda *a, **k: orig(*a, **{**k, "deep": deep})
            loc.fit(tr, g, op)
            inv.train_inverse = orig
            # measure the two contributions to the ranking, per episode, over candidates
            mags, sups = [], []
            with torch.no_grad():
                for e in te[:400]:
                    r = e.r if e.r.ndim == 2 else e.r[:, None]
                    scale = np.linalg.norm(r[:, 0]) + 1e-6
                    rt = torch.tensor((r / scale)[None].astype(np.float32))
                    Gm, rc = fidelity(loc.operator, rt, "cpu")
                    logit, s, _ = loc.model(rt, Gm, rc, loc._A, loc._nf)
                    contrib = (loc.model.mag_skip * s)[0].numpy()
                    mags.append(contrib.std()); sups.append(logit[0].numpy().std())
            m = hard_top1(loc.predict_many(te, g, op), te, g, cands)
            print(f"  mag_w={mag_w:5.1f} deep={deep:.1f} | std(mag_skip*s)={np.mean(mags):8.4f} "
                  f"std(logit)={np.mean(sups):7.4f} | hard_top1={m['hard_top1']:.3f}", flush=True)
print("\nDONE", flush=True)
