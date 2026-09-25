"""Decisive test: give FSI an IDEAL residual and see what it achieves. Matched sweep pairs provide the
ground-truth counterfactual (identical command + init, faulted vs not), i.e. the residual f_theta is trying to
approximate. If FSI reaches ~1.0 here, the method is sound and the only practical gap is forward-model quality."""
import os, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])  # repository root (code)
ROOT = os.environ.get("FSI_ROOT", REPO)  # data/results root, see docs/CONFIGURATION.md
import sys, json, glob, argparse
import numpy as np
sys.path.insert(0, REPO)
from fsi.graph import franka_joint_graph, anymal_sim_graph
from fsi.core.types import Episode, FaultLabel, Operator, Onset
from fsi.models.forward import _reduce, operator_channels
from fsi.models import FSILocalizer, RBCLocalizer
from fsi.baselines import LargestResidual, DirectSolve, GraphSL
from fsi.metrics import evaluate_localization

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True)
ap.add_argument("--graph", required=True, choices=["franka_joint", "anymal_sim"])
args = ap.parse_args()
g = {"franka_joint": franka_joint_graph, "anymal_sim": anymal_sim_graph}[args.graph]()

S = sorted(glob.glob(f"{args.raw}/*_sweep_*.npz")); R = sorted(glob.glob(f"{args.raw}/*_sweepref_*.npz"))
eps = []
for fs, fr in zip(S, R):
    ds, dr = np.load(fs, allow_pickle=True), np.load(fr, allow_pickle=True)
    lab = json.loads(str(ds["labels"])); on = int(lab[0]["onset"])
    d = ds["signals"] - dr["signals"]          # ORACLE counterfactual residual
    for e in range(d.shape[0]):
        post = d[e, on:]
        pre = d[e, max(0, on - 40):on]
        f = _reduce(np.sqrt((post ** 2).sum(-1)), post, post.shape[-1])
        if len(pre) >= 5:
            f = f - _reduce(np.sqrt((pre ** 2).sum(-1)), pre, pre.shape[-1])
        L = lab[e]
        eps.append(Episode(args.graph, g.n, f.astype(np.float32),
                           [FaultLabel(L["joint"], L["kind"], L["severity"], Onset.ABRUPT, on)],
                           meta={"onset_step": on}))
rng = np.random.default_rng(0); rng.shuffle(eps)
ntr = int(0.7 * len(eps)); train, test = eps[:ntr], eps[ntr:]
print(f"{len(eps)} oracle episodes ({len(train)} train / {len(test)} test), {g.n} nodes, "
      f"{len(g.fail_nodes)} candidates, chance={1/len(g.fail_nodes):.3f}", flush=True)

# field_dim - 2, not - 1: the LAST column is the onset time, a fraction in [0,1] that does not scale with
# severity, so dividing it by severity to build an operator column is meaningless. `operator_channels` is the
# one place that arithmetic lives, so it cannot drift from `_reduce`'s layout again.
C = operator_channels(eps[0].r.shape[1])
G = np.zeros((g.n, g.n), np.float32); Gm = np.zeros((C, g.n, g.n), np.float32)
for j in range(g.n):
    at = [e for e in train if e.support() == [j]]
    if not at:
        continue
    sev = np.array([max(e.faults[0].magnitude, 1e-3) for e in at])[:, None]
    G[:, j] = np.mean(np.stack([e.r[:, 0] for e in at]) / sev, 0)
    for c in range(C):
        Gm[c, :, j] = np.mean(np.stack([e.r[:, 1 + c] for e in at]) / sev, 0)
op = Operator(args.graph, G, g.names(), {"oracle": True}, G_multi=Gm)
col = G / (np.linalg.norm(G, axis=0, keepdims=True) + 1e-9)
M = np.abs(col.T @ col); np.fill_diagonal(M, 0)
print(f"oracle operator coherence {M.max():.3f}", flush=True)

for loc in [LargestResidual(), DirectSolve(lam=0.03), GraphSL(), RBCLocalizer(), FSILocalizer(epochs=80)]:
    try:
        loc.fit(train, g, op)
        m = evaluate_localization(loc.predict_many(test, g, op), test, g, op)
        print(f"  {loc.name:16s} top1={m['top1']:.3f}  cvv={m['cvv']:.3f}  hit@3={m['hit@3']:.3f}  mrr={m['mrr']:.3f}", flush=True)
    except Exception as e:
        print(f"  {loc.name:16s} FAILED {type(e).__name__}: {str(e)[:100]}", flush=True)
