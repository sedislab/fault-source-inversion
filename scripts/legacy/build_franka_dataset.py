"""Stage B for Franka: turn raw rollouts into the FSI dataset. Train f_theta on healthy rollouts, compute the
residual field for every episode, estimate the operator G from the labeled fault corpus (column j = mean residual
per unit severity of faults at joint j), and save labeled episodes + G + graph. Runs on CPU or a GPU node."""
import os, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])  # repository root (code)
ROOT = os.environ.get("FSI_ROOT", REPO)  # data/results root, see docs/CONFIGURATION.md
import sys, json, glob
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, REPO)
from fsi.graph import franka_joint_graph
from fsi.core.types import Episode, FaultLabel, Operator, Onset
from fsi.models import ForwardModel, train_forward, residual_field, standardize_fields
from fsi.eval.stats import characterize, report_md

RAW = f"{ROOT}/raw/franka"
OUT = Path(f"{ROOT}/data/franka")
WINDOW = 60

def load_raw():
    heal, fault = [], []
    for f in sorted(glob.glob(f"{RAW}/healthy_*.npz")):
        d = np.load(f)
        for e in range(d["signals"].shape[0]):
            heal.append((d["signals"][e], d["actions"][e], None))
    for f in sorted(glob.glob(f"{RAW}/fault_*.npz")):
        d = np.load(f); labels = json.loads(str(d["labels"]))
        for e in range(d["signals"].shape[0]):
            fault.append((d["signals"][e], d["actions"][e], labels[e]))
    return heal, fault

def make_episode(sig, act, lab, g):
    faults = []
    if lab is not None:
        faults = [FaultLabel(lab["joint"], lab["kind"], lab["severity"], Onset.ABRUPT, lab["onset"])]
    return Episode("franka_joint", g.n, np.zeros((g.n, 3), np.float32), faults, y=sig, u=act,
                   meta={"onset_step": lab["onset"] if lab else 0})

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = franka_joint_graph()
    heal_raw, fault_raw = load_raw()
    print(f"loaded {len(heal_raw)} healthy, {len(fault_raw)} fault episodes; device {dev}", flush=True)
    heal = [make_episode(*x, g) for x in heal_raw]
    fault = [make_episode(*x, g) for x in fault_raw]
    fm = ForwardModel(g.node_features().shape[1], du=heal[0].u.shape[1], hist=8, hidden=64, layers=2, channels=3)
    train_forward(fm, heal, g, epochs=40, device=dev, verbose=True)
    print("f_theta trained; computing residual fields...", flush=True)
    for ep in heal + fault:
        ep.r = residual_field(fm, ep, g, window=WINDOW, device=dev)
    standardize_fields(heal + fault, heal)   # cancel f_theta's model-error floor
    C = 3 + 3                     # per-channel rms + early/mid/late; timing channels stay prior-only (not severity-linear)
    G = np.zeros((g.n, g.n), np.float32)
    Gm = np.zeros((C, g.n, g.n), np.float32)
    for j in range(g.n):
        at_j = [ep for ep in fault if ep.support() == [j] and ep.faults[0].fault_type == "torque_loss"]
        if not at_j:
            continue
        sev = np.array([max(ep.faults[0].magnitude, 1e-3) for ep in at_j])[:, None]
        G[:, j] = np.median(np.stack([ep.energy() for ep in at_j]) / sev, 0)
        for c in range(C):
            Gm[c, :, j] = np.median(np.stack([ep.r[:, 1 + c] for ep in at_j]) / sev, 0)
    op = Operator("franka_joint", G, g.names(), {"kind": "corpus_estimated", "channels": C}, G_multi=Gm)
    for c in range(C):
        col = Gm[c] / (np.linalg.norm(Gm[c], axis=0, keepdims=True) + 1e-9)
        M = np.abs(col.T @ col); np.fill_diagonal(M, 0)
        print(f"  channel {c}: coherence {M.max():.4f}, diag-dominance {np.mean(np.argmax(Gm[c], 0) == np.arange(g.n)):.2f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    from fsi.data import save_episodes
    save_episodes(heal + fault, OUT / "all")
    op.save(OUT / "operator_G.npz"); g.save(OUT / "graph.json")
    st = characterize(heal + fault, g, op)
    (OUT / "characterization.md").write_text(report_md(st))
    print("DATASET STATS:", {k: st[k] for k in ("n_episodes", "cause_victim_gap", "mean_propagation_hops", "coherence")}, flush=True)
    print("saved to", OUT, flush=True)

if __name__ == "__main__":
    main()
