"""Stage B for the multi-task Franka corpus. Trains f_theta on healthy rollouts across tasks, computes residual
fields, and estimates the operator from the dedicated UNIT-FAULT SWEEP (fixed severity, systematic joint, many
command variations) so each column is a property of the fault rather than of a trajectory. Episodes carry their
task so a held-out-task split can test whether detection/localization is task-agnostic."""
import os, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])  # repository root (code)
ROOT = os.environ.get("FSI_ROOT", REPO)  # data/results root, see docs/CONFIGURATION.md
import sys, json, glob, os
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, REPO)
from fsi.graph import franka_joint_graph
from fsi.core.types import Episode, FaultLabel, Operator, Onset
from fsi.models import ForwardModel, train_forward, residual_field, standardize_fields, normalize_signals
from fsi.eval.stats import characterize, report_md

RAW = f"{ROOT}/raw/franka_v2"
OUT = Path(f"{ROOT}/data/franka_v2")
WINDOW = 60

def load_group(pattern, g):
    eps = []
    for f in sorted(glob.glob(f"{RAW}/{pattern}")):
        d = np.load(f, allow_pickle=True)
        labels = json.loads(str(d["labels"])) if "labels" in d.files else None
        task = os.path.basename(f).split("_healthy")[0].split("_fault")[0].split("_sweep")[0]
        for e in range(d["signals"].shape[0]):
            lab = labels[e] if labels else None
            faults = []
            if lab is not None:
                onset_kind = Onset.ABRUPT if lab.get("ramp", 0) == 0 else Onset.INCIPIENT
                faults = [FaultLabel(lab["joint"], lab["kind"], lab["severity"], onset_kind, lab["onset"])]
            eps.append(Episode("franka_joint", g.n, np.zeros((g.n, 4), np.float32), faults,
                               y=d["signals"][e], u=d["actions"][e],
                               meta={"onset_step": lab["onset"] if lab else 0, "task": task,
                                     "sweep": bool(lab.get("sweep", False)) if lab else False}))
    return eps

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = franka_joint_graph()
    heal = load_group("*_healthy_*.npz", g)
    fault = load_group("*_fault_*.npz", g)
    sweep = load_group("*_sweep_*.npz", g)
    sweepref = load_group("*_sweepref_*.npz", g)
    tasks = sorted({e.meta["task"] for e in heal + fault})
    print(f"healthy {len(heal)} | fault {len(fault)} | sweep {len(sweep)} | tasks {tasks} | dev {dev}", flush=True)
    normalize_signals(heal + fault + sweep + sweepref, heal)   # per-channel scale so the loss isn't torque-dominated
    fm = ForwardModel(g.node_features().shape[1], du=heal[0].u.shape[1], hist=8, hidden=96, layers=3, channels=3)
    train_forward(fm, heal, g, epochs=60, device=dev, verbose=True)
    for ep in heal + fault + sweep + sweepref:
        ep.r = residual_field(fm, ep, g, window=WINDOW, device=dev)
    standardize_fields(heal + fault + sweep + sweepref, heal)   # cancel f_theta's model-error floor
    NCH = 3                       # signal channels (pos, vel, torque)
    C = NCH + 3                   # fidelity channels: per-channel rms + early/mid/late (these scale with severity)
    G = np.zeros((g.n, g.n), np.float32)
    Gm = np.zeros((C, g.n, g.n), np.float32)
    matched = bool(sweep) and len(sweepref) == len(sweep)
    if matched:
        # fingerprint = faulted MINUS its matched healthy reference (same command + init), so the column is the
        # fault's own contribution rather than the command-driven response every fault shares
        src = [(ep, np.clip(ep.r - ref.r, 0, None)) for ep, ref in zip(sweep, sweepref)]
    else:
        src = [(ep, ep.r) for ep in (sweep if sweep else fault)]
    print(f"operator from {'MATCHED-PAIR SWEEP' if matched else 'unmatched'} ({len(src)} episodes)", flush=True)
    for j in range(g.n):
        at_j = [(ep, d) for ep, d in src if ep.support() == [j] and ep.faults[0].fault_type == "torque_loss"]
        if not at_j:
            continue
        sev = np.array([max(ep.faults[0].magnitude, 1e-3) for ep, _ in at_j])[:, None]
        G[:, j] = np.median(np.stack([d[:, 0] for _, d in at_j]) / sev, 0)
        for c in range(C):
            Gm[c, :, j] = np.median(np.stack([d[:, 1 + c] for _, d in at_j]) / sev, 0)
    op = Operator("franka_joint", G, g.names(), {"kind": "sweep" if sweep else "corpus", "channels": C}, G_multi=Gm)
    for c in range(C):
        col = Gm[c] / (np.linalg.norm(Gm[c], axis=0, keepdims=True) + 1e-9)
        M = np.abs(col.T @ col); np.fill_diagonal(M, 0)
        print(f"  channel {c}: coherence {M.max():.4f} diag-dominance {np.mean(np.argmax(Gm[c], 0) == np.arange(g.n)):.2f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    from fsi.data import save_episodes
    save_episodes(heal + fault, OUT / "all")
    op.save(OUT / "operator_G.npz"); g.save(OUT / "graph.json")
    st = characterize(heal + fault, g, op)
    (OUT / "characterization.md").write_text(report_md(st))
    print("OVERALL:", {k: st[k] for k in ("n_episodes", "cause_victim_gap", "mean_propagation_hops", "coherence")}, flush=True)
    for t in tasks:
        sub = [e for e in heal + fault if e.meta["task"] == t]
        s2 = characterize(sub, g, op)
        print(f"  {t}: n={s2['n_episodes']} gap={s2['cause_victim_gap']:.3f}", flush=True)
    print("saved to", OUT, flush=True)

if __name__ == "__main__":
    main()
