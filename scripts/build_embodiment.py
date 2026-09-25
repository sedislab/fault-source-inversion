"""Generic stage B: raw rollouts -> f_theta -> residual fields -> matched-pair operator -> labeled episodes.
Works for any embodiment; the graph and raw directory are arguments. The operator comes from the unit-fault
sweep MINUS its matched healthy reference, so each column is the fault's own contribution."""
import os, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[1])  # repository root (code)
ROOT = os.environ.get("FSI_ROOT", REPO)  # data/results root, see docs/CONFIGURATION.md
import sys, json, glob, os, argparse
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, REPO)
from fsi.graph import franka_joint_graph, anymal_sim_graph
from fsi.graph.embodiments import franka_sensor_graph, anymal_sensor_graph
from fsi.core.types import Episode, FaultLabel, Operator, Onset
from fsi.models import ForwardModel, train_forward, residual_field, standardize_fields, normalize_signals
from fsi.models.forward import operator_channels
from fsi.eval.stats import characterize, report_md

GRAPHS = {"franka_joint": franka_joint_graph, "anymal_sim": anymal_sim_graph,
          "franka_sensor": franka_sensor_graph, "anymal_sensor": anymal_sensor_graph}

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--graph", required=True, choices=list(GRAPHS))
ap.add_argument("--channels", type=int, default=3)
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--window", type=int, default=60)
ap.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 8])
args = ap.parse_args()

def load_group(pattern, g, raw):
    eps = []
    for f in sorted(glob.glob(f"{raw}/{pattern}")):
        d = np.load(f, allow_pickle=True)
        labels = json.loads(str(d["labels"])) if "labels" in d.files else None
        base = os.path.basename(f)
        task = base.split("_healthy")[0].split("_fault")[0].split("_sweep")[0]
        sig, act = d["signals"], d["actions"]
        for e in range(sig.shape[0]):
            lab = labels[e] if labels else None
            faults = []
            if lab is not None:
                onset_kind = Onset.ABRUPT if lab.get("ramp", 0) == 0 else Onset.INCIPIENT
                faults = [FaultLabel(lab["joint"], lab["kind"], lab["severity"], onset_kind, lab["onset"])]
            eps.append(Episode(args.graph, g.n, np.zeros((g.n, 1), np.float32), faults, y=sig[e], u=act[e],
                               meta={"onset_step": lab["onset"] if lab else 0, "task": task}))
    return eps

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = GRAPHS[args.graph]()
    heal = load_group("*_healthy_*.npz", g, args.raw)
    fault = load_group("*_fault_*.npz", g, args.raw)
    sweep = load_group("*_sweep_*.npz", g, args.raw)
    sweepref = load_group("*_sweepref_*.npz", g, args.raw)
    tasks = sorted({e.meta["task"] for e in heal + fault})
    print(f"healthy {len(heal)} | fault {len(fault)} | sweep {len(sweep)}/{len(sweepref)} | tasks {tasks} | {dev}", flush=True)
    _, sd = normalize_signals(heal + fault + sweep + sweepref, heal)
    fm = ForwardModel(g.node_features().shape[1], du=heal[0].u.shape[1], hist=8, hidden=96, layers=3,
                      channels=args.channels, n_nodes=g.n, horizons=tuple(args.horizons))
    train_forward(fm, heal, g, epochs=args.epochs, device=dev, verbose=True)
    # Healthy episodes carry onset_step=0 and would otherwise skip the pre-onset self-reference that every
    # faulted episode gets, so healthy and faulted fields would be built by two different formulas -- a
    # difference a detector can read off directly (it is what made ANYmal detection recall 0.997 instead of
    # 0.185). Give every fault-free episode a pseudo-onset drawn from the observed fault onset distribution.
    onsets = np.array([e.faults[0].onset_step for e in fault]) if fault else np.array([60])
    rng = np.random.default_rng(0)
    for ep in heal + fault + sweep + sweepref:
        on = None if ep.faults else int(rng.choice(onsets))
        ep.r = residual_field(fm, ep, g, window=args.window, device=dev, onset=on, scale=sd)
    standardize_fields(heal + fault + sweep + sweepref, heal)
    # Read the channel count off the field the reducer actually produced rather than re-deriving it from the
    # horizon/channel args: the two drifted apart the moment `_reduce` gained time bins, and the failure mode is
    # silent (a short G_multi simply drops the tail channels, so the operator loses exactly the new information).
    C = operator_channels(heal[0].r.shape[1])
    G = np.zeros((g.n, g.n), np.float32); Gm = np.zeros((C, g.n, g.n), np.float32)
    matched = bool(sweep) and len(sweepref) == len(sweep)

    def unit_effect(ep, ref):
        """The fault's own counterfactual contribution, clipped ONLY on the energy column.

        Clipping the whole matched-pair difference at 0 -- which is what this did -- strips the sign off every
        operator column and makes G_multi estimate E[max(x,0)] instead of E[x]. Measured on the shipped
        franka_v8 operator: G_multi.min() == 0.0, negative fraction 0.000 across all 75 channels, and 6-10% of
        entries hard-zeroed. That is the sign-collapse this pipeline was already caught making once: any two
        columns of a non-negative matrix have non-negative inner product, so coherence is pinned near 1 by
        arithmetic rather than by physics. It also silently cancels the whole point of `_reduce` emitting SIGNED
        per-channel statistics, and it leaves the field (which `standardize_fields` keeps signed) and the
        operator it is inverted against living in different spaces. Column 0 is a residual ENERGY -- a magnitude
        -- so clipping that one is meaningful; columns 1..F-2 are signed by design and must stay signed."""
        d = ep.r - ref.r
        d[:, 0] = np.clip(d[:, 0], 0, None)
        return d

    src = ([(ep, unit_effect(ep, ref)) for ep, ref in zip(sweep, sweepref)] if matched
           else [(ep, ep.r) for ep in (sweep or fault)])
    print(f"operator from {'MATCHED-PAIR SWEEP' if matched else 'unmatched'} ({len(src)} eps)", flush=True)
    # One column per node from that node's own unit-fault sweep. The kind is whatever the sweep injected there,
    # not a hard-coded "torque_loss": on the sensor graphs the canonical unit fault at an encoder node is an
    # encoder bias, and filtering on torque_loss would silently leave every sensor column at zero -- a zero
    # column has no fingerprint, so RBC marks the node dead and no method can ever name it.
    for j in range(g.n):
        at_j = [(ep, d) for ep, d in src if ep.support() == [j]]
        if not at_j:
            continue
        kinds_j = {ep.faults[0].fault_type for ep, _ in at_j}
        assert len(kinds_j) == 1, f"node {j}: sweep mixes fault kinds {kinds_j}; one column cannot mean two things"
        sev = np.array([max(ep.faults[0].magnitude, 1e-3) for ep, _ in at_j])[:, None]
        # MEAN, not median: which victim lights up varies with stance/phase, so a median collapses the
        # fingerprint to zero for most nodes even when propagation is strong.
        G[:, j] = np.mean(np.stack([d[:, 0] for _, d in at_j]) / sev, 0)
        for c in range(C):
            Gm[c, :, j] = np.mean(np.stack([d[:, 1 + c] for _, d in at_j]) / sev, 0)
    op = Operator(args.graph, G, g.names(), {"matched": matched, "channels": C}, G_multi=Gm)
    faulted_nodes = sorted({e.support()[0] for e in fault})
    for c in range(C):
        col = Gm[c] / (np.linalg.norm(Gm[c], axis=0, keepdims=True) + 1e-9)
        M = np.abs(col.T @ col); np.fill_diagonal(M, 0)
        sub = M[np.ix_(faulted_nodes, faulted_nodes)]
        dd = np.mean([np.argmax(Gm[c][:, j]) == j for j in faulted_nodes])
        print(f"  ch{c}: coherence(all) {M.max():.4f} coherence(faultable) {sub.max():.4f} diag-dom {dd:.2f}", flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    from fsi.data import save_episodes
    save_episodes(heal + fault, out / "all")
    op.save(out / "operator_G.npz"); g.save(out / "graph.json")
    st = characterize(heal + fault, g, op)
    (out / "characterization.md").write_text(report_md(st))
    print("OVERALL:", {k: st[k] for k in ("n_episodes", "cause_victim_gap", "mean_propagation_hops", "coherence")}, flush=True)
    print("saved to", out, flush=True)

main()
