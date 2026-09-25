"""Part 1 with >=3 forward-model seeds, plus the onset-deployability result.

One job = one (embodiment, forward-model seed). It retrains f_theta from the raw rollouts exactly as
scripts/build_embodiment.py does (same hyperparameters, same pseudo-onset for healthy episodes, same
matched-pair operator), so each seed is a full independent draw of the residual field the detector reads.
The published datasets data/franka_v6 and data/anymal_v4 are ONE such draw with an unrecorded torch seed;
detection_released.py reports that draw, this script reports the spread.

Deployability: `residual_field` currently reads onset_step from the LABEL, which does not exist at deployment.
Here the same pipeline is rebuilt with the CUSUM change-point estimate in its place and the localizer is
re-scored, so the cost of removing the oracle onset is measured rather than assumed. The operator G is still
built with true onsets -- it comes from a simulator sweep where the injection time is known by construction --
so only the test-time (and optionally train-time) field uses an estimate.
"""
import os, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])  # repository root (code)
ROOT = os.environ.get("FSI_ROOT", REPO)  # data/results root, see docs/CONFIGURATION.md
import sys, os, glob, json, argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/scripts/paper")
from fsi.graph import franka_joint_graph, anymal_sim_graph
from fsi.core.types import Episode, FaultLabel, Operator, Onset
from fsi.models import ForwardModel, train_forward, residual_field, standardize_fields, normalize_signals
from fsi.models.forward import operator_channels
from fsi.models import FSILocalizer
from fsi.baselines import LargestResidual, ConstantScore, AntiEnergy, RandomScore
from fsi.data import make_splits
from fsi.metrics import evaluate_localization
import detection_lib as dl

EMB = {"franka": dict(raw=f"{ROOT}/raw/franka_v4", graph="franka_joint", dataset="franka_v6"),
       "anymal": dict(raw=f"{ROOT}/raw/anymal_v4", graph="anymal_sim", dataset="anymal_v4")}
GRAPHS = {"franka_joint": franka_joint_graph, "anymal_sim": anymal_sim_graph}

ap = argparse.ArgumentParser()
ap.add_argument("--emb", required=True, choices=list(EMB))
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--out", default=f"{ROOT}/results/paper/detection/shards")
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--window", type=int, default=60)
ap.add_argument("--channels", type=int, default=3)
ap.add_argument("--cap", type=int, default=6000)
ap.add_argument("--nu", type=float, default=2.0)
ap.add_argument("--fsi_epochs", type=int, default=80)
args = ap.parse_args()

CFG = EMB[args.emb]
HZ = (1, 4, 8)


def load_group(pattern, g, raw, graph_name):
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
                kind = Onset.ABRUPT if lab.get("ramp", 0) == 0 else Onset.INCIPIENT
                faults = [FaultLabel(lab["joint"], lab["kind"], lab["severity"], kind, lab["onset"])]
            eps.append(Episode(graph_name, g.n, np.zeros((g.n, 1), np.float32), faults,
                               y=sig[e], u=act[e],
                               meta={"onset_step": lab["onset"] if lab else 0, "task": task}))
    return eps


def build_operator(g, sweep, sweepref, channels):
    """Matched-pair operator, verbatim from scripts/build_embodiment.py."""
    C = operator_channels(sweep[0].r.shape[1])
    G = np.zeros((g.n, g.n), np.float32); Gm = np.zeros((C, g.n, g.n), np.float32)
    matched = bool(sweep) and len(sweepref) == len(sweep)
    src = ([(ep, np.clip(ep.r - ref.r, 0, None)) for ep, ref in zip(sweep, sweepref)] if matched
           else [(ep, ep.r) for ep in sweep])
    for j in range(g.n):
        at_j = [(ep, d) for ep, d in src if ep.support() == [j] and ep.faults[0].fault_type == "torque_loss"]
        if not at_j:
            continue
        sev = np.array([max(ep.faults[0].magnitude, 1e-3) for ep, _ in at_j])[:, None]
        G[:, j] = np.mean(np.stack([d[:, 0] for _, d in at_j]) / sev, 0)
        for c in range(C):
            Gm[c, :, j] = np.mean(np.stack([d[:, 1 + c] for _, d in at_j]) / sev, 0)
    return Operator(g.name or CFG["graph"], G, g.names(), {"matched": matched, "channels": C}, G_multi=Gm)


def fields(model, eps, g, dev, sd, onsets):
    """onsets[i] is the onset to hand residual_field for eps[i] (None -> use the label)."""
    return [residual_field(model, ep, g, window=args.window, device=dev, onset=o, scale=sd)
            for ep, o in zip(eps, onsets)]


def main():
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    g = GRAPHS[CFG["graph"]]()
    raw = CFG["raw"]
    heal = load_group("*_healthy_*.npz", g, raw, CFG["graph"])
    fault = load_group("*_fault_*.npz", g, raw, CFG["graph"])
    sweep = load_group("*_sweep_*.npz", g, raw, CFG["graph"])
    sweepref = load_group("*_sweepref_*.npz", g, raw, CFG["graph"])
    print(f"{args.emb} seed {args.seed} | healthy {len(heal)} fault {len(fault)} "
          f"sweep {len(sweep)}/{len(sweepref)} | {dev}", flush=True)

    _, sd = normalize_signals(heal + fault + sweep + sweepref, heal)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    fm = ForwardModel(g.node_features().shape[1], du=heal[0].u.shape[1], hist=8, hidden=96, layers=3,
                      channels=args.channels, n_nodes=g.n, horizons=HZ)
    train_forward(fm, heal, g, epochs=args.epochs, device=dev, verbose=True)
    print(f"forward trained in {time.time() - t0:.0f}s", flush=True)

    # --- fields with the TRUE (oracle) onset, pseudo-onset for healthy -- exactly the released pipeline
    onsets_f = np.array([e.faults[0].onset_step for e in fault])
    rng = np.random.default_rng(0)
    allg = heal + fault + sweep + sweepref
    pseudo = {id(e): (None if e.faults else int(rng.choice(onsets_f))) for e in allg}
    for ep in allg:
        ep.r = residual_field(fm, ep, g, window=args.window, device=dev, onset=pseudo[id(ep)], scale=sd)
    standardize_fields(allg, heal)
    for ep in allg:
        ep.r_true = ep.r
    op = build_operator(g, sweep, sweepref, args.channels)
    print(f"operator built ({time.time() - t0:.0f}s) coherence {op.coherence():.4f}", flush=True)

    # --- splits reproduced exactly as scripts/eval_embodiment.py builds them
    eps = heal + fault
    r2 = np.random.default_rng(0)
    if len(eps) > args.cap:
        eps = [eps[i] for i in r2.choice(len(eps), args.cap, replace=False)]
    sp = make_splits(eps, seed=0)
    realizable = sorted({s for e in sp["train"] + sp["cal"] for s in e.support()})
    cands = [v for v in g.fail_nodes if v in set(realizable)]
    cands = cands if 0 < len(cands) < len(g.fail_nodes) else None
    meta = {"embodiment": args.emb, "dataset": CFG["dataset"] + "_rebuilt", "fwd_seed": args.seed,
            "n_nodes": g.n}
    res = dl.analyze_detection(sp, meta, nu_star=args.nu)
    for k in ("pr_roc", "far_sweep", "cusum", "onset"):
        pd.DataFrame(res[k]).to_csv(out / f"{k}__{args.emb}__s{args.seed}.csv", index=False)
    print(f"detection analysis done ({time.time() - t0:.0f}s)", flush=True)

    # --- deployability: rebuild the field with the CUSUM-estimated onset -------------------
    tr_h = [e for e in sp["train"] if e.is_healthy]
    mu_r, sd_r = dl.healthy_residual_stats(tr_h)
    h = res["onset_threshold"]                      # CUSUM threshold at FAR 0.05 on the cal-healthy split
    est_all, alarmed = {}, {}
    for name in ("train", "cal", "test"):
        chunk = sp[name]
        Z = dl.stream(chunk, mu_r, sd_r)
        G = dl.cusum_batch(Z, args.nu)
        for e, gg in zip(chunk, G):
            oh, alarm = dl._one_onset(gg, h)
            est_all[id(e)] = oh
            alarmed[id(e)] = alarm >= 0
        del Z, G
    n_al = sum(1 for e in sp["test"] if not e.is_healthy and alarmed[id(e)])
    print(f"CUSUM(nu={args.nu}, h={h:.1f}) alarms on {n_al}/"
          f"{sum(1 for e in sp['test'] if not e.is_healthy)} test faults", flush=True)

    # Rebuild every train/cal/test field with the ESTIMATED onset (fall back to the same pseudo-onset draw when
    # CUSUM never alarms, which is what a deployed system with no alarm would have to do).
    used = heal + fault
    for ep in used:
        o = est_all.get(id(ep), -1)
        ep.r = residual_field(fm, ep, g, window=args.window, device=dev,
                              onset=(int(o) if o >= 0 else pseudo[id(ep)]), scale=sd)
    standardize_fields(used, heal)
    for ep in used:
        ep.r_est = ep.r
    print(f"estimated-onset fields built ({time.time() - t0:.0f}s)", flush=True)

    def set_field(episodes, which):
        for e in episodes:
            e.r = getattr(e, which)

    def score(loc, subset, tag, onset_source, train_field):
        preds = loc.predict_many(subset, g, op)
        m = evaluate_localization(preds, subset, g, op, candidates=cands)
        return {**meta, "localizer": loc.name, "subset": tag, "onset_source": onset_source,
                "train_field_onset": train_field, "nu": args.nu, "n_episodes": len(subset),
                "n_candidates": len(cands or g.fail_nodes), "chance": 1 / len(cands or g.fail_nodes),
                **{k: m[k] for k in ("top1", "hard_top1", "hit@3", "mrr", "w1g", "n_hard", "n_fault")}}

    te_f = [e for e in sp["test"] if not e.is_healthy]
    gated = [e for e in te_f if alarmed[id(e)]]
    rows = []
    for factory in (lambda: FSILocalizer(epochs=args.fsi_epochs, seed=args.seed, device=dev),
                    LargestResidual, ConstantScore, AntiEnergy, RandomScore):
        # (a) train on true-onset fields, test on true-onset fields -- the published protocol
        set_field(eps, "r_true")
        loc = factory(); torch.manual_seed(args.seed); np.random.seed(args.seed)
        loc.fit(sp["train"], g, op)
        for subset, tag in ((te_f, "all_test_faults"), (gated, "cusum_alarmed")):
            if subset:
                rows.append(score(loc, subset, tag, "true_label", "true_label"))
        # (b) same fitted model, test fields rebuilt from the CUSUM onset estimate
        set_field(te_f, "r_est")
        for subset, tag in ((te_f, "all_test_faults"), (gated, "cusum_alarmed")):
            if subset:
                rows.append(score(loc, subset, tag, "cusum_est", "true_label"))
        # (c) train AND test on estimated-onset fields (no label onset anywhere at test time)
        set_field(eps, "r_est")
        loc2 = factory(); torch.manual_seed(args.seed); np.random.seed(args.seed)
        loc2.fit(sp["train"], g, op)
        for subset, tag in ((te_f, "all_test_faults"), (gated, "cusum_alarmed")):
            if subset:
                rows.append(score(loc2, subset, tag, "cusum_est", "cusum_est"))
        print(f"  {loc.name} done ({time.time() - t0:.0f}s)", flush=True)
    set_field(eps, "r_true")
    df = pd.DataFrame(rows)
    df.to_csv(out / f"onset_localization__{args.emb}__s{args.seed}.csv", index=False)
    print(df.to_string(index=False), flush=True)
    print(f"TOTAL {time.time() - t0:.0f}s -> {out}", flush=True)


main()
