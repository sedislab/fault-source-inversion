"""Component ablation (leave-one-out) for FSI.

Derived from scripts/build_embodiment.py (stage B) + scripts/eval_embodiment.py (localization only).
ONE job = ONE (embodiment, arm): rebuild the residual-field dataset + operator under that arm's config,
then evaluate fsi / fsi_rbc / largest_residual / zz_constant over >=3 localizer seeds.

Arms (leave-one-out from FULL = arch=attn, node_emb on, horizons=(1,4,8), de-normalized field,
onset-relative windows, mag_w=50, masked=True):
  full             -- everything on
  no_attention     -- arch='graph' (old 1-hop neighbour aggregation)
  no_node_emb      -- arch='attn' but n_nodes=None (learned node embedding absent)
  no_multi_horizon -- horizons=(1,)
  no_denorm        -- residual_field(scale=None): field stays z-scored
  no_onset_windows -- old fixed-tail window (post=resid[-window:], pre=resid[onset-window:onset])
  no_masking       -- masked=False (persistence-predictor negative control)
  no_mag_w         -- mag_w=1.0 in the inverse; DATASET IDENTICAL TO full, so this arm is run as a second
                      eval inside the `full` job rather than as its own build.

Everything not named by the arm is held fixed, including the healthy pseudo-onset, the matched-pair sweep
operator, standardize_fields, the 80/20 healthy split used for the forward nMSE, and the eval protocol.
"""
import os, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])  # repository root (code)
ROOT = os.environ.get("FSI_ROOT", REPO)  # data/results root, see docs/CONFIGURATION.md
import sys, json, glob, os, argparse, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, REPO)
from fsi.graph import franka_joint_graph, anymal_sim_graph
from fsi.core.types import Episode, FaultLabel, Operator, Onset
from fsi.models import ForwardModel, train_forward, residual_field, standardize_fields, normalize_signals
from fsi.models.forward import _chan, _reduce, _windows, norm_adj_noself, operator_channels
from fsi.models.inverse import norm_adj
from fsi.eval.stats import characterize
from fsi.data import make_splits
from fsi.baselines import LargestResidual, ConstantScore, AntiEnergy, RandomScore
from fsi.models import FSILocalizer, RBCLocalizer
from fsi.eval import run_comparison

GRAPHS = {"franka_joint": franka_joint_graph, "anymal_sim": anymal_sim_graph}

# arm -> which single component is switched OFF
ARMS = {
    "full":             dict(),
    "no_attention":     dict(arch="graph"),
    "no_node_emb":      dict(node_emb=False),
    "no_multi_horizon": dict(horizons=(1,)),
    "no_denorm":        dict(denorm=False),
    "no_onset_windows": dict(onset_windows=False),
    "no_masking":       dict(masked=False),
    "no_mag_w":         dict(mag_w=1.0),          # eval-only arm, dataset == full
}

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True)
ap.add_argument("--out", required=True, help="scratch dataset dir for this arm")
ap.add_argument("--csv_dir", required=True, help="where the tidy per-arm csv goes")
ap.add_argument("--graph", required=True, choices=list(GRAPHS))
ap.add_argument("--embodiment", required=True, help="label used in the csv (franka_v4 / anymal_v4)")
ap.add_argument("--arm", required=True, choices=list(ARMS))
ap.add_argument("--channels", type=int, default=3)
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--window", type=int, default=60)
ap.add_argument("--build_seed", type=int, default=0)
ap.add_argument("--eval_seeds", type=int, default=3)
ap.add_argument("--cap", type=int, default=6000)
ap.add_argument("--fsi_epochs", type=int, default=80)
ap.add_argument("--also_eval_no_mag_w", action="store_true",
                help="only meaningful on --arm full: run a second eval with mag_w=1.0 on the SAME dataset")
ap.add_argument("--save_dataset", action="store_true")
args = ap.parse_args()

CFG = ARMS[args.arm]
BASE_HORIZONS = (1, 4, 8)
HORIZONS = tuple(CFG.get("horizons", BASE_HORIZONS))
ARCH = CFG.get("arch", "attn")
NODE_EMB = CFG.get("node_emb", True)
DENORM = CFG.get("denorm", True)
ONSET_WINDOWS = CFG.get("onset_windows", True)
MASKED = CFG.get("masked", True)
MAG_W = CFG.get("mag_w", 50.0)


# ---------------------------------------------------------------- legacy residual field (-onset_windows arm)
@torch.no_grad()
def residual_field_legacy(model, episode, graph, window=48, device="cpu", onset=None, scale=None):
    """Byte-for-byte the current residual_field EXCEPT the window rule, which reverts to the old fixed tail:
        post = resid[-window:]          (always the end of the episode, regardless of onset)
        pre  = resid[max(0,onset-window):onset]
    With randomized onsets those two windows overlap, so the self-reference subtracts away part of the fault
    itself and the "post" window carries pre-onset samples. Everything else (de-normalization, pseudo-onset,
    horizon alignment, _reduce) is unchanged, so this isolates the windowing change only."""
    model.eval()
    A_hat = torch.tensor(norm_adj(graph.adjacency()), device=device)
    A_nosl = torch.tensor(norm_adj_noself(graph.adjacency()), device=device)
    nf = torch.tensor(graph.node_features(), device=device)
    y, u, H = _chan(episode.y), episode.u, model.hist
    hz, nch, T0 = model.horizons, y.shape[-1], len(y)
    X = torch.tensor(np.stack([y[t - H:t].transpose(1, 0, 2) for t in range(H, T0)]).astype(np.float32), device=device)
    U = torch.tensor(u[H:].astype(np.float32), device=device)
    pred = model(X, U, A_hat, nf, A_nosl).cpu().numpy()
    pred = pred.reshape(len(pred), y.shape[1], len(hz), nch)
    res = [np.zeros_like(y) for _ in hz]
    for j, h in enumerate(hz):
        tau = np.arange(H + h - 1, T0)
        res[j][tau] = y[tau] - pred[tau - (H + h - 1), :, j, :]
    resid = np.concatenate(res, -1)[H:]
    episode.yhat = (y - res[0]).astype(np.float32)
    if scale is not None:
        resid = resid * np.tile(np.asarray(scale, np.float32), len(hz))
    C, T = resid.shape[-1], len(resid)
    onset = int(episode.meta.get("onset_step", 0) if onset is None else onset) - H
    post = resid[-window:]                                    # OLD RULE: fixed tail
    pre = resid[max(0, onset - window):onset] if onset > 0 else None
    feats = _reduce(np.sqrt((post ** 2).sum(-1)), post, C)
    if pre is not None and len(pre) >= 5:
        feats = feats - _reduce(np.sqrt((pre ** 2).sum(-1)), pre, C)
    return feats.astype(np.float32)


# ---------------------------------------------------------------- data loading (identical to build_embodiment)
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


@torch.no_grad()
def forward_nmse(model, episodes, graph, device, batch=512):
    """nMSE on held-out healthy episodes = MSE(pred, target) / MSE(target.mean(), target), on the z-scored
    signals the model is trained against. 1.0 = no better than predicting the healthy mean."""
    model.eval()
    A_hat = torch.tensor(norm_adj(graph.adjacency()), device=device)
    A_nosl = torch.tensor(norm_adj_noself(graph.adjacency()), device=device)
    nf = torch.tensor(graph.node_features(), device=device)
    X, U, Y = _windows(episodes, model.hist, model.horizons)
    X = torch.tensor(X, device=device); U = torch.tensor(U, device=device)
    Y = torch.tensor(Y, device=device).reshape(len(Y), Y.shape[1], -1)
    se, n = 0.0, 0
    for i in range(0, len(X), batch):
        p = model(X[i:i + batch], U[i:i + batch], A_hat, nf, A_nosl)
        se += float(((p - Y[i:i + batch]) ** 2).sum()); n += Y[i:i + batch].numel()
    mse = se / n
    var = float(((Y - Y.mean(0, keepdim=True)) ** 2).mean())
    return mse / max(var, 1e-12), mse


# ---------------------------------------------------------------- evaluation
def evaluate(eps, g, op, mag_w, tag):
    rng = np.random.default_rng(0)
    ev = eps
    if len(ev) > args.cap:
        ev = [ev[i] for i in rng.choice(len(ev), args.cap, replace=False)]
    sp = make_splits(ev, seed=0)
    realizable = sorted({s for e in sp["train"] + sp["cal"] for s in e.support()})
    cands = [v for v in g.fail_nodes if v in set(realizable)]
    cands = cands if 0 < len(cands) < len(g.fail_nodes) else None
    n_cand = len(cands) if cands else len(g.fail_nodes)
    print(f"[{tag}] candidates {n_cand}/{len(g.fail_nodes)} chance {1/n_cand:.3f} "
          f"train {len(sp['train'])} test {len(sp['test'])}", flush=True)
    rl = [LargestResidual, RBCLocalizer, (lambda: FSILocalizer(epochs=args.fsi_epochs, mag_w=mag_w)),
          ConstantScore, AntiEnergy, RandomScore]
    df = run_comparison(rl, sp["train"], sp["test"], g, op,
                        seeds=tuple(range(args.eval_seeds)), candidates=cands)
    return df, n_cand


def main():
    t_start = time.time()
    Path(args.csv_dir).mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.build_seed); np.random.seed(args.build_seed)
    g = GRAPHS[args.graph]()
    heal = load_group("*_healthy_*.npz", g, args.raw)
    fault = load_group("*_fault_*.npz", g, args.raw)
    sweep = load_group("*_sweep_*.npz", g, args.raw)
    sweepref = load_group("*_sweepref_*.npz", g, args.raw)
    print(f"ARM {args.arm} | arch={ARCH} node_emb={NODE_EMB} horizons={HORIZONS} denorm={DENORM} "
          f"onset_windows={ONSET_WINDOWS} masked={MASKED} mag_w={MAG_W}", flush=True)
    print(f"healthy {len(heal)} | fault {len(fault)} | sweep {len(sweep)}/{len(sweepref)} | {dev}", flush=True)

    _, sd = normalize_signals(heal + fault + sweep + sweepref, heal)

    # 80/20 healthy split so the forward model has a genuine held-out nMSE. Fixed seed, identical across arms.
    # NOTE this differs from scripts/build_embodiment.py, which trains f_theta on ALL healthy episodes.
    hsplit = np.random.default_rng(1234).permutation(len(heal))
    n_ho = max(1, int(0.2 * len(heal)))
    ho_idx, tr_idx = set(hsplit[:n_ho].tolist()), hsplit[n_ho:].tolist()
    heal_tr = [heal[i] for i in tr_idx]
    heal_ho = [heal[i] for i in sorted(ho_idx)]

    fm = ForwardModel(g.node_features().shape[1], du=heal[0].u.shape[1], hist=8, hidden=96, layers=3,
                      channels=args.channels, masked=MASKED, arch=ARCH,
                      n_nodes=(g.n if NODE_EMB else None), horizons=HORIZONS)
    train_forward(fm, heal_tr, g, epochs=args.epochs, device=dev, verbose=True)
    nmse, raw_mse = forward_nmse(fm, heal_ho, g, dev)
    print(f"held-out healthy forward nMSE {nmse:.4f} (mse {raw_mse:.5f}, {len(heal_ho)} episodes)", flush=True)

    onsets = np.array([e.faults[0].onset_step for e in fault]) if fault else np.array([60])
    rng = np.random.default_rng(0)
    field_fn = residual_field if ONSET_WINDOWS else residual_field_legacy
    scale = sd if DENORM else None
    for ep in heal + fault + sweep + sweepref:
        on = None if ep.faults else int(rng.choice(onsets))
        ep.r = field_fn(fm, ep, g, window=args.window, device=dev, onset=on, scale=scale)
    standardize_fields(heal + fault + sweep + sweepref, heal)

    C = operator_channels(heal[0].r.shape[1])
    G = np.zeros((g.n, g.n), np.float32); Gm = np.zeros((C, g.n, g.n), np.float32)
    matched = bool(sweep) and len(sweepref) == len(sweep)
    src = ([(ep, np.clip(ep.r - ref.r, 0, None)) for ep, ref in zip(sweep, sweepref)] if matched
           else [(ep, ep.r) for ep in (sweep or fault)])
    print(f"operator from {'MATCHED-PAIR SWEEP' if matched else 'unmatched'} ({len(src)} eps)", flush=True)
    for j in range(g.n):
        at_j = [(ep, d) for ep, d in src if ep.support() == [j] and ep.faults[0].fault_type == "torque_loss"]
        if not at_j:
            continue
        sev = np.array([max(ep.faults[0].magnitude, 1e-3) for ep, _ in at_j])[:, None]
        G[:, j] = np.mean(np.stack([d[:, 0] for _, d in at_j]) / sev, 0)
        for c in range(C):
            Gm[c, :, j] = np.mean(np.stack([d[:, 1 + c] for _, d in at_j]) / sev, 0)
    op = Operator(args.graph, G, g.names(), {"matched": matched, "channels": C}, G_multi=Gm)

    eps = heal + fault
    st = characterize(eps, g, op)
    print("OVERALL:", {k: st[k] for k in ("n_episodes", "cause_victim_gap", "mean_propagation_hops",
                                          "coherence", "mean_coherence")}, flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    op.save(out / "operator_G.npz"); g.save(out / "graph.json")
    if args.save_dataset:
        from fsi.data import save_episodes
        save_episodes(eps, out / "all")

    import pandas as pd
    evals = [(args.arm, MAG_W)]
    if args.also_eval_no_mag_w and args.arm == "full":
        evals.append(("no_mag_w", 1.0))
    for arm_name, mw in evals:
        rows = []
        df, n_cand = evaluate(eps, g, op, mw, arm_name)
        df.to_csv(Path(args.csv_dir) / f"table_{args.embodiment}_{arm_name}_b{args.build_seed}.csv", index=False)
        for _, r in df.iterrows():
            for metric in ("top1", "hard_top1"):
                rows.append(dict(embodiment=args.embodiment, arm=arm_name, build_seed=args.build_seed,
                                 localizer=r["localizer"], metric=metric, value=r.get(metric),
                                 sd=r.get(f"{metric}_sd"), n_seeds=int(r.get("n_seeds", 0)),
                                 n_hard=r.get("n_hard"), n_fault=r.get("n_fault"), n_candidates=n_cand,
                                 chance=1.0 / n_cand, error=r.get("error", "")))
        # dataset-level scalars, carried on the same tidy frame with localizer='__dataset__'
        for metric, value in (("forward_nmse_heldout_healthy", nmse),
                              ("forward_mse_heldout_healthy", raw_mse),
                              ("operator_max_coherence", st["coherence"]),
                              ("operator_mean_coherence", st["mean_coherence"]),
                              ("cause_victim_gap", st["cause_victim_gap"]),
                              ("mean_propagation_hops", st["mean_propagation_hops"])):
            rows.append(dict(embodiment=args.embodiment, arm=arm_name, build_seed=args.build_seed,
                             localizer="__dataset__", metric=metric, value=float(value),
                             sd=float("nan"), n_seeds=1, n_hard=float("nan"), n_fault=st["n_fault"],
                             n_candidates=n_cand, chance=1.0 / n_cand, error=""))
        # written per-eval so a failure in a later eval cannot lose an earlier one
        pd.DataFrame(rows).to_csv(
            Path(args.csv_dir) / f"{args.embodiment}_{arm_name}_b{args.build_seed}.csv", index=False)
        print(f"WROTE {args.embodiment}/{arm_name}", flush=True)
    print(f"DONE {args.embodiment}/{args.arm} in {(time.time()-t_start)/60:.1f} min", flush=True)


main()
