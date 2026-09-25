"""Figure -- why the old primary metric was retired.

Runs every localizer ONCE on franka_v6 and scores it under both the retired CvV and the replacement hard top-1,
so the comparison is on identical predictions and identical episodes. CvV is vendored here (it has been deleted
from the library) purely so this figure can be produced; nothing else should import it.

Form: scatter, because the claim is about the RELATIONSHIP between two metrics -- specifically that a scorer can
sit at the top of one while sitting at chance on the other. A bar chart of either metric alone cannot show that."""
import os, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])  # repository root (code)
ROOT = os.environ.get("FSI_ROOT", REPO)  # data/results root, see docs/CONFIGURATION.md
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
sys.path.insert(0, REPO)
from fsi.core.graph import StructuralGraph
from fsi.core.types import Operator
from fsi.data import load_episodes, make_splits
from fsi.baselines import (LargestResidual, DirectSolve, GDN, TranAD, GraphSL, RCD, AERCA,
                           ConstantScore, AntiEnergy, RandomScore)
from fsi.models import FSILocalizer, RBCLocalizer
from fsi.metrics.localization import hard_top1, top1, _cand
from fsi.viz.paper import (use_paper_style, color_for, is_ref, pretty, save, COL, WIDE, INK, INK2,
                           FLOOR, ACCENT, OURS)
import dataclasses

OUT = Path(f"{ROOT}/results/paper/metric_validity")

def cvv_retired(preds, episodes, graph, operator, reach_tol=0.15):
    """The retired metric, verbatim from the pre-2026-07-30 fsi/metrics/localization.py."""
    Gn = np.abs(operator.G) / (np.abs(operator.G).max(0, keepdims=True) + 1e-12)
    fail = set(graph.fail_nodes)
    wins = 0.0; pairs = 0
    for pred, ep in zip(preds, episodes):
        if ep.is_healthy:
            continue
        e = ep.energy(); supp = set(ep.support())
        for s in ep.support():
            for w in fail:
                if w == s or w in supp:
                    continue
                if e[w] >= e[s] and Gn[w, s] > reach_tol:
                    pairs += 1
                    ss, sw = pred.scores[s], pred.scores[w]
                    wins += 1.0 if ss > sw else (0.5 if ss == sw else 0.0)
    return wins / pairs if pairs else float("nan"), pairs

def main(replot_only=True):
    csv = OUT / 'both_metrics.csv'
    if replot_only and csv.exists():
        return plot(pd.read_csv(csv))
    data = Path(f"{ROOT}/data/franka_v6")
    eps = load_episodes(data / "all")
    rng = np.random.default_rng(0)
    if len(eps) > 6000:
        eps = [eps[i] for i in rng.choice(len(eps), 6000, replace=False)]
    g = StructuralGraph.load(data / "graph.json"); op = Operator.load(data / "operator_G.npz")
    sp = make_splits(eps, seed=0)
    tr, te = sp["train"], sp["test"]
    series = lambda E: [dataclasses.replace(e, y=(e.y[..., 2] if e.y is not None else None),
                                            yhat=(e.yhat[..., 2] if getattr(e, "yhat", None) is not None
                                                  and np.ndim(e.yhat) == 3 else None)) for e in E]
    specs = [(LargestResidual, 0), (lambda: DirectSolve(lam=0.03), 0), (GraphSL, 0), (RBCLocalizer, 0),
             (lambda: FSILocalizer(epochs=80), 0), (ConstantScore, 0), (AntiEnergy, 0), (RandomScore, 0),
             (GDN, 1), (TranAD, 1), (RCD, 1), (AERCA, 1)]
    rows = []
    for make, use_series in specs:
        torch.manual_seed(0); np.random.seed(0)
        loc = make()
        TR, TE = (series(tr), series(te)) if use_series else (tr, te)
        try:
            loc.fit(TR, g, op)
            preds = loc.predict_many(TE, g, op)
            c, npairs = cvv_retired(preds, TE, g, op)
            h = hard_top1(preds, TE, g)
            t1 = float(np.mean([v for v in (top1(p, e, g) for p, e in zip(preds, TE)) if v is not None]))
            rows.append({"localizer": loc.name, "cvv_retired": c, "n_cvv_pairs": npairs,
                         "hard_top1": h["hard_top1"], "top1": t1, "chance": h["chance"], "n_hard": h["n_hard"]})
            print(f"  {loc.name:17s} cvv={c:.3f}  hard_top1={h['hard_top1']:.3f}  top1={t1:.3f}", flush=True)
        except Exception as e:
            print(f"  {loc.name:17s} FAILED {type(e).__name__}: {e}", flush=True)
    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "both_metrics.csv", index=False)
    return plot(d)

def plot(d):
    use_paper_style()
    fig, ax = plt.subplots(figsize=(COL * 1.55, 3.0))
    chance = float(d["chance"].iloc[0])
    ax.axhline(chance, color=FLOOR, linestyle=(0, (4, 3)), linewidth=1.0, zorder=1)
    ax.axvline(0.5, color=FLOOR, linestyle=(0, (1, 2)), linewidth=1.0, zorder=1)
    ax.annotate("chance", (0.02, chance), xytext=(0, 3), textcoords="offset points",
                fontsize=6.5, color=INK2, va="bottom")
    for _, r in d.iterrows():
        ref = is_ref(r["localizer"])
        ax.scatter(r["cvv_retired"], r["hard_top1"], s=54 if not ref else 46,
                   color=color_for(r["localizer"]), marker="D" if ref else "o",
                   edgecolor="white", linewidth=0.9, zorder=5)
    # Label only what carries the argument. The middle of this plot is a dense cluster of near-tied methods;
    # labelling all twelve would be unreadable and none of them is the point.
    KEY = {"zz_anti_energy": (-10, 16, "right"), "zz_constant": (-9, -19, "right"),
           "largest_residual": (9, 3, "left"), "fsi": (-9, 7, "right"), "tranad": (9, 1, "left")}
    for _, r in d.iterrows():
        if r["localizer"] not in KEY:
            continue
        dx, dy, ha = KEY[r["localizer"]]
        ax.annotate(pretty(r["localizer"]), (r["cvv_retired"], r["hard_top1"]),
                    textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=6.8,
                    color=INK if r["localizer"] == "fsi" else INK2,
                    fontweight="bold" if r["localizer"] == "zz_anti_energy" else "normal",
                    arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.5,
                                    shrinkA=0, shrinkB=3) if r["localizer"] == "zz_anti_energy" else None)
    ae = d.set_index("localizer").loc["zz_anti_energy"]
    ax.annotate(f"ranks nodes by NEGATIVE residual energy:\ntop of CvV, chance on hard top-1 "
                f"(top-1 {ae['top1']:.3f})",
                (0.985, ae["hard_top1"]), textcoords="offset points", xytext=(-10, -20),
                ha="right", va="top", fontsize=6.3, color=INK2)
    ax.set_xlabel("CvV  (retired metric)")
    ax.set_ylabel("hard top-1  (replacement)")
    ax.set_xlim(-0.05, 1.08); ax.set_ylim(0, max(0.34, d["hard_top1"].max() * 1.25))
    ax.set_axisbelow(True); ax.grid(color="#e2e1dc", linewidth=0.6)
    ax.set_title("A scorer with no information about the source\ntops the retired metric", loc="left", pad=6)
    print("wrote", save(fig, f"{ROOT}/figures/fig_metric_validity"))

main()
