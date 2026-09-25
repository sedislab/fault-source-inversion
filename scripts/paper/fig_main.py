"""Figure 1 -- main localization results on both real embodiments.

Form: horizontal bars, sorted by the metric, one panel per embodiment. Bars are right for ranked magnitude across
a labelled category axis, and horizontal keeps the twelve method names readable without rotation. Colour encodes
ROLE (ours / baseline / no-information floor), not rank, because these are not twelve peer categories."""
import os, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])  # repository root (code)
ROOT = os.environ.get("FSI_ROOT", REPO)  # data/results root, see docs/CONFIGURATION.md
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, REPO)
from fsi.viz.paper import (use_paper_style, color_for, is_ref, pretty, save, ygrid, COL, WIDE,
                           INK, INK2, FLOOR, OURS)

R = Path(f"{ROOT}/results")
XLABEL = {"hard_top1": "hard top-1", "top1": "top-1 (all fault episodes)"}

PANELS = [("Franka (7 joints)", R / "franka_v6/franka_v6_in_distribution.csv", 0.143, 0.390),
          ("ANYmal (12 joints)", R / "anymal_v4/anymal_v4_in_distribution.csv", 0.083, 0.885)]

def panel(ax, csv, chance, gap, title, metric="hard_top1"):
    d = pd.read_csv(csv).dropna(subset=[metric]).sort_values(metric)
    floor = float(d.set_index("localizer")[metric].get("zz_constant", float("nan")))
    y = range(len(d))
    bars = ax.barh(list(y), d[metric], height=0.68,
                   color=[color_for(n) for n in d["localizer"]],
                   hatch=["///" if is_ref(n) else "" for n in d["localizer"]],
                   edgecolor="white", linewidth=0.8, zorder=3)
    sd = d.get(f"{metric}_sd")
    if sd is not None:
        ax.errorbar(d[metric], list(y), xerr=sd.fillna(0), fmt="none", ecolor=INK2,
                    elinewidth=0.7, capsize=1.6, zorder=4)
    ax.set_yticks(list(y)); ax.set_yticklabels([pretty(n) for n in d["localizer"]])
    for lbl, n in zip(ax.get_yticklabels(), d["localizer"]):
        if n in ("fsi", "fsi_rbc"):
            lbl.set_color(INK); lbl.set_fontweight("bold")
        elif is_ref(n):
            lbl.set_color(INK2)
    # Value labels clear the error-bar whisker, not the bar end, or they collide with it.
    err = sd.fillna(0) if sd is not None else [0] * len(d)
    for v, e, yy in zip(d[metric], err, y):
        ax.annotate(f"{v:.3f}", (v + e, yy), textcoords="offset points", xytext=(4, 0),
                    ha="left", va="center", fontsize=6.3, color=INK2)
    ax.axvline(floor, color=FLOOR, linestyle=(0, (4, 3)), linewidth=1.0, zorder=2)
    ax.annotate("floor", (floor, -0.9), textcoords="offset points", xytext=(0, 0),
                ha="center", va="top", fontsize=6.3, color=INK2, annotation_clip=False)
    ax.set_xlim(0, max(0.32, float(d[metric].max()) * 1.30))
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.set_xlabel(XLABEL[metric])
    ax.set_title(f"{title}\nchance {chance:.3f} · cause–victim gap {gap:.2f}", loc="left", pad=6)
    ax.set_axisbelow(True); ax.grid(axis="x", color="#e2e1dc", linewidth=0.6)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)

def main():
    use_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 3.3))
    for ax, (title, csv, chance, gap) in zip(axes, PANELS):
        if not csv.exists():
            ax.text(0.5, 0.5, f"missing\n{csv.name}", ha="center", va="center"); continue
        panel(ax, csv, chance, gap, title)
    fig.suptitle("")
    fig.tight_layout(w_pad=3.0)
    print("wrote", save(fig, f"{ROOT}/figures/fig1_main_results"))

    # Companion: the same methods on plain top-1, which tells a different and more flattering story. Showing both
    # is the point -- aggregate top-1 hides the cause/victim failure that hard top-1 exposes.
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 3.3))
    for ax, (title, csv, chance, gap) in zip(axes, PANELS):
        if not csv.exists():
            continue
        panel(ax, csv, chance, gap, title, metric="top1")
    fig.tight_layout(w_pad=2.4)
    print("wrote", save(fig, f"{ROOT}/figures/fig1b_top1"))

main()
