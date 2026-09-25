"""Figure -- component ablation, as a signed-delta (tornado) chart.

Form: signed horizontal bars from a zero line, not grouped bars of absolute values. The claim is about the SIZE
AND SIGN of each component's contribution, and on Franka about the absence of one; absolute-value bars would make
Franka's 0.18-0.30 range look like real variation when it is inside build-seed noise.

The shaded band on each panel is the headroom -- FULL minus the no-information floor. On ANYmal it is 0.60 wide
and the bars live inside it. On Franka it is 0.053 wide and the bars spill past it, which is the honest visual
statement that Franka has no headroom for any component to buy."""
import os, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])  # repository root (code)
ROOT = os.environ.get("FSI_ROOT", REPO)  # data/results root, see docs/CONFIGURATION.md
import sys
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, REPO)
from fsi.viz.paper import use_paper_style, save, WIDE, INK, INK2, FLOOR, OURS, ACCENT

SRC = Path(f"{ROOT}/results/paper/ablation")
LOCAL = Path(f"{ROOT}/results/paper/ablation")
ARM = {"no_node_emb": "learned node embedding", "no_masking": "node masking (DOS)",
       "no_attention": "all-node attention", "no_denorm": "de-normalized field",
       "no_multi_horizon": "multi-horizon heads", "no_onset_windows": "onset-relative windows",
       "no_mag_w": "mag_w = 50"}

def main():
    p = (SRC if (SRC / "ablation_deltas.csv").exists() else LOCAL) / "ablation_deltas.csv"
    d = pd.read_csv(p)
    d = d[(d.localizer == "fsi") & (d.metric == "hard_top1")]
    full = pd.read_csv(p.parent / "ablation.csv")
    use_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.9))
    order = (d[d.embodiment == "anymal_v4"].set_index("arm")["delta_vs_full"].sort_values().index.tolist())
    for ax, emb, title in zip(axes, ["anymal_v4", "franka_v4"], ["ANYmal (gap 0.89)", "Franka (gap 0.39)"]):
        sub = d[d.embodiment == emb].set_index("arm").reindex(order).dropna(subset=["delta_vs_full"])
        f = full[(full.embodiment == emb) & (full.localizer == "fsi") & (full.metric == "hard_top1") &
                 (full.arm == "full")]["value"]
        fl = full[(full.embodiment == emb) & (full.localizer == "zz_constant") & (full.metric == "hard_top1") &
                  (full.arm == "full")]["value"]
        head = (float(f.iloc[0]) - float(fl.iloc[0])) if len(f) and len(fl) else np.nan
        y = np.arange(len(sub))
        # headroom band: how much there is to lose before hitting the no-information floor
        if np.isfinite(head):
            ax.axvspan(-head, 0, color=FLOOR, alpha=0.22, zorder=0, lw=0)
            ax.annotate(f"headroom above floor {head:.3f}", (-head / 2, -0.95),
                        ha="center", va="center", fontsize=6.3, color=INK2, annotation_clip=False)
        cols = [ACCENT if v > 0 else OURS for v in sub["delta_vs_full"]]
        ax.barh(y, sub["delta_vs_full"], height=0.62, color=cols, edgecolor="white", linewidth=0.7, zorder=3)
        ax.errorbar(sub["delta_vs_full"], y, xerr=sub["se_delta"], fmt="none", ecolor=INK2,
                    elinewidth=0.7, capsize=1.6, zorder=4)
        ax.axvline(0, color=INK2, linewidth=0.8, zorder=2)
        for v, e, z, yy in zip(sub["delta_vs_full"], sub["se_delta"], sub["z"], y):
            off = 4 if v >= 0 else -4
            ax.annotate(f"{v:+.3f}" + ("" if abs(z) >= 2 else "  n.s."),
                        (v + np.sign(v) * e, yy), textcoords="offset points", xytext=(off, 0),
                        ha="left" if v >= 0 else "right", va="center", fontsize=6.3,
                        color=INK2 if abs(z) >= 2 else FLOOR)
        ax.set_yticks(y); ax.set_yticklabels([ARM.get(a, a) for a in sub.index])
        ax.set_xlabel("$\\Delta$ FSI hard top-1 when removed")
        ax.set_title(title, loc="left", pad=6)
        ax.set_axisbelow(True); ax.grid(axis="x", color="#e2e1dc", linewidth=0.6)
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
        lim = max(0.08, float(np.nanmax(np.abs(sub["delta_vs_full"]))))
        ax.set_xlim(-lim * 1.55, lim * 0.85)
        ax.set_ylim(-1.4, len(sub) - 0.35)
    fig.tight_layout(w_pad=2.6)
    print("wrote", save(fig, f"{ROOT}/figures/fig_ablation"))

    # Companion: forward-model accuracy against localization, one point per arm. The negative control is the
    # whole message -- removing masking gives the BEST healthy predictor and a collapsed localizer.
    a = pd.read_csv(p.parent / "ablation.csv")
    piv = a.pivot_table(index=["embodiment", "arm"], columns="metric", values="value").reset_index()
    fig, ax = plt.subplots(figsize=(WIDE * 0.5, 2.7))
    for emb, mk, lab in (("anymal_v4", "o", "ANYmal"), ("franka_v4", "s", "Franka")):
        s = piv[piv.embodiment == emb]
        if "forward_nmse_heldout_healthy" not in s or s.empty:
            continue
        fsi_h = a[(a.embodiment == emb) & (a.localizer == "fsi") & (a.metric == "hard_top1")].set_index("arm")["value"]
        x = s.set_index("arm")["forward_nmse_heldout_healthy"]
        common = x.index.intersection(fsi_h.index)
        ax.scatter(x[common], fsi_h[common], marker=mk, s=42,
                   color=[ACCENT if aa == "no_masking" else OURS for aa in common],
                   edgecolor="white", linewidth=0.8, label=lab, zorder=4)
        if "no_masking" in common:
            ax.annotate("masking removed:\nbest predictor, worst localizer",
                        (x["no_masking"], fsi_h["no_masking"]), textcoords="offset points",
                        xytext=(8, 6), fontsize=6.3, color=INK2,
                        arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.5))
    ax.set_xlabel("held-out healthy forward nMSE  (lower = better predictor)")
    ax.set_ylabel("FSI hard top-1")
    ax.set_axisbelow(True); ax.grid(color="#e2e1dc", linewidth=0.6)
    ax.legend(loc="lower left")
    ax.set_title("A better healthy model is not a better localizer", loc="left", pad=6)
    fig.tight_layout()
    print("wrote", save(fig, f"{ROOT}/figures/fig_forward_vs_localize"))

main()
