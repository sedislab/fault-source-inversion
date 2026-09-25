"""Remaining paper figures: multi-fault, cross-embodiment transfer, severity/MDF, identifiability."""
import os, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])  # repository root (code)
ROOT = os.environ.get("FSI_ROOT", REPO)  # data/results root, see docs/CONFIGURATION.md
import sys
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, REPO)
from fsi.viz.paper import (use_paper_style, save, pretty, color_for, COL, WIDE, INK, INK2,
                           FLOOR, OURS, OURS_ALT, ACCENT, BASELINE)

P = Path(f"{ROOT}/results/paper")
F = f"{ROOT}/figures"
EMB = ["franka", "anymal", "vehicle", "crazyflie"]

def fig_multifault():
    """Line chart in k. The floor RISES with k, so it is drawn as a filled area rather than a line -- raw
    support-F1 in k is misleading without it."""
    d = pd.read_csv(P / "synthetic/multifault.csv")
    use_paper_style()
    fig, axes = plt.subplots(1, 4, figsize=(WIDE, 2.2), sharey=True)
    for ax, emb in zip(axes, EMB):
        s = d[d.embodiment == emb]
        ks = sorted(s.k.unique())
        fl = [s[(s.k == k)]["zz_floor_support_f1"].max() for k in ks]
        ax.fill_between(ks, 0, fl, color=FLOOR, alpha=0.30, lw=0, zorder=1)
        for loc, col in (("fsi", OURS), ("fsi_rbc", OURS_ALT)):
            t = s[s.localizer == loc].sort_values("k")
            if t.empty:
                continue
            ax.plot(t.k, t.support_f1, "-o", color=col, zorder=4,
                    label=pretty(loc) if emb == EMB[0] else None)
            ax.fill_between(t.k, t.support_f1 - t.support_f1_sd.fillna(0),
                            t.support_f1 + t.support_f1_sd.fillna(0), color=col, alpha=0.18, lw=0, zorder=2)
        ax.set_xticks(ks); ax.set_xlabel("simultaneous faults $k$")
        ax.set_title(emb, loc="left", pad=4)
        ax.set_axisbelow(True); ax.grid(axis="y", color="#e2e1dc", linewidth=0.6)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("support F1")
    axes[0].annotate("no-information floor\n(rises with $k$)", (1.02, 0.10), fontsize=6.2, color=INK2, va="bottom")
    fig.tight_layout(w_pad=1.2, rect=(0, 0.06, 1, 1))
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=2)
    print("wrote", save(fig, f"{F}/fig_multifault"))

def fig_transfer():
    d = pd.read_csv(P / "synthetic/transfer.csv")
    d = d[d.condition.isin(["transfer", "target_only", "pooled"])]
    use_paper_style()
    fig, ax = plt.subplots(figsize=(WIDE * 0.62, 2.5))
    conds = ["target_only", "transfer", "pooled"]
    cols = {"target_only": OURS, "transfer": ACCENT, "pooled": BASELINE}
    w, x = 0.26, np.arange(len(EMB))
    for i, c in enumerate(conds):
        v = [d[(d.embodiment == e) & (d.condition == c)]["hard_top1"].mean() for e in EMB]
        sd = [d[(d.embodiment == e) & (d.condition == c)]["hard_top1_sd"].mean() for e in EMB]
        ax.bar(x + (i - 1) * w, v, w, yerr=sd, color=cols[c], label=c.replace("_", " "),
               edgecolor="white", linewidth=0.7, error_kw=dict(ecolor=INK2, elinewidth=0.7, capsize=1.6), zorder=3)
    for j, e in enumerate(EMB):
        fl = d[d.embodiment == e]["zz_floor_hard_top1"].max()
        ax.plot([j - 1.6 * w, j + 1.6 * w], [fl, fl], color=FLOOR, ls=(0, (4, 3)), lw=1.1, zorder=5)
    # The one negative result in this figure must be visible, not buried in a caption.
    cz = EMB.index("crazyflie")
    ax.annotate("below floor", (cz, d[(d.embodiment == "crazyflie") & (d.condition == "transfer")]["hard_top1"].mean()),
                textcoords="offset points", xytext=(0, 16), ha="center", fontsize=6.3, color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT, linewidth=0.7))
    ax.set_xticks(x); ax.set_xticklabels(EMB)
    ax.set_ylabel("FSI hard top-1"); ax.set_xlabel("held-out target embodiment")
    ax.set_axisbelow(True); ax.grid(axis="y", color="#e2e1dc", linewidth=0.6)
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.16))
    ax.set_title("dashed line = no-information floor", loc="right", pad=6, fontsize=6.5, color=INK2)
    fig.tight_layout()
    print("wrote", save(fig, f"{F}/fig_transfer"))

def fig_severity():
    """Two axes side by side. Commanded severity is what the experimenter set; realized effect ratio is what the
    fault actually did. They are nearly uncorrelated (R^2 < 0.02), and only the realized axis is monotone."""
    d = pd.read_csv(P / "severity/severity.csv")
    use_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(WIDE * 0.75, 2.5), sharey=True)
    for ax, emb in zip(axes, ["anymal", "franka"]):
        for axis, col, mk in (("commanded_severity", BASELINE, "s"), ("effect_ratio", OURS, "o")):
            s = d[(d.embodiment == emb) & (d.axis == axis) & (d.localizer == "fsi") &
                  (d.metric == "hard_top1")].sort_values("bin_index")
            if s.empty:
                continue
            xx = np.arange(len(s))
            ax.errorbar(xx, s.value, yerr=s.value_sd_across_splits.fillna(0), fmt=f"-{mk}", color=col,
                        elinewidth=0.7, capsize=1.6, label=axis.replace("_", " "), zorder=4)
        ch = d[(d.embodiment == emb) & (d.localizer == "fsi")]["chance"].dropna()
        if len(ch):
            ax.axhline(float(ch.iloc[0]), color=FLOOR, ls=(0, (4, 3)), lw=1.0)
            ax.annotate("chance", (0.02, float(ch.iloc[0])), xytext=(0, 3), textcoords="offset points",
                        fontsize=6.3, color=INK2)
        ax.set_xlabel("severity sextile  (low $\\rightarrow$ high)")
        ax.set_title(emb, loc="left", pad=4)
        ax.set_axisbelow(True); ax.grid(axis="y", color="#e2e1dc", linewidth=0.6)
    axes[0].set_ylabel("FSI hard top-1"); axes[0].legend(loc="upper left")
    fig.tight_layout(w_pad=1.6)
    print("wrote", save(fig, f"{F}/fig_severity"))

def fig_identifiability():
    """The theory-validation figure, reporting a NEGATIVE result. Scatter, because the claim is about a
    relationship; a bar of correlation coefficients alone would hide that the sign is inverted."""
    pr = pd.read_csv(P / "identifiability/pairs.csv")
    co = pd.read_csv(P / "identifiability/correlation.csv")
    use_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(WIDE * 0.72, 2.6))
    for ax, emb, lab in zip(axes, ["franka_v6", "anymal_v4"], ["Franka", "ANYmal"]):
        s = pr[(pr.embodiment == emb) & (pr.localizer == "fsi") & (pr.episode_subset == "all_fault")]
        ax.scatter(s.coherence, s.confusion_rate, s=22, color=OURS, alpha=0.75,
                   edgecolor="white", linewidth=0.5, zorder=4)
        r = co[(co.embodiment == emb) & (co.localizer == "fsi") & (co.predictor == "coherence_G") &
               (co.episode_subset == "all_fault") & (co.target == "confusion_rate")]
        rh = co[(co.embodiment == emb) & (co.localizer == "fsi") & (co.predictor == "hop_distance") &
                (co.episode_subset == "all_fault") & (co.target == "confusion_rate")]
        if len(r):
            ax.set_title(f"{lab}\n" + r"coherence: $\rho$=" + f"{r.spearman_rho.iloc[0]:+.2f} "
                         f"(p={r.p_perm.iloc[0]:.3f}),  hop: " + r"$\rho$=" +
                         f"{rh.spearman_rho.iloc[0]:+.2f} (p={rh.p_perm.iloc[0]:.2f})",
                         loc="left", pad=6, fontsize=7.2)
        ax.set_xlabel("operator column coherence  |cos($g_i$, $g_j$)|")
        ax.set_axisbelow(True); ax.grid(color="#e2e1dc", linewidth=0.6)
    axes[0].set_ylabel("empirical confusion rate")
    axes[0].annotate("theory predicts a POSITIVE slope;\nFranka is significantly negative",
                     (0.03, 0.93), xycoords="axes fraction", fontsize=6.4, color=ACCENT, va="top")
    fig.tight_layout(w_pad=1.8)
    print("wrote", save(fig, f"{F}/fig_identifiability"))

for fn in (fig_multifault, fig_transfer, fig_severity, fig_identifiability):
    try:
        fn()
    except Exception as e:
        print(f"!! {fn.__name__} FAILED {type(e).__name__}: {e}")
