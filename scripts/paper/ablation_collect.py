"""Collate the per-(embodiment, arm, build_seed) csvs written by ablation_run.py.

Writes
  results/paper/ablation/ablation.csv                -- one row per (embodiment, arm, localizer, metric)
  results/paper/ablation/ablation_by_build_seed.csv  -- the same, split out per dataset-build seed
and prints the leave-one-out delta tables the README quotes.

Aggregation: each build seed rebuilds f_theta + the residual field + the operator from scratch and then runs
each localizer over `n_eval_seeds` localizer seeds. `value` is the grand mean over build seeds; `sd` is the
spread ACROSS BUILD SEEDS of the per-build means, i.e. the error bar that includes dataset-build variance --
the term that dominates here and that a localizer-seed-only sd hides.
"""
import os, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])  # repository root (code)
ROOT = os.environ.get("FSI_ROOT", REPO)  # data/results root, see docs/CONFIGURATION.md
import sys, glob
from pathlib import Path
import numpy as np
import pandas as pd

RAWDIR = Path(f"{ROOT}/results/paper/ablation/raw")
OUT = Path(f"{ROOT}/results/paper/ablation")
ARM_ORDER = ["full", "no_attention", "no_node_emb", "no_multi_horizon", "no_denorm",
             "no_onset_windows", "no_masking", "no_mag_w"]

files = [f for f in sorted(glob.glob(str(RAWDIR / "*_b*.csv"))) if not Path(f).name.startswith("table_")]
if not files:
    sys.exit(f"no per-arm csvs found in {RAWDIR}")
per = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
per["arm"] = pd.Categorical(per["arm"], ARM_ORDER, ordered=True)
per = per.sort_values(["embodiment", "arm", "localizer", "metric", "build_seed"]).reset_index(drop=True)
OUT.mkdir(parents=True, exist_ok=True)
per.to_csv(OUT / "ablation_by_build_seed.csv", index=False)

key = ["embodiment", "arm", "localizer", "metric"]
agg = per.groupby(key, observed=True).agg(
    value=("value", "mean"),
    sd=("value", lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")),
    n_build_seeds=("build_seed", "nunique"),
    n_eval_seeds_per_build=("n_seeds", "max"),
    n_hard=("n_hard", "mean"),
    n_fault=("n_fault", "mean"),
    n_candidates=("n_candidates", "max"),
    chance=("chance", "max"),
).reset_index()
agg["n_seeds"] = agg["n_build_seeds"] * agg["n_eval_seeds_per_build"]
agg = agg[["embodiment", "arm", "localizer", "metric", "value", "sd", "n_seeds",
           "n_build_seeds", "n_eval_seeds_per_build", "n_hard", "n_fault", "n_candidates", "chance"]]
agg.to_csv(OUT / "ablation.csv", index=False)
print(f"wrote {OUT/'ablation.csv'} ({len(agg)} rows) and ablation_by_build_seed.csv ({len(per)} rows) "
      f"from {len(files)} arm files")

# ---- leave-one-out deltas with a build-seed standard error -------------------
drows = []
for (emb, loc, met), grp in agg.groupby(["embodiment", "localizer", "metric"], observed=True):
    g = grp.set_index("arm")
    if "full" not in g.index:
        continue
    f = g.loc["full"]
    for arm in g.index:
        if arm == "full":
            continue
        a = g.loc[arm]
        nb = max(1, int(a["n_build_seeds"])); nf_ = max(1, int(f["n_build_seeds"]))
        sa = 0.0 if not np.isfinite(a["sd"]) else a["sd"]
        sf = 0.0 if not np.isfinite(f["sd"]) else f["sd"]
        se = float(np.sqrt(sa ** 2 / nb + sf ** 2 / nf_))
        d = float(a["value"] - f["value"])
        drows.append(dict(embodiment=emb, localizer=loc, metric=met, arm=arm,
                          full_value=float(f["value"]), arm_value=float(a["value"]),
                          delta_vs_full=d, se_delta=se,
                          z=(d / se if se > 0 else float("nan")),
                          n_build_seeds=nb, n_eval_seeds_per_build=int(a["n_eval_seeds_per_build"])))
dd = pd.DataFrame(drows)
dd["arm"] = pd.Categorical(dd["arm"], ARM_ORDER, ordered=True)
dd = dd.sort_values(["embodiment", "metric", "localizer", "arm"]).reset_index(drop=True)
dd.to_csv(OUT / "ablation_deltas.csv", index=False)
print(f"wrote {OUT/'ablation_deltas.csv'} ({len(dd)} rows)")

for emb in sorted(agg.embodiment.unique()):
    d = agg[agg.embodiment == emb]
    print(f"\n================ {emb} ================")
    ds = d[d.localizer == "__dataset__"]
    for stat in ("value", "sd"):
        pv = ds.pivot_table(index="arm", columns="metric", values=stat, observed=True)
        cols = [c for c in ("forward_nmse_heldout_healthy", "operator_max_coherence",
                            "operator_mean_coherence", "cause_victim_gap") if c in pv.columns]
        print(f"\n-- dataset scalars ({stat}) --")
        print(pv[cols].round(4).to_string())
    for metric in ("hard_top1", "top1"):
        m = d[(d.metric == metric) & (d.localizer != "__dataset__")]
        pv = m.pivot_table(index="arm", columns="localizer", values="value", observed=True)
        sv = m.pivot_table(index="arm", columns="localizer", values="sd", observed=True)
        print(f"\n-- {metric} mean over build x eval seeds --")
        print(pv.round(3).to_string())
        print(f"-- {metric} sd across build seeds --")
        print(sv.round(3).to_string())
        if "full" in pv.index:
            dlt = (pv - pv.loc["full"]).drop(index="full", errors="ignore")
            print(f"-- {metric} DELTA vs full (negative = switching the component OFF hurts) --")
            print(dlt.round(3).to_string())
            if "fsi" in dlt.columns:
                order = dlt["fsi"].sort_values()
                print(f">> fsi {metric} leave-one-out ranking on {emb} (most damaging first):")
                for a, v in order.items():
                    s = sv.loc[a, "fsi"] if a in sv.index else float("nan")
                    print(f"     {a:18s} {v:+.3f}  (arm sd {s:.3f}, full sd {sv.loc['full','fsi']:.3f})")
