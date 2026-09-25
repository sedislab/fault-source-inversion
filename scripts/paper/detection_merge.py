"""Merge the per-(embodiment, seed) shards with the released-dataset run into the deliverable CSVs, and print
the headline comparison. Deliverables written to results/paper/detection/:
  pr_roc.csv, far_sweep.csv, cusum.csv, onset.csv, onset_localization.csv  (+ *_summary.csv aggregates)
"""
import os, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])  # repository root (code)
ROOT = os.environ.get("FSI_ROOT", REPO)  # data/results root, see docs/CONFIGURATION.md
import sys, glob
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(f"{ROOT}/results/paper/detection")
SH = OUT / "shards"
KEYS = ["pr_roc", "far_sweep", "cusum", "onset", "onset_localization"]


def load(key):
    parts = [pd.read_csv(f) for f in sorted(glob.glob(str(SH / f"{key}__*.csv")))]
    rel = OUT / f"{key}_released.csv"
    if rel.exists():
        parts.append(pd.read_csv(rel))
    df = pd.concat(parts, ignore_index=True)
    df["fwd_seed"] = df["fwd_seed"].astype(str)
    return df


def agg(df, by, cols):
    """mean / sd / n over forward-model seeds, excluding the 'released' single draw."""
    d = df[df.fwd_seed != "released"]
    g = d.groupby(by, dropna=False)
    out = g[cols].agg(["mean", "std", "count"])
    out.columns = [f"{a}_{'sd' if b == 'std' else b}" if b != "mean" else a for a, b in out.columns]
    return out.reset_index()


def main():
    dfs = {}
    for k in KEYS:
        df = load(k)
        df.to_csv(OUT / f"{k}.csv", index=False)
        dfs[k] = df
        print(f"{k}.csv: {len(df)} rows, seeds {sorted(df.fwd_seed.unique())}")

    # --- headline: matched-FAR detection comparison, averaged over forward-model seeds ------
    c = dfs["cusum"]
    s = agg(c, ["embodiment", "statistic", "nu", "target_far"],
            ["recall", "realized_far", "recall_at_matched_test_far", "precision", "f1", "auprc", "auroc",
             "mean_delay_steps_matched", "median_delay_steps_matched", "n_delay_matched",
             "mean_abs_onset_err_steps_matched", "median_abs_onset_err_steps_matched",
             "mean_abs_onset_err_const_steps_matched", "frac_alarm_before_onset_matched"])
    s.to_csv(OUT / "cusum_summary.csv", index=False)
    f = agg(dfs["far_sweep"], ["embodiment", "statistic", "nu", "target_far"],
            ["recall", "realized_far", "precision", "f1", "auprc", "auroc"])
    f.to_csv(OUT / "far_sweep_summary.csv", index=False)
    # compact AUC-only table (pr_roc.csv itself is ~4 MB of curve points)
    pr = dfs["pr_roc"].drop_duplicates(["embodiment", "statistic", "nu", "fwd_seed"])
    pr = pr[["embodiment", "statistic", "nu", "fwd_seed", "auprc", "auroc",
             "n_test_healthy", "n_test_fault"]]
    pr["prevalence"] = pr.n_test_fault / (pr.n_test_fault + pr.n_test_healthy)
    pr.to_csv(OUT / "pr_roc_auc.csv", index=False)
    a = agg(pr, ["embodiment", "statistic", "nu"], ["auprc", "auroc", "prevalence"])
    a.to_csv(OUT / "pr_roc_summary.csv", index=False)

    for emb in ("franka", "anymal"):
        print(f"\n===== {emb}: matched-FAR 0.05, mean +- sd over 3 forward-model seeds =====")
        t = s[(s.embodiment == emb) & (s.target_far == 0.05)].copy()
        t["recall_cal"] = t.apply(lambda r: f"{r.recall:.3f}+-{r.recall_sd:.3f}", 1)
        t["recall_matched"] = t.apply(
            lambda r: f"{r.recall_at_matched_test_far:.3f}+-{r.recall_at_matched_test_far_sd:.3f}", 1)
        t["AUPRC"] = t.apply(lambda r: f"{r.auprc:.3f}+-{r.auprc_sd:.3f}", 1)
        t["AUROC"] = t.apply(lambda r: f"{r.auroc:.3f}+-{r.auroc_sd:.3f}", 1)
        t["delay"] = t.apply(lambda r: f"{r.mean_delay_steps_matched:.1f}+-{r.mean_delay_steps_matched_sd:.1f}", 1)
        t["onset_err"] = t.apply(
            lambda r: f"{r.mean_abs_onset_err_steps_matched:.1f}+-{r.mean_abs_onset_err_steps_matched_sd:.1f}", 1)
        print(t[["statistic", "nu", "recall_cal", "realized_far", "recall_matched", "AUPRC", "AUROC",
                 "delay", "n_delay_matched", "onset_err",
                 "mean_abs_onset_err_const_steps_matched"]].to_string(index=False))

    # --- onset estimate quality -------------------------------------------------------------
    o = dfs["onset"]
    o = o[o.fwd_seed != "released"]
    det = o[o.detected == 1]
    rows = []
    for (emb, sd_), gsub in det.groupby(["embodiment", "fwd_seed"]):
        allg = o[(o.embodiment == emb) & (o.fwd_seed == sd_)]
        rows.append({"embodiment": emb, "fwd_seed": sd_, "n_fault_test": len(allg),
                     "n_alarmed": len(gsub), "alarm_rate": len(gsub) / len(allg),
                     "mean_abs_onset_err": gsub.abs_onset_err_steps.mean(),
                     "median_abs_onset_err": gsub.abs_onset_err_steps.median(),
                     "mean_signed_onset_err": gsub.signed_onset_err_steps.mean(),
                     "mean_abs_onset_err_const": gsub.abs_onset_err_const_steps.mean(),
                     "mean_delay": gsub.delay_steps.mean()})
    od = pd.DataFrame(rows)
    od.to_csv(OUT / "onset_summary.csv", index=False)
    print("\n===== CUSUM onset estimate (nu=2.0, FAR 0.05), per forward seed =====")
    print(od.round(3).to_string(index=False))

    # --- deployability -----------------------------------------------------------------------
    ol = dfs["onset_localization"]
    k = agg(ol, ["embodiment", "localizer", "subset", "onset_source", "train_field_onset"],
            ["top1", "hard_top1", "w1g", "n_episodes", "chance"])
    k.to_csv(OUT / "onset_localization_summary.csv", index=False)
    print("\n===== localization with LABEL onset vs CUSUM-estimated onset =====")
    for emb in ("franka", "anymal"):
        print(f"-- {emb}")
        t = k[(k.embodiment == emb)].copy()
        t["top1_s"] = t.apply(lambda r: f"{r.top1:.3f}+-{r.top1_sd:.3f}", 1)
        t["hard_s"] = t.apply(lambda r: f"{r.hard_top1:.3f}+-{r.hard_top1_sd:.3f}", 1)
        print(t[["localizer", "subset", "onset_source", "train_field_onset", "n_episodes",
                 "top1_s", "hard_s", "chance"]].to_string(index=False))


main()
