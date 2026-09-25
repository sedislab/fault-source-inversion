"""Aggregate the per-FSI-seed conformal outputs into figure-ready tables:
  conformal_alpha_summary.csv   coverage / set size vs alpha, mean+sd over FSI seeds
  conformal_sizes_hist.csv      set-size histogram at alpha = 0.1 (mean+sd of the per-seed fraction)
  conformal_group_summary.csv   per-fault-type (Mondrian group) coverage, mean+sd over FSI seeds
"""
import os, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])  # repository root (code)
ROOT = os.environ.get("FSI_ROOT", REPO)  # data/results root, see docs/CONFIGURATION.md
from pathlib import Path
import pandas as pd

OUT = Path(f"{ROOT}/results/paper/detection")
KEY = ["embodiment", "dataset", "candidate_scope", "n_candidates", "mode", "grouping"]


def summarize(df, by, cols):
    g = df.groupby(by, dropna=False)[cols].agg(["mean", "std", "count"])
    g.columns = [f"{a}_{'sd' if b == 'std' else b}" if b != "mean" else a for a, b in g.columns]
    return g.reset_index()


def main():
    a = pd.read_csv(OUT / "conformal_alpha.csv")
    summarize(a, KEY + ["alpha", "target_coverage"],
              ["coverage", "coverage_deviation", "mean_set_size", "median_set_size", "p90_set_size",
               "frac_singleton", "frac_full_set", "n_cal", "n_test"]
              ).rename(columns={"coverage_count": "n_fsi_seeds"}).to_csv(
        OUT / "conformal_alpha_summary.csv", index=False)

    s = pd.read_csv(OUT / "conformal_sizes.csv")
    # Reindex onto the full 1..n_candidates size grid so a size that a given seed never produced counts as 0
    # rather than being dropped (dropping it would inflate that size's mean fraction).
    parts = []
    for keys, sub in s.groupby(KEY + ["alpha", "fsi_seed"], dropna=False):
        n_cand = int(dict(zip(KEY + ["alpha", "fsi_seed"], keys))["n_candidates"])
        cnt = sub.set_size.value_counts().reindex(range(1, n_cand + 1), fill_value=0).sort_index()
        d = pd.DataFrame({"set_size": cnt.index, "n_episodes": cnt.values,
                          "fraction_of_episodes": cnt.values / len(sub)})
        for c, v in zip(KEY + ["alpha", "fsi_seed"], keys):
            d[c] = v
        parts.append(d)
    h = pd.concat(parts, ignore_index=True)
    summarize(h, KEY + ["alpha", "set_size"], ["fraction_of_episodes", "n_episodes"]).rename(
        columns={"fraction_of_episodes_count": "n_fsi_seeds"}).to_csv(
        OUT / "conformal_sizes_hist.csv", index=False)

    g = pd.read_csv(OUT / "conformal_group.csv")
    summarize(g, KEY + ["alpha", "target_coverage", "fault_type"],
              ["coverage", "coverage_deviation", "mean_set_size", "n_test"]).rename(
        columns={"coverage_count": "n_fsi_seeds"}).to_csv(OUT / "conformal_group_summary.csv", index=False)
    print("wrote conformal_alpha_summary.csv, conformal_sizes_hist.csv, conformal_group_summary.csv")


main()
