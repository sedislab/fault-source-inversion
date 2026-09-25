"""Rebuild top_coherent_pairs.csv from pairs.csv with the reference level a confusion rate has to be read
against: the mean off-diagonal confusion rate of that same localizer on that embodiment. Without it, "the most
coherent pair is confused 3% of the time" is unreadable -- 3% is above chance on ANYmal and below it on Franka.
Adds the five LEAST coherent pairs as the contrast the theory predicts should be the safe ones.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])  # repository root (code)
ROOT = os.environ.get("FSI_ROOT", REPO)  # data/results root, see docs/CONFIGURATION.md
from pathlib import Path
import pandas as pd

OUT = Path(f"{ROOT}/results/paper/identifiability")
INFORMATIVE = ["fsi", "fsi_rbc", "graphsl", "direct_l1", "largest_residual", "gdn", "tranad", "rcd", "aerca"]

d = pd.read_csv(OUT / "pairs.csv")
d = d[d.episode_subset == "all_fault"].copy()
d["pair"] = [tuple(sorted(t)) for t in zip(d.node_i, d.node_j)]
rows = []
for tag, sub in d.groupby("embodiment"):
    names = dict(zip(sub.node_i, sub.name_i))
    ref = sub.groupby("localizer")["confusion_rate"].mean()          # mean off-diagonal rate per localizer
    fsi = sub[sub.localizer == "fsi"]
    inf = sub[sub.localizer.isin(INFORMATIVE)]
    g = fsi.groupby("pair").agg(coherence=("coherence", "first"), hop_distance=("hop_distance", "first"),
                                coherence_multi_mean=("coherence_multi_mean", "first"),
                                confusion_rate_fsi=("confusion_rate", "mean")).reset_index()
    g["confusion_rate_mean_informative_localizers"] = g["pair"].map(
        inf.groupby("pair")["confusion_rate"].mean())
    g["embodiment"] = tag
    g["node_i"] = [p[0] for p in g["pair"]]; g["node_j"] = [p[1] for p in g["pair"]]
    g["name_i"] = g.node_i.map(names); g["name_j"] = g.node_j.map(names)
    g["mean_offdiag_confusion_rate_fsi"] = float(ref["fsi"])
    g["mean_offdiag_confusion_rate_informative_localizers"] = float(ref[INFORMATIVE].mean())
    g["fsi_confusion_vs_embodiment_mean"] = g.confusion_rate_fsi / float(ref["fsi"])
    g = g.sort_values("coherence", ascending=False).drop(columns=["pair"])
    rows.append(g.head(5).assign(rank_type="most_coherent"))
    rows.append(g.tail(5).assign(rank_type="least_coherent"))
out = pd.concat(rows)[["embodiment", "rank_type", "node_i", "node_j", "name_i", "name_j", "coherence",
                       "coherence_multi_mean", "hop_distance", "confusion_rate_fsi",
                       "mean_offdiag_confusion_rate_fsi", "fsi_confusion_vs_embodiment_mean",
                       "confusion_rate_mean_informative_localizers",
                       "mean_offdiag_confusion_rate_informative_localizers"]]
out.to_csv(OUT / "top_coherent_pairs.csv", index=False)
print(out.round(4).to_string(index=False))
