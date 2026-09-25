"""Node-level decomposition with permutation p-values: across candidate source nodes, does a node's operator
column norm (fingerprint strength) or its mean coherence to the other columns predict how often that node is
missed as a source / falsely blamed? n is only 7 (Franka) or 12 (ANYmal) nodes, so these are reported with an
exact-ish permutation p over 5000 relabelings and read as a direction, not a precise effect size."""
from __future__ import annotations
import os, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])  # repository root (code)
ROOT = os.environ.get("FSI_ROOT", REPO)  # data/results root, see docs/CONFIGURATION.md
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

OUT = Path(f"{ROOT}/results/paper/identifiability")
B = 5000

def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    da, db = np.sqrt((a * a).sum()), np.sqrt((b * b).sum())
    return float("nan") if da < 1e-12 or db < 1e-12 else float((a * b).sum() / (da * db))

def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))

d = pd.read_csv(OUT / "node_level.csv")
rng = np.random.default_rng(3)
rows = []
for (tag, loc), s in d.groupby(["embodiment", "localizer"]):
    for pred in ("operator_column_norm", "mean_coherence_to_other_candidates", "mean_hop_to_other_candidates"):
        for tgt in ("error_rate_as_source", "false_blame_rate_as_predicted"):
            x, y = s[pred].values, s[tgt].values
            obs = spearman(x, y)
            if not np.isfinite(obs):
                rows.append({"embodiment": tag, "localizer": loc, "predictor": pred, "target": tgt,
                             "spearman_rho": obs, "p_perm": float("nan"), "n_perm": 0, "n_nodes": len(x)})
                continue
            hits = sum(1 for _ in range(B) if abs(spearman(x, rng.permutation(y))) >= abs(obs) - 1e-12)
            rows.append({"embodiment": tag, "localizer": loc, "predictor": pred, "target": tgt,
                         "spearman_rho": obs, "p_perm": (1 + hits) / (1 + B), "n_perm": B, "n_nodes": len(x)})
df = pd.DataFrame(rows)
df.to_csv(OUT / "node_correlation.csv", index=False)
pd.set_option("display.width", 220)
print(df[df.target == "error_rate_as_source"].round(4).to_string(index=False))
