"""Second pass on the identifiability question, run entirely off the matrices the first pass saved (no refitting).

The pooled pair-level Spearman in correlation.csv mixes two very different effects:
  * a ROW effect -- some source nodes are simply easier to localize, so every off-diagonal entry in their row is
    small regardless of which node j it points at;
  * the PAIR effect the theory actually predicts -- given that the method erred on source i, the error should
    land on the node whose operator column is most collinear with column i.
Only the second is the identifiability claim. This script isolates it: for each true source i separately, rank the
K-1 candidate wrong answers by coherence(i, .) and by how often the method actually picked them, take the
Spearman per row, and average. The null shuffles the off-diagonal entries WITHIN each row independently, so
whatever makes a row easy or hard is held fixed by construction. Hop distance goes through the identical test.

It also reports the node-level decomposition (mean coherence vs. that node's error rate and vs. how often it is
falsely blamed) so the sign of the pooled correlation can be attributed.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])  # repository root (code)
ROOT = os.environ.get("FSI_ROOT", REPO)  # data/results root, see docs/CONFIGURATION.md
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

OUT = Path(f"{ROOT}/results/paper/identifiability")
B = 2000
LOCS = ["fsi", "fsi_rbc", "graphsl", "direct_l1", "largest_residual", "gdn", "tranad", "rcd", "aerca",
        "zz_constant", "zz_anti_energy", "zz_random"]

def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    da, db = np.sqrt((a * a).sum()), np.sqrt((b * b).sum())
    return float("nan") if da < 1e-12 or db < 1e-12 else float((a * b).sum() / (da * db))

def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))

def offdiag_rows(M):
    """(K, K-1) array: row i holds M[i, j] for j != i, in increasing j."""
    K = M.shape[0]
    return np.stack([np.delete(M[i], i) for i in range(K)])

def mean_row_rho(P, Cr):
    v = [spearman(P[i], Cr[i]) for i in range(P.shape[0])]
    v = [x for x in v if np.isfinite(x)]
    return (float(np.mean(v)), len(v)) if v else (float("nan"), 0)

def main():
    rows, node_rows = [], []
    for tag in ("franka_v6", "anymal_v4"):
        M = np.load(OUT / f"coherence_{tag}.npy")
        hop = np.load(OUT / f"hop_{tag}.npy")
        Mc = np.load(OUT / f"coherence_multi_{tag}.npy")
        names = json.loads((OUT / f"names_{tag}.json").read_text())
        K = M.shape[0]
        # Alternative operator-derived pair predictors, to say what DOES explain the errors if coherence does not.
        # G_response[i, j] = G[j, i] = the operator's predicted residual at node j given a unit fault at source i,
        # i.e. "how loud a victim does i make of j" -- the loudest-victim story rather than the collinearity story.
        cand = np.load(OUT / f"candidates_{tag}.npy")
        G = np.load(f"{ROOT}/data/{tag}/operator_G.npz", allow_pickle=True)["G"]
        Gc = G[np.ix_(cand, cand)]
        Gresp = np.abs(Gc.T)                                  # [i, j] = |G[j, i]|
        colnorm = np.linalg.norm(G[:, cand], axis=0)          # fingerprint strength of each candidate column
        Gcolnorm = np.tile(colnorm[None, :], (K, 1))          # [i, j] = ||G[:, j]||  (a pure column effect)
        preds = {"coherence_G": offdiag_rows(M), "hop_distance": offdiag_rows(hop),
                 "coherence_multi_mean": offdiag_rows(Mc.mean(0)),
                 "coherence_multi_max": offdiag_rows(Mc.max(0)),
                 "G_response_at_j": offdiag_rows(Gresp),
                 "column_norm_of_j": offdiag_rows(Gcolnorm)}
        rng = np.random.default_rng(7)
        for loc in LOCS:
            f = OUT / f"confusion_{tag}_{loc}.npy"
            if not f.exists():
                continue
            R = np.load(f)                       # row-normalised P(pred=j | true=i), pooled over 5 seeds
            Cr = offdiag_rows(R)                 # (K, K-1) where the errors of source i went
            for pname, P in preds.items():
                obs, nr = mean_row_rho(P, Cr)
                if not np.isfinite(obs):
                    rows.append({"embodiment": tag, "localizer": loc, "predictor": pname,
                                 "mean_row_spearman": obs, "p_perm": float("nan"), "n_perm": 0,
                                 "n_rows": nr, "n_per_row": K - 1})
                    continue
                hits = 0
                for _ in range(B):
                    Cp = np.stack([rng.permutation(r) for r in Cr])
                    v, _ = mean_row_rho(P, Cp)
                    if np.isfinite(v) and abs(v) >= abs(obs) - 1e-12:
                        hits += 1
                rows.append({"embodiment": tag, "localizer": loc, "predictor": pname,
                             "mean_row_spearman": obs, "p_perm": (1 + hits) / (1 + B), "n_perm": B,
                             "n_rows": nr, "n_per_row": K - 1})
            # node-level decomposition
            offmask = ~np.eye(K, dtype=bool)
            mean_coh_row = np.array([M[i][offmask[i]].mean() for i in range(K)])
            mean_hop_row = np.array([hop[i][offmask[i]].mean() for i in range(K)])
            err_rate = 1.0 - np.diag(R)
            blame = np.array([R[:, j][offmask[:, j]].mean() for j in range(K)])
            for i in range(K):
                node_rows.append({"embodiment": tag, "localizer": loc, "node_name": names[i],
                                  "mean_coherence_to_other_candidates": float(mean_coh_row[i]),
                                  "mean_hop_to_other_candidates": float(mean_hop_row[i]),
                                  "operator_column_norm": float(colnorm[i]),
                                  "error_rate_as_source": float(err_rate[i]),
                                  "false_blame_rate_as_predicted": float(blame[i])})
            rows.append({"embodiment": tag, "localizer": loc, "predictor": "node_colnorm_vs_errorrate",
                         "mean_row_spearman": spearman(colnorm, err_rate), "p_perm": float("nan"),
                         "n_perm": 0, "n_rows": K, "n_per_row": 1})
            rows.append({"embodiment": tag, "localizer": loc, "predictor": "node_colnorm_vs_falseblame",
                         "mean_row_spearman": spearman(colnorm, blame), "p_perm": float("nan"),
                         "n_perm": 0, "n_rows": K, "n_per_row": 1})
            rows.append({"embodiment": tag, "localizer": loc, "predictor": "node_meancoh_vs_errorrate",
                         "mean_row_spearman": spearman(mean_coh_row, err_rate), "p_perm": float("nan"),
                         "n_perm": 0, "n_rows": K, "n_per_row": 1})
            rows.append({"embodiment": tag, "localizer": loc, "predictor": "node_meancoh_vs_falseblame",
                         "mean_row_spearman": spearman(mean_coh_row, blame), "p_perm": float("nan"),
                         "n_perm": 0, "n_rows": K, "n_per_row": 1})
    pd.DataFrame(rows).to_csv(OUT / "within_row_correlation.csv", index=False)
    pd.DataFrame(node_rows).to_csv(OUT / "node_level.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))

main()
