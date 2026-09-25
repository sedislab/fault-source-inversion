"""THE decisive experiment. Matched sweep pairs give the ground-truth counterfactual residual
r* = y_faulted - y_matched_healthy (identical command and initial state): the cleanest fault signal physically
obtainable, with zero model error. If a strong classifier on r* still cannot identify the faulted joint, the
limit is physics/identifiability, not the pipeline. If it can, the pipeline is the bottleneck."""
import sys, json, glob, argparse
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True)
ap.add_argument("--nj", type=int, default=7)
ap.add_argument("--node_off", type=int, default=0, help="label offset: anymal stores node ids (joint+2)")
args = ap.parse_args()

S = sorted(glob.glob(f"{args.raw}/*_sweep_*.npz"))
R = sorted(glob.glob(f"{args.raw}/*_sweepref_*.npz"))
assert len(S) == len(R) and S, f"need matched sweeps, got {len(S)}/{len(R)}"
X, Y = [], []
for fs, fr in zip(S, R):
    ds, dr = np.load(fs, allow_pickle=True), np.load(fr, allow_pickle=True)
    lab = json.loads(str(ds["labels"]))
    d = ds["signals"] - dr["signals"]                       # (N,T,node,C) ORACLE counterfactual residual
    on = int(lab[0]["onset"])
    for e in range(d.shape[0]):
        post = d[e, on:]                                    # post-onset only
        mag = np.sqrt((post ** 2).sum(-1))                  # (t, node)
        feats = np.concatenate([
            post.mean(0).ravel(), post.std(0).ravel(),      # signed per-channel
            np.abs(post).max(0).ravel(),
            mag.mean(0), mag.max(0),
            mag.argmax(0) / max(1, len(mag) - 1),           # WHEN each node peaks
            (mag > 0.2 * (mag.max() + 1e-9)).argmax(0) / max(1, len(mag) - 1),  # first crossing, global thresh
        ])
        X.append(feats); Y.append(lab[e]["joint"] - args.node_off)
X, Y = np.stack(X), np.asarray(Y)
print(f"{len(X)} matched-pair episodes, {len(set(Y))} classes, chance={1/len(set(Y)):.3f}, dim={X.shape[1]}", flush=True)
Xtr, Xte, ytr, yte = train_test_split(X, Y, test_size=0.25, random_state=0, stratify=Y)
clf = HistGradientBoostingClassifier(max_iter=400, random_state=0).fit(Xtr, ytr)
pred = clf.predict(Xte)
print(f"ORACLE-residual top1 = {accuracy_score(yte, pred):.3f}", flush=True)
# how far off when wrong?
import collections
cm = collections.Counter((int(a), int(b)) for a, b in zip(yte, pred) if a != b)
print("most common confusions (true->pred):", cm.most_common(6), flush=True)
