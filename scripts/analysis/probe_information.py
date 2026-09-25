"""Where is the source identity lost? Train the same simple classifier to predict the faulted node from
(a) the RAW post-onset window, (b) the reduced residual field the localizer actually sees. A large gap proves the
residual reduction destroys information rather than the models being weak."""
import os, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])  # repository root (code)
ROOT = os.environ.get("FSI_ROOT", REPO)  # data/results root, see docs/CONFIGURATION.md
import sys, argparse
import numpy as np
sys.path.insert(0, REPO)
from pathlib import Path
from fsi.data import load_episodes
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--cap", type=int, default=4000)
ap.add_argument("--stride", type=int, default=4)
args = ap.parse_args()

eps = load_episodes(Path(args.data) / "all")
rng = np.random.default_rng(0)
faults = [e for e in eps if not e.is_healthy]
if len(faults) > args.cap:
    faults = [faults[i] for i in rng.choice(len(faults), args.cap, replace=False)]
y_lab = np.array([e.support()[0] for e in faults])
print(f"{len(faults)} fault episodes, {len(set(y_lab))} classes, chance={1/len(set(y_lab)):.3f}", flush=True)

def raw_feats(e):
    w = e.y[-60:][::args.stride]                       # (k, n, C)
    u = e.u[-60:][::args.stride]
    return np.concatenate([w.ravel(), u.ravel()])

def resid_feats(e):
    return e.r.ravel()


def selfref_feats(e):
    """Within-episode reference: the SAME episode's pre-onset window is a healthy baseline for that exact command,
    so post-minus-pre cancels the command and the model error that no cross-episode statistic can."""
    onset = int(e.meta.get("onset_step", len(e.y) // 2))
    pre = e.y[max(0, onset - 50):max(5, onset)]
    post = e.y[min(onset + 5, len(e.y) - 5):]
    def f(w):
        return np.concatenate([w.mean(0).ravel(), w.std(0).ravel(), np.abs(np.diff(w, axis=0)).mean(0).ravel()])
    return f(post) - f(pre)

for name, fn in (("RAW window (y,u)", raw_feats), ("REDUCED residual field r", resid_feats), ("SELF-REFERENCED (post-pre)", selfref_feats)):
    X = np.stack([fn(e) for e in faults])
    Xtr, Xte, ytr, yte = train_test_split(X, y_lab, test_size=0.25, random_state=0, stratify=y_lab)
    clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0).fit(Xtr, ytr)
    acc = accuracy_score(yte, clf.predict(Xte))
    print(f"  {name:28s} dim={X.shape[1]:5d}  top1={acc:.3f}", flush=True)
