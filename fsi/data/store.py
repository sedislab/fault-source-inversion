"""Compact on-disk episodes: residual fields + source labels as arrays, faults/meta as JSON, and the raw
time-series only when present. One shard per file so the dataset streams from /work without loading whole."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from ..core.types import Episode, FaultLabel, Onset

def save_episodes(episodes: list[Episode], path: str | Path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    n = episodes[0].n
    r = np.stack([e.r if e.r.ndim == 2 else e.r[:, None] for e in episodes]).astype(np.float32)
    src = np.stack([e.source_vector() for e in episodes]).astype(np.float32)
    arrays = {"r": r, "src": src}
    if all(e.y is not None for e in episodes):
        arrays["y"] = np.stack([e.y for e in episodes]).astype(np.float32)
        arrays["u"] = np.stack([e.u for e in episodes]).astype(np.float32)
    # Guard each series separately: `np.stack([None, ...]).astype(np.float32)` silently writes an array of NaN
    # rather than raising, which is how the stored `yhat` became all-NaN and quietly killed every method that
    # reads y - yhat.
    if all(e.yhat is not None for e in episodes):
        arrays["yhat"] = np.stack([e.yhat for e in episodes]).astype(np.float32)
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    labels = [{"embodiment": e.embodiment, "n": e.n, "meta": e.meta,
               "faults": [{"node": f.node, "fault_type": f.fault_type, "magnitude": f.magnitude,
                           "onset": f.onset.value, "onset_step": f.onset_step, "end_step": f.end_step}
                          for f in e.faults]} for e in episodes]
    path.with_suffix(".json").write_text(json.dumps(labels))

def load_episodes(path: str | Path) -> list[Episode]:
    path = Path(path)
    d = np.load(path.with_suffix(".npz"))
    labels = json.loads(path.with_suffix(".json").read_text())
    has_series = "y" in d.files
    r_all, y_all = d["r"], (d["y"] if has_series else None)   # materialize once: NpzFile re-decompresses per access
    yh_all = d["yhat"] if has_series and "yhat" in d.files else None
    u_all = d["u"] if has_series and "u" in d.files else None
    out = []
    for i, lab in enumerate(labels):
        faults = [FaultLabel(f["node"], f["fault_type"], f["magnitude"], Onset(f["onset"]),
                             f["onset_step"], f["end_step"]) for f in lab["faults"]]
        out.append(Episode(lab["embodiment"], lab["n"], r_all[i], faults,
                           y=y_all[i] if y_all is not None else None,
                           yhat=yh_all[i] if yh_all is not None else None,
                           u=u_all[i] if u_all is not None else None, meta=lab["meta"]))
    return out
