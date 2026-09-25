"""Dataset splits at the EPISODE level (never per timestep, which would leak autocorrelation into calibration).
Beyond the in-distribution train/cal/test, we hold out whole fault types and provide a fault-free split so the
false-alarm rate is measured on data no method trained on."""
from __future__ import annotations
import numpy as np
from ..core.types import Episode

def make_splits(episodes: list[Episode], cal=0.15, test=0.25, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(episodes))
    n = len(episodes); nt = int(test * n); nc = int(cal * n)
    pick = lambda ii: [episodes[i] for i in ii]
    return {"test": pick(idx[:nt]), "cal": pick(idx[nt:nt + nc]), "train": pick(idx[nt + nc:])}

def fault_type(ep: Episode) -> str | None:
    return ep.faults[0].fault_type if ep.faults else None

def holdout_fault_types(episodes, held: set[str]) -> dict:
    """Split into in-distribution (no held fault type present) and held-out (contains a held fault type)."""
    in_d, out = [], []
    for e in episodes:
        (out if any(f.fault_type in held for f in e.faults) else in_d).append(e)
    return {"in_dist": in_d, "held_fault": out}

def fault_free(episodes) -> list[Episode]:
    return [e for e in episodes if e.is_healthy]

def by_fault_count(episodes) -> dict:
    out = {}
    for e in episodes:
        out.setdefault(e.k, []).append(e)
    return out
