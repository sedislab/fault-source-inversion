"""Core records: a fault label, an episode (one rollout), and the propagation operator G."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np

class Onset(str, Enum):
    ABRUPT = "abrupt"
    INCIPIENT = "incipient"

@dataclass
class FaultLabel:
    node: int
    fault_type: str
    magnitude: float
    onset: Onset = Onset.ABRUPT
    onset_step: int = 0
    end_step: int | None = None
    def __post_init__(self):
        self.onset = Onset(self.onset)

@dataclass
class Episode:
    """One rollout. `r` is the residual field (channel 0 is the energy that G is fit to). `faults` is the
    exact source label from injection; healthy episodes have none."""
    embodiment: str
    n: int
    r: np.ndarray
    faults: list[FaultLabel] = field(default_factory=list)
    y: np.ndarray | None = None
    yhat: np.ndarray | None = None
    u: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        return len(self.faults)

    @property
    def is_healthy(self) -> bool:
        return self.k == 0

    def support(self) -> list[int]:
        return sorted(f.node for f in self.faults)

    def source_vector(self) -> np.ndarray:
        s = np.zeros(self.n, np.float32)
        for f in self.faults:
            s[f.node] = f.magnitude
        return s

    def energy(self) -> np.ndarray:
        """Per-node scalar residual energy (channel 0)."""
        return self.r[:, 0] if self.r.ndim == 2 else self.r

@dataclass
class Operator:
    """Propagation operator: column v is the residual fingerprint of a unit fault at node v. `G` is the primary
    (energy) operator. `G_multi` optionally stacks one operator per signal channel, (C, n, n): a fault's torque
    fingerprint is near-diagonal (direct evidence) while its position fingerprint propagates (victims), and
    inverting both together is what lets FSI use the direct signal and the propagation at once."""
    embodiment: str
    G: np.ndarray
    node_names: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    G_multi: np.ndarray | None = None

    @property
    def n(self) -> int:
        return self.G.shape[0]

    def column(self, v: int) -> np.ndarray:
        return self.G[:, v]

    def coherence(self) -> float:
        """Mutual coherence: worst-case absolute cosine between distinct columns. Lower is more identifiable."""
        Gn = self.G / (np.linalg.norm(self.G, axis=0, keepdims=True) + 1e-12)
        M = np.abs(Gn.T @ Gn)
        np.fill_diagonal(M, 0.0)
        return float(M.max())

    def coherence_matrix(self) -> np.ndarray:
        Gn = self.G / (np.linalg.norm(self.G, axis=0, keepdims=True) + 1e-12)
        M = np.abs(Gn.T @ Gn)
        np.fill_diagonal(M, 0.0)
        return M

    def save(self, path: str | Path):
        extra = {} if self.G_multi is None else {"G_multi": self.G_multi}
        np.savez(Path(path), G=self.G, embodiment=self.embodiment,
                 node_names=np.asarray(self.node_names, object), meta=np.asarray(self.meta, object), **extra)

    @classmethod
    def load(cls, path: str | Path) -> Operator:
        d = np.load(Path(path), allow_pickle=True)
        return cls(str(d["embodiment"]), d["G"], list(d["node_names"]), dict(d["meta"].item()),
                   G_multi=d["G_multi"] if "G_multi" in d.files else None)
