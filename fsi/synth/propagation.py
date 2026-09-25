"""A synthetic robot-like system whose faults genuinely propagate, so cause != loudest victim.

The residual field is the propagated fault: r = reduce(y - yhat) over a window, and r ~ G s where G is the
Green's function of the coupling and s is the sparse source. Downstream nodes amplify (victim_gain), so the
loudest residual is often a descendant, not the source. Everything the inverse/metrics/baselines consume comes
from this exactly as it will from Isaac Lab, so the whole stack is validated on CPU before any sim data exists.
"""
from __future__ import annotations
import numpy as np
import networkx as nx
from ..core.graph import StructuralGraph, Node, Edge, NodeType, EdgeKind
from ..core.types import Episode, Operator, FaultLabel, Onset

def chain_graph(n_joints: int = 4, name: str = "synth_chain") -> StructuralGraph:
    """Serial-manipulator-like: base, then alternating actuator/link, each actuator with a torque sensor, tool tip."""
    nodes, edges = [Node(0, "base", NodeType.LINK)], []
    prev = 0
    for j in range(n_joints):
        a = len(nodes); nodes.append(Node(a, f"J{j+1}", NodeType.ACTUATOR))
        s = len(nodes); nodes.append(Node(s, f"tau{j+1}", NodeType.SENSOR))
        edges += [Edge(prev, a, EdgeKind.PHYSICAL), Edge(a, s, EdgeKind.PHYSICAL)]
        prev = a
        if j < n_joints - 1:
            l = len(nodes); nodes.append(Node(l, f"L{j+1}", NodeType.LINK))
            edges.append(Edge(prev, l, EdgeKind.PHYSICAL)); prev = l
    tool = len(nodes); nodes.append(Node(tool, "tool", NodeType.SENSOR))
    edges.append(Edge(prev, tool, EdgeKind.PHYSICAL))
    return StructuralGraph(nodes, edges, name)

def tree_graph(n_legs: int = 4, name: str = "synth_tree") -> StructuralGraph:
    """Quadruped-like: a base with an IMU, branching into legs of hip/knee actuators ending in a foot sensor."""
    nodes = [Node(0, "base", NodeType.BODY), Node(1, "IMU", NodeType.SENSOR)]
    edges = [Edge(0, 1, EdgeKind.PHYSICAL)]
    for leg in range(n_legs):
        h = len(nodes); nodes.append(Node(h, f"H{leg+1}", NodeType.ACTUATOR))
        k = len(nodes); nodes.append(Node(k, f"K{leg+1}", NodeType.ACTUATOR))
        c = len(nodes); nodes.append(Node(c, f"C{leg+1}", NodeType.SENSOR))
        edges += [Edge(0, h, EdgeKind.PHYSICAL), Edge(h, k, EdgeKind.PHYSICAL), Edge(k, c, EdgeKind.PHYSICAL)]
    return StructuralGraph(nodes, edges, name)

def star_graph(n_rotors: int = 4, name: str = "synth_star") -> StructuralGraph:
    """Multirotor-like: a body with IMU/baro/GNSS, coupled to rotor actuators through allocation."""
    nodes = [Node(0, "body", NodeType.BODY)]
    edges = []
    for s in ("IMU", "baro", "GNSS"):
        i = len(nodes); nodes.append(Node(i, s, NodeType.SENSOR)); edges.append(Edge(0, i, EdgeKind.PHYSICAL))
    for r in range(n_rotors):
        i = len(nodes); nodes.append(Node(i, f"M{r+1}", NodeType.ACTUATOR)); edges.append(Edge(0, i, EdgeKind.ALLOCATION))
    return StructuralGraph(nodes, edges, name)

class SyntheticSystem:
    def __init__(self, graph: StructuralGraph, root: int = 0, decay: float = 0.68, victim_gain: float = 2.6,
                 signature: float = 0.9, sig_dim: int = 6, jitter: float = 0.3, sat: float = 6.0,
                 noise: float = 0.04, du: int = 4, T: int = 160, window: int = 96, seed: int = 0):
        self.g = graph
        self.root = root
        self.decay = decay
        self.victim_gain = victim_gain
        self.signature = signature
        self.sig_dim = sig_dim
        self.jitter = jitter
        self.sat = sat
        self.noise = noise
        self.du = du
        self.T = T
        self.window = window
        rng = np.random.default_rng(seed)
        self.P = self._build_P(rng)
        self.W = rng.normal(0, 1.0, (graph.n, du)).astype(np.float32)
        self.phase = rng.uniform(0, 2 * np.pi, graph.n).astype(np.float32)
        self.carrier_hz = rng.uniform(0.05, 0.15, graph.n).astype(np.float32)
        self._names = graph.names()

    def _amp(self) -> np.ndarray:
        """Victim-proneness per node: sensors and central/body nodes read aggregated consequences, so they
        light up as loud victims; ordinary actuators/links do not. A fault at an actuator therefore peaks at a
        downstream sensor (propagation), while a fault at a sensor tends to peak at itself (localizes)."""
        amp = np.ones(self.g.n, np.float32)
        for nd in self.g.nodes:
            if nd.type in (NodeType.SENSOR, NodeType.BODY):
                amp[nd.id] = self.victim_gain
        return amp

    def _build_P(self, rng) -> np.ndarray:
        """Column j is the residual fingerprint of a unit fault at j: hop-decay into victim-amplified nodes,
        modulated by a source/observer signature so distinct faults excite the shared victims differently.
        `signature` trades identifiability (distinct columns) against pure propagation (loud shared victims)."""
        D = self.g.hop_distance()
        amp = self._amp()
        jit = rng.uniform(1 - self.jitter, 1 + self.jitter, (self.g.n, self.g.n)).astype(np.float32)
        phi = rng.normal(0, 1, (self.g.n, self.sig_dim)).astype(np.float32)
        psi = rng.normal(0, 1, (self.g.n, self.sig_dim)).astype(np.float32)
        sig = 1.0 + self.signature * np.tanh((phi @ psi.T) / np.sqrt(self.sig_dim))
        P = np.zeros((self.g.n, self.g.n), np.float32)
        for j in range(self.g.n):
            for i in range(self.g.n):
                if D[i, j] < self.g.n:
                    P[i, j] = amp[i] * (self.decay ** D[i, j]) * jit[i, j] * max(sig[i, j], 0.05)
        return P

    def operator_true(self) -> Operator:
        return Operator(self.g.name, self.P.copy(), self._names, {"kind": "analytic"})

    def _command(self, rng) -> np.ndarray:
        t = np.arange(self.T)[:, None]
        freqs = rng.uniform(0.01, 0.08, self.du)
        ph = rng.uniform(0, 2 * np.pi, self.du)
        return np.sin(2 * np.pi * freqs * t + ph).astype(np.float32)

    def _healthy(self, u: np.ndarray) -> np.ndarray:
        z = u @ self.W.T + self.phase
        return (np.sin(z) + 0.3 * np.sin(2 * z)).astype(np.float32)

    def _activation(self, onset: Onset, onset_step: int) -> np.ndarray:
        a = np.zeros(self.T, np.float32)
        if onset == Onset.ABRUPT:
            a[onset_step:] = 1.0
        else:
            ramp = np.linspace(0, 1, max(1, self.T - onset_step), dtype=np.float32)
            a[onset_step:] = ramp
        return a

    def _field(self, s: np.ndarray) -> np.ndarray:
        f = self.P @ s
        return self.sat * np.tanh(f / self.sat)

    def _reduce(self, resid: np.ndarray) -> np.ndarray:
        """Per-node residual features over the analysis window: [rms energy, peak, mean-abs]."""
        w = resid[-self.window:]
        rms = np.sqrt((w ** 2).mean(0))
        peak = np.abs(w).max(0)
        mad = np.abs(w).mean(0)
        return np.stack([rms, peak, mad], 1).astype(np.float32)

    def episode(self, rng, support: list[int], mags: list[float], onset: Onset = Onset.ABRUPT,
                onset_step: int | None = None, keep_series: bool = True) -> Episode:
        if onset_step is None:
            onset_step = int(self.T - self.window)
        u = self._command(rng)
        healthy = self._healthy(u)
        s = np.zeros(self.g.n, np.float32)
        for v, m in zip(support, mags):
            s[v] = m
        field = self._field(s)
        a = self._activation(onset, onset_step)
        t = np.arange(self.T)[:, None]
        carrier = np.sin(2 * np.pi * self.carrier_hz * t)
        y = healthy + field[None, :] * a[:, None] * carrier + rng.normal(0, self.noise, healthy.shape)
        yhat = healthy + rng.normal(0, self.noise, healthy.shape)
        r = self._reduce(y - yhat)
        faults = [FaultLabel(v, "synthetic", float(m), onset, onset_step) for v, m in zip(support, mags)]
        ep = Episode(self.g.name, self.g.n, r, faults,
                     y=y if keep_series else None, yhat=yhat if keep_series else None,
                     u=u if keep_series else None, meta={"onset_step": onset_step})
        return ep

    def sample(self, rng, k: int, mag_range=(0.6, 1.6), onset: Onset | None = None, keep_series: bool = True) -> Episode:
        fail = self.g.fail_nodes
        support = list(rng.choice(fail, size=min(k, len(fail)), replace=False)) if k else []
        mags = [float(rng.uniform(*mag_range)) for _ in support]
        onset = onset or (Onset.ABRUPT if rng.random() < 0.6 else Onset.INCIPIENT)
        return self.episode(rng, support, mags, onset, keep_series=keep_series)

    def sweep_operator(self, unit: float = 1.0, reps: int = 8, seed: int = 12345) -> Operator:
        """Empirical G: inject a unit fault at each failable node, average the residual field -> that column.

        Mirrors the Isaac Lab unit-fault sweep; the recovered G should track operator_true up to the reduction."""
        rng = np.random.default_rng(seed)
        G = np.zeros((self.g.n, self.g.n), np.float32)
        for j in self.g.fail_nodes:
            cols = [self.episode(rng, [j], [unit], keep_series=False).energy() for _ in range(reps)]
            G[:, j] = np.mean(cols, 0)
        return Operator(self.g.name, G, self._names, {"kind": "swept", "unit": unit, "reps": reps})

    def dataset(self, n_healthy: int, n_single: int, n_double: int, seed: int = 0, keep_series: bool = False) -> list[Episode]:
        rng = np.random.default_rng(seed)
        eps = [self.sample(rng, 0, keep_series=keep_series) for _ in range(n_healthy)]
        eps += [self.sample(rng, 1, keep_series=keep_series) for _ in range(n_single)]
        eps += [self.sample(rng, 2, keep_series=keep_series) for _ in range(n_double)]
        rng.shuffle(eps)
        return eps
