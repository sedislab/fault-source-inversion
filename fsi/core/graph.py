"""The robot as a structural graph: nodes are components that can fail, edges are coupling channels."""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import numpy as np
import networkx as nx

class NodeType(str, Enum):
    ACTUATOR = "actuator"
    SENSOR = "sensor"
    LINK = "link"
    BODY = "body"
    SOFTWARE = "software"

class EdgeKind(str, Enum):
    PHYSICAL = "physical"
    ALLOCATION = "allocation"
    DATAFLOW = "dataflow"

_FAILABLE = {NodeType.ACTUATOR, NodeType.SENSOR, NodeType.LINK, NodeType.SOFTWARE}

@dataclass
class Node:
    id: int
    name: str
    type: NodeType
    signal_dim: int = 1
    can_fail: bool | None = None
    meta: dict = field(default_factory=dict)
    def __post_init__(self):
        self.type = NodeType(self.type)
        if self.can_fail is None:
            self.can_fail = self.type in _FAILABLE

@dataclass
class Edge:
    u: int
    v: int
    kind: EdgeKind = EdgeKind.PHYSICAL
    directed: bool = False
    def __post_init__(self):
        self.kind = EdgeKind(self.kind)

class StructuralGraph:
    """Nodes wired by coupling edges. Physical edges are symmetric; dataflow/allocation are directed."""
    def __init__(self, nodes: list[Node], edges: list[Edge], name: str = ""):
        self.name = name
        self.nodes = sorted(nodes, key=lambda x: x.id)
        self.edges = edges
        assert [n.id for n in self.nodes] == list(range(len(self.nodes))), "node ids must be 0..n-1"
        self._hop = None

    @property
    def n(self) -> int:
        return len(self.nodes)

    @property
    def fail_nodes(self) -> list[int]:
        return [n.id for n in self.nodes if n.can_fail]

    def node(self, i: int) -> Node:
        return self.nodes[i]

    def names(self) -> list[str]:
        return [n.name for n in self.nodes]

    def adjacency(self, symmetric: bool = True) -> np.ndarray:
        A = np.zeros((self.n, self.n), np.float32)
        for e in self.edges:
            A[e.u, e.v] = 1.0
            if symmetric or not e.directed:
                A[e.v, e.u] = 1.0
        return A

    def edge_index(self, symmetric: bool = True) -> np.ndarray:
        src, dst = [], []
        for e in self.edges:
            src.append(e.u); dst.append(e.v)
            if symmetric or not e.directed:
                src.append(e.v); dst.append(e.u)
        return np.asarray([src, dst], np.int64)

    def hop_distance(self) -> np.ndarray:
        """Shortest-path hops between every node pair; disconnected pairs get n (a finite large value)."""
        if self._hop is None:
            g = nx.Graph()
            g.add_nodes_from(range(self.n))
            g.add_edges_from((e.u, e.v) for e in self.edges)
            D = np.full((self.n, self.n), float(self.n), np.float32)
            for i, dmap in nx.all_pairs_shortest_path_length(g):
                for j, d in dmap.items():
                    D[i, j] = d
            self._hop = D
        return self._hop

    def node_features(self) -> np.ndarray:
        """Per-node static features: type one-hot, can_fail flag, log signal_dim."""
        types = list(NodeType)
        X = np.zeros((self.n, len(types) + 2), np.float32)
        for nd in self.nodes:
            X[nd.id, types.index(nd.type)] = 1.0
            X[nd.id, len(types)] = float(nd.can_fail)
            X[nd.id, len(types) + 1] = np.log1p(nd.signal_dim)
        return X

    def descendants(self, i: int) -> set[int]:
        """Nodes reachable from i following directed edges (physical edges count both ways)."""
        g = nx.DiGraph()
        g.add_nodes_from(range(self.n))
        for e in self.edges:
            g.add_edge(e.u, e.v)
            if not e.directed:
                g.add_edge(e.v, e.u)
        return nx.descendants(g, i)

    def to_dict(self) -> dict:
        return {"name": self.name,
                "nodes": [asdict(n) | {"type": n.type.value} for n in self.nodes],
                "edges": [asdict(e) | {"kind": e.kind.value} for e in self.edges]}

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict) -> StructuralGraph:
        nodes = [Node(**n) for n in d["nodes"]]
        edges = [Edge(**e) for e in d["edges"]]
        return cls(nodes, edges, d.get("name", ""))

    @classmethod
    def load(cls, path: str | Path) -> StructuralGraph:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def __repr__(self) -> str:
        return f"StructuralGraph({self.name!r}, n={self.n}, edges={len(self.edges)}, failable={len(self.fail_nodes)})"
