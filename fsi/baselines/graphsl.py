"""GCNSI / IVGD-style learned inverse: a small GNN that message-passes the residual field into per-node
source logits, trained supervised on (source, field) pairs. Unlike FSI it never uses the physics operator G in a
data-fidelity unrolling and emits no conformal set -- it just learns field->source, so it is a strong but beatable
cause-style competitor. Simplified vs the full GraphSL library: single GCNSI-style head (no IVGD invertible forward
model or validity projection), fields are the reduced residual features rather than diffusion label states."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from ..localize.base import Localizer, Prediction

class _GCNSI(nn.Module):
    def __init__(self, in_dim, hidden, layers):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(in_dim, hidden)])
        for _ in range(layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.head = nn.Linear(hidden, 1)
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = torch.relu(conv(x, edge_index))
        return self.head(x).squeeze(-1)

def _features(ep, static):
    r = ep.r if ep.r.ndim == 2 else ep.r[:, None]
    r = r / (np.abs(r).max() + 1e-8)
    return np.concatenate([static, r.astype(np.float32)], 1)

class GraphSL(Localizer):
    name = "graphsl"

    def __init__(self, hidden=32, epochs=15, layers=3, lr=1e-2, seed=0):
        self.hidden = hidden; self.epochs = epochs; self.layers = layers; self.lr = lr; self.seed = seed
        self.model = None; self.edge_index = None

    def fit(self, episodes, graph, operator=None) -> "GraphSL":
        torch.manual_seed(self.seed)
        n = graph.n
        static = graph.node_features()
        ei = torch.as_tensor(graph.edge_index(), dtype=torch.long)
        self.edge_index = ei
        faults = [ep for ep in episodes if not ep.is_healthy]
        X = torch.stack([torch.as_tensor(_features(ep, static)) for ep in faults])
        Y = torch.stack([torch.as_tensor((ep.source_vector() > 0).astype(np.float32)) for ep in faults])
        big_x = X.reshape(-1, X.shape[-1])
        big_y = Y.reshape(-1)
        big_ei = torch.cat([ei + i * n for i in range(len(faults))], 1)
        self.model = _GCNSI(X.shape[-1], self.hidden, self.layers)
        pos = big_y.mean().clamp(1e-3, 1 - 1e-3)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=(1 - pos) / pos)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss_fn(self.model(big_x, big_ei), big_y).backward()
            opt.step()
        self.model.eval()
        return self

    def predict(self, episode, graph, operator=None) -> Prediction:
        assert self.model is not None, "GraphSL must be fit before predict"
        x = torch.as_tensor(_features(episode, graph.node_features()))
        with torch.no_grad():
            scores = torch.sigmoid(self.model(x, self.edge_index)).numpy()
        return Prediction(scores=scores.astype(float), meta={"kind": "learned_inverse"})
