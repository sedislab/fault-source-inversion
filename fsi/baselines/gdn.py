"""GDN - Graph Deviation Network (Deng & Hooi, AAAI 2021): a learned embedding-kNN graph-attention forecaster
scored by per-node normalized forecast deviation. Trains unsupervised on healthy episodes (forecast next step);
at predict time each node is scored by its mean normalized deviation over the fault window and the most-deviant
node is named
(victim-style, so cvv <= ~0.5 by design). Simplified vs the paper: a per-node shared output head instead of the
stacked all-node head, and a static top-k neighbour mask recomputed from the current embeddings rather than a
fully differentiable learned graph. The core mechanisms (embedding kNN graph, attention forecasting from
neighbours, normalized deviation localization) are faithful."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from ..localize.base import Localizer, Prediction

class _GDNNet(nn.Module):
    def __init__(self, n, sw, dim, topk):
        super().__init__()
        self.n, self.topk = n, topk
        self.emb = nn.Embedding(n, dim)
        self.lin = nn.Linear(sw, dim)
        self.att = nn.Linear(4 * dim, 1)
        self.out = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))
        self.leaky = nn.LeakyReLU(0.2)

    def _mask(self):
        e = torch.nn.functional.normalize(self.emb.weight, dim=1)
        sim = e @ e.t()
        k = min(self.topk, self.n - 1)
        idx = sim.topk(k + 1, dim=1).indices
        return torch.zeros_like(sim).scatter_(1, idx, 1.0)

    def forward(self, x):
        b = x.shape[0]
        emb = self.emb.weight
        h = self.lin(x)
        g = torch.cat([emb.unsqueeze(0).expand(b, -1, -1), h], dim=2)
        gi = g.unsqueeze(2).expand(b, self.n, self.n, -1)
        gj = g.unsqueeze(1).expand(b, self.n, self.n, -1)
        score = self.leaky(self.att(torch.cat([gi, gj], dim=3)).squeeze(-1))
        score = score.masked_fill(self._mask().unsqueeze(0) == 0, float("-inf"))
        alpha = torch.softmax(score, dim=2)
        z = torch.relu(torch.einsum("bij,bjd->bid", alpha, h))
        return self.out(emb.unsqueeze(0) * z).squeeze(-1)

class GDN(Localizer):
    """Unsupervised graph-deviation forecaster; localizes to the loudest normalized-deviation node."""
    name = "gdn"
    needs_series = True

    def __init__(self, sw=16, dim=32, topk=8, epochs=12, lr=1e-2, batch=256, stride=2, seed=0):
        self.sw, self.dim, self.topk = sw, dim, topk
        self.epochs, self.lr, self.batch, self.stride, self.seed = epochs, lr, batch, stride, seed
        self.net = self.mu = self.sd = self.err_med = self.err_iqr = None

    def _windows(self, y, stride=1):
        T = y.shape[0]
        ts = range(self.sw, T, stride)
        X = np.stack([y[t - self.sw:t].T for t in ts]).astype(np.float32)
        Y = np.stack([y[t] for t in ts]).astype(np.float32)
        return X, Y

    def fit(self, episodes, graph, operator=None):
        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        healthy = [e for e in episodes if e.is_healthy and e.y is not None]
        if not healthy:
            raise ValueError("GDN trains on healthy episodes with series; none found (sample with keep_series=True)")
        ys = np.concatenate([e.y for e in healthy], 0)
        self.mu = ys.mean(0, keepdims=True).astype(np.float32)
        self.sd = (ys.std(0, keepdims=True) + 1e-6).astype(np.float32)
        Xs, Ys = zip(*[self._windows((e.y - self.mu) / self.sd, self.stride) for e in healthy])
        X = torch.tensor(np.concatenate(Xs, 0)); Y = torch.tensor(np.concatenate(Ys, 0))
        self.net = _GDNNet(graph.n, self.sw, self.dim, self.topk)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        for _ in range(self.epochs):
            perm = torch.randperm(X.shape[0])
            for i in range(0, len(perm), self.batch):
                idx = perm[i:i + self.batch]
                opt.zero_grad()
                loss = loss_fn(self.net(X[idx]), Y[idx])
                loss.backward(); opt.step()
        self.net.eval()
        with torch.no_grad():
            err = (Y - self.net(X)).abs().numpy()
        self.err_med = np.median(err, 0)
        q75, q25 = np.percentile(err, 75, 0), np.percentile(err, 25, 0)
        self.err_iqr = (q75 - q25) + 1e-6
        return self

    def predict(self, episode, graph, operator=None):
        if episode.y is None:
            raise ValueError("GDN needs_series: episode.y is None (sample with keep_series=True)")
        y = (episode.y - self.mu) / self.sd
        X, Yt = self._windows(y, 1)
        with torch.no_grad():
            pred = self.net(torch.tensor(X)).numpy()
        dev = np.abs(np.abs(Yt - pred) - self.err_med) / self.err_iqr
        onset = episode.meta.get("onset_step", 0)
        rows = np.array([t for t in range(self.sw, episode.y.shape[0])]) >= onset
        dev = dev[rows] if rows.any() else dev
        return Prediction(scores=dev.mean(0).astype(float), meta={"kind": "victim"})
