"""TranAD (Tuli et al., VLDB 2022): a transformer that reconstructs the multivariate signal with self-conditioning
and a two-phase (adversarial-focus) pass; per-channel reconstruction deviation over the fault window ranks the
source. Non-graph, victim-style -- the "does structure even help?" ablation. Trained UNSUPERVISED on healthy y.
Simplification: the GAN min-max between the two decoders is realized as the authors' epoch-weighted two-phase
objective (self-conditioning + both reconstructions retained), the form in their reference implementation."""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
from ..localize.base import Localizer, Prediction

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(1))
    def forward(self, x):
        return x + self.pe[: x.size(0)]

class _TranADNet(nn.Module):
    def __init__(self, feats, ff=16):
        super().__init__()
        self.feats = feats
        d = 2 * feats
        self.pos = _PositionalEncoding(d)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, feats, ff, dropout=0.1), 1)
        self.decoder1 = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, feats, ff, dropout=0.1), 1)
        self.decoder2 = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, feats, ff, dropout=0.1), 1)
        self.fcn = nn.Sequential(nn.Linear(d, feats), nn.Sigmoid())
    def _encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2) * math.sqrt(self.feats)
        memory = self.encoder(self.pos(src))
        return tgt.repeat(1, 1, 2), memory
    def forward(self, src, tgt):
        c = torch.zeros_like(src)
        x1 = self.fcn(self.decoder1(*self._encode(src, c, tgt)))
        c = (x1 - src) ** 2
        x2 = self.fcn(self.decoder2(*self._encode(src, c, tgt)))
        return x1, x2

class TranAD(Localizer):
    name = "tranad"
    needs_series = True

    def __init__(self, epochs=12, hidden=16, window=10, lr=0.01, batch=256, max_windows=6000, seed=0):
        self.epochs = epochs; self.hidden = hidden; self.w = window
        self.lr = lr; self.batch = batch; self.max_windows = max_windows; self.seed = seed
        self.net = None; self.lo = None; self.hi = None

    def _norm(self, y):
        return np.clip((y - self.lo) / (self.hi - self.lo + 1e-8), 0.0, 1.0).astype(np.float32)

    def _windows(self, y):
        pad = np.repeat(y[:1], self.w - 1, 0)
        yp = np.concatenate([pad, y], 0)
        return np.stack([yp[i:i + self.w] for i in range(y.shape[0])]).astype(np.float32)

    def fit(self, episodes, graph, operator=None):
        healthy = [e for e in episodes if e.is_healthy]
        if not healthy or healthy[0].y is None:
            raise ValueError("TranAD trains unsupervised on healthy episodes with time-series (keep_series=True)")
        torch.manual_seed(self.seed)
        Y = np.stack([e.y for e in healthy]).astype(np.float32)
        self.lo = Y.reshape(-1, graph.n).min(0); self.hi = Y.reshape(-1, graph.n).max(0)
        win = np.concatenate([self._windows(self._norm(y)) for y in Y], 0)
        rng = np.random.default_rng(self.seed)
        if len(win) > self.max_windows:
            win = win[rng.choice(len(win), self.max_windows, replace=False)]
        X = torch.from_numpy(win)
        self.net = _TranADNet(graph.n, self.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        mse = nn.MSELoss()
        self.net.train()
        for epoch in range(self.epochs):
            nfac = epoch + 1
            perm = torch.randperm(len(X))
            for i in range(0, len(X), self.batch):
                b = X[perm[i:i + self.batch]].permute(1, 0, 2)
                tgt = b[-1:].clone()
                o1, o2 = self.net(b, tgt)
                loss = (1 / nfac) * mse(o1, tgt) + (1 - 1 / nfac) * mse(o2, tgt)
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    def predict(self, episode, graph, operator=None) -> Prediction:
        if episode.y is None:
            raise ValueError("TranAD needs the episode time-series (keep_series=True)")
        assert self.net is not None, "call fit before predict"
        y = self._norm(episode.y.astype(np.float32))
        onset = int(episode.meta.get("onset_step", 0))
        src = torch.from_numpy(self._windows(y)).permute(1, 0, 2)
        tgt = src[-1:].clone()
        self.net.eval()
        with torch.no_grad():
            o1, o2 = self.net(src, tgt)
        err = (0.5 * (o1 - tgt) ** 2 + 0.5 * (o2 - tgt) ** 2)[0].numpy()
        scores = err[onset:].mean(0).astype(float)
        return Prediction(scores=scores, meta={"kind": "reconstruction", "phase": "two"})
