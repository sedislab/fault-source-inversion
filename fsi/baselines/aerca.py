"""AERCA (Han et al., ICLR 2025): autoencoder + nonlinear-Granger root-cause localizer.

Faithful core: a SENNGC generalized-VAR encoder maps recent history to input-dependent per-lag
coefficient matrices and predicts x_hat_t; the exogenous term u_t = x_hat_t - x_t is pushed toward
N(0,I) by a KL prior while coefficients are elastic-net-sparse and temporally smooth; a second SENNGC
decoder reconstructs x from the (past+current) exogenous window, closing the autoencoder. Trained
unsupervised on healthy series only. Root-cause score is the per-variable |z| of u against its healthy
mean/std, max-reduced over the fault window -- a cause detector that explains reactive victims away.
Simplifications vs the paper: no decoder_prev (past-x reconstruction path) and no POT/EVT threshold
(we use the Gaussian z-score directly); single fixed window length.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from ..localize.base import Localizer, Prediction

def _win(seq, order):
    return seq.unfold(1, order, 1).permute(0, 1, 3, 2).contiguous()

class SENNGC(nn.Module):
    """Per-lag MLP emits an input-dependent coefficient matrix; prediction = sum_k Psi_k(x_{t-k}) x_{t-k}."""
    def __init__(self, p, order, hidden):
        super().__init__()
        self.p, self.order = p, order
        self.nets = nn.ModuleList(
            nn.Sequential(nn.Linear(p, hidden), nn.ReLU(), nn.Linear(hidden, p * p))
            for _ in range(order))

    def forward(self, w):
        b = w.shape[0]
        pred = torch.zeros(b, self.p, device=w.device)
        coeffs = []
        for k in range(self.order):
            xk = w[:, k, :]
            ck = self.nets[k](xk).view(b, self.p, self.p)
            pred = pred + torch.bmm(ck, xk.unsqueeze(-1)).squeeze(-1)
            coeffs.append(ck)
        return pred, torch.stack(coeffs, 1)

class AERCA(Localizer):
    name = "aerca"
    needs_series = True

    def __init__(self, order=4, hidden=32, epochs=12, beta=0.1, lam_sparse=0.02, lam_smooth=0.02,
                 lr=5e-3, ep_batch=16, reduce="mean", seed=0):
        self.order, self.hidden, self.epochs = order, hidden, epochs
        self.beta, self.lam_sparse, self.lam_smooth = beta, lam_sparse, lam_smooth
        self.lr, self.ep_batch, self.reduce, self.seed = lr, ep_batch, reduce, seed
        self.enc = self.dec = None
        self.mu_x = self.sd_x = self.mu_u = self.sd_u = None

    def _series(self, episodes):
        xs = [e.y for e in episodes if e.is_healthy and e.y is not None]
        if not xs:
            raise ValueError("AERCA trains on healthy series; pass keep_series=True healthy episodes")
        return np.stack(xs).astype(np.float32)

    def _standardize(self, X):
        return (X - self.mu_x) / self.sd_x

    def _forward(self, x):
        wins = _win(x, self.order)
        prev = wins[:, :-1]
        tgt = x[:, self.order:]
        b, m = prev.shape[0], prev.shape[1]
        xhat, coeffs = self.enc(prev.reshape(b * m, self.order, self.p))
        xhat = xhat.view(b, m, self.p)
        u = xhat - tgt
        uw = _win(u, self.order)
        rtgt = x[:, 2 * self.order - 1:]
        b2, m2 = uw.shape[0], uw.shape[1]
        xrec, _ = self.dec(uw.reshape(b2 * m2, self.order, self.p))
        xrec = xrec.view(b2, m2, self.p)
        return u, xhat, tgt, xrec, rtgt, coeffs.view(b, m, self.order, self.p, self.p)

    def fit(self, episodes, graph, operator=None):
        X = self._series(episodes)
        self.p = graph.n
        self.mu_x = X.reshape(-1, self.p).mean(0)
        self.sd_x = X.reshape(-1, self.p).std(0) + 1e-6
        Xs = torch.tensor(self._standardize(X))
        torch.manual_seed(self.seed)
        self.enc = SENNGC(self.p, self.order, self.hidden)
        self.dec = SENNGC(self.p, self.order, self.hidden)
        opt = torch.optim.Adam(list(self.enc.parameters()) + list(self.dec.parameters()), lr=self.lr)
        rng = np.random.default_rng(self.seed)
        n = Xs.shape[0]
        for _ in range(self.epochs):
            for i in range(0, n, self.ep_batch):
                idx = rng.permutation(n)[i:i + self.ep_batch]
                x = Xs[idx]
                u, xhat, tgt, xrec, rtgt, coeffs = self._forward(x)
                enc_mse = (xhat - tgt).pow(2).mean()
                dec_mse = (xrec - rtgt).pow(2).mean()
                uf = u.reshape(-1, self.p)
                mu, var = uf.mean(0), uf.var(0) + 1e-6
                kl = 0.5 * (var + mu.pow(2) - 1 - torch.log(var)).mean()
                elastic = coeffs.abs().mean() + coeffs.pow(2).mean()
                smooth = coeffs.diff(dim=1).pow(2).mean()
                loss = enc_mse + dec_mse + self.beta * kl + self.lam_sparse * elastic + self.lam_smooth * smooth
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            u, *_ = self._forward(Xs)
            uf = u.reshape(-1, self.p)
            self.mu_u = uf.mean(0).numpy()
            self.sd_u = (uf.std(0) + 1e-6).numpy()
        return self

    def predict(self, episode, graph, operator=None) -> Prediction:
        if episode.y is None:
            raise ValueError("AERCA needs the time series; sample with keep_series=True")
        x = torch.tensor(self._standardize(episode.y.astype(np.float32)))[None]
        with torch.no_grad():
            u, *_ = self._forward(x)
        u = u[0].numpy()
        z = np.abs(u - self.mu_u) / self.sd_u
        onset = int(episode.meta.get("onset_step", 0))
        w = z[max(0, onset - self.order):]
        w = w if w.shape[0] else z
        scores = w.max(0) if self.reduce == "max" else w.mean(0)
        return Prediction(scores=scores.astype(float), meta={"kind": "cause"})
