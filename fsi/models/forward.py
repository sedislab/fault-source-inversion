"""The healthy forward model f_theta: a command-conditioned graph network that predicts each node's next-step
healthy signal from a short history and its neighbours. Trained on healthy rollouts only (no fault labels), it
turns raw signals into the residual field r = y - yhat that both detection and localization read. Each node may
carry several signal channels (a joint reports position, velocity, torque); residual energy combines them."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .inverse import DenseGCN, norm_adj

def norm_adj_noself(A):
    """Normalized adjacency WITHOUT self-loops: node i aggregates only its neighbours."""
    d = A.sum(1)
    dinv = 1.0 / np.sqrt(np.clip(d, 1e-8, None))
    return (dinv[:, None] * A * dinv[None, :]).astype(np.float32)

def _chan(y):
    return y if y.ndim == 3 else y[..., None]

class ForwardModel(nn.Module):
    """The healthy predictor. `arch` selects how a masked node is reconstructed:
    'attn'  -- self-masked multi-head attention over ALL other nodes (default)
    'graph' -- single 1-hop neighbour aggregation over the no-self-loop adjacency
    and masked=False is the unmasked persistence-style predictor, kept only as a negative control.

    Attention needs the LEARNED node embedding to work at all. `graph.node_features()` returns one-hot node TYPE,
    which is identical for all 7 Franka joints -- so a query built from command+node_feats alone is the same
    vector for every node, and attention cannot tell J1 from J7 (measured: nMSE 0.86 vs 0.67 for the 1-hop
    aggregation, i.e. barely better than predicting zero). The 1-hop path was not winning on inductive bias, it
    was winning because the adjacency was the only thing carrying node identity. With `node_emb` added, attention
    reaches nMSE 0.42 and lifts FSI hard-top1 from 0.227 to 0.385 (chance 0.143), because J1 can read J5 instead
    of only its single chain neighbour J2.

    Multiple horizons are predicted from the same history window. Persistence decays with horizon, so a longer
    horizon carries more fault-sensitive information; the residuals at each horizon become extra channels."""
    def __init__(self, node_feat_dim: int, du: int, hist: int = 8, hidden: int = 64, layers: int = 2,
                 channels: int = 1, masked: bool = True, arch: str = "attn", heads: int = 4,
                 n_nodes: int | None = None, horizons=(1, 4, 8)):
        super().__init__()
        self.hist = hist
        self.channels = channels
        self.masked = masked
        self.arch = arch
        self.heads = heads
        self.horizons = tuple(horizons)
        self.enc = nn.Linear(hist * channels + du + node_feat_dim, hidden)
        self.hist_enc = nn.Linear(hist * channels, hidden)
        self.ctx_enc = nn.Linear(du + node_feat_dim, hidden)
        self.node_emb = nn.Parameter(torch.randn(n_nodes, hidden) * 0.02) if n_nodes else None
        self.q, self.k, self.v, self.o = (nn.Linear(hidden, hidden) for _ in range(4))
        self.gcn = nn.ModuleList([DenseGCN(hidden, hidden) for _ in range(layers)])
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.out = nn.Linear(hidden, channels * len(self.horizons))

    def _attend(self, q, kv):
        """Multi-head attention from the query stream into the per-node history stream, diagonal masked.
        The key/value stream is each node's OWN history and is never mixed across nodes, so no depth or head
        can route node i's history back to node i."""
        B, n, h = q.shape
        dk = h // self.heads
        shape = lambda x: x.view(B, n, self.heads, dk).transpose(1, 2)
        Q, K, V = shape(self.q(q)), shape(self.k(kv)), shape(self.v(kv))
        bias = torch.zeros(n, n, device=q.device, dtype=q.dtype).fill_diagonal_(float("-inf"))
        att = torch.softmax(Q @ K.transpose(-2, -1) / dk ** 0.5 + bias, -1)
        return self.o((att @ V).transpose(1, 2).reshape(B, n, h))

    def forward(self, hist, u_t, A_hat, node_feats, A_nosl=None):
        """hist:(B,n,H,C) u_t:(B,du) node_feats:(n,F) -> (B, n, C * len(horizons)).

        masked=True is the Dedicated-Observer-Scheme analogue: node i is predicted from the OTHER nodes'
        histories, the command and its own static identity, but NOT its own signal history. Without this, a
        one-step predictor with access to y_i is a persistence predictor -- it absorbs the fault at its source and
        drives r_source to ~0, which is precisely what made the source the quietest node instead of the loudest."""
        B, n = hist.shape[:2]
        flat = hist.reshape(B, n, -1)
        ctx = torch.cat([u_t[:, None, :].expand(B, n, -1), node_feats[None].expand(B, -1, -1)], -1)
        if self.masked:
            q = self.ctx_enc(ctx)                      # query seed carries NO signal history of any node
            if self.node_emb is not None:
                q = q + self.node_emb[None]            # the only thing that distinguishes one joint from another
            kv = self.hist_enc(flat)                   # key/value: node j's own history, never mixed across nodes
            mix = (torch.einsum("ij,bjf->bif", A_nosl, kv) if self.arch == "graph" and A_nosl is not None
                   else self._attend(q, kv))
            return self.out(self.mlp(F.gelu(mix + q)))
        h = F.gelu(self.enc(torch.cat([flat, ctx], -1)))
        for g in self.gcn:
            h = g(h, A_hat)
        return self.out(h)

def _windows(episodes, hist, horizons=(1,)):
    """Windows ending at t, with one target per horizon: horizon h predicts y[t+h-1]."""
    H, hz, X, U, Y = hist, tuple(horizons), [], [], []
    hmax = max(hz)
    for e in episodes:
        y, u = _chan(e.y), e.u
        for t in range(H, len(y) - hmax + 1):
            X.append(y[t - H:t].transpose(1, 0, 2)); U.append(u[t])
            Y.append(np.stack([y[t + h - 1] for h in hz], 1))       # (n, len(hz), C)
    return np.asarray(X, np.float32), np.asarray(U, np.float32), np.asarray(Y, np.float32)

def train_forward(model, healthy_episodes, graph, epochs=40, lr=2e-3, batch=256, device="cpu", verbose=False):
    model.to(device).train()
    A_hat = torch.tensor(norm_adj(graph.adjacency()), device=device)
    A_nosl = torch.tensor(norm_adj_noself(graph.adjacency()), device=device)
    nf = torch.tensor(graph.node_features(), device=device)
    X, U, Y = _windows(healthy_episodes, model.hist, model.horizons)
    X = torch.tensor(X, device=device); U = torch.tensor(U, device=device)
    Y = torch.tensor(Y, device=device).reshape(len(Y), Y.shape[1], -1)   # (N, n, len(hz)*C) matches model.out
    N = len(X)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        perm = torch.randperm(N, device=device); tot = 0.0
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            loss = F.mse_loss(model(X[idx], U[idx], A_hat, nf, A_nosl), Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  fwd epoch {ep:3d} mse {tot / N:.5f}")
    return model

def normalize_signals(episodes, healthy):
    """Z-score each node/channel signal by healthy statistics before training f_theta. Raw channels differ by
    orders of magnitude (torque in Nm swamps position in rad), so an unnormalized loss models torque well and
    position badly, inflating the residual floor exactly where propagation lives."""
    Y = np.concatenate([_chan(e.y) for e in healthy], 0)
    mu, sd = Y.mean(0), Y.std(0) + 1e-6
    for e in episodes:
        e.y = ((_chan(e.y) - mu) / sd).astype(np.float32)
    return mu, sd

def standardize_fields(episodes, healthy, clip=False):
    """Z-score each node/channel residual against the HEALTHY residual distribution. f_theta has a non-zero model
    error floor that is command-driven and common to every episode; left in, it dominates the field and every
    fault fingerprint looks alike (operator coherence -> 1). Standardizing removes that floor so the field
    measures deviation-from-normal. Clipped at 0 because a fault raises residuals and the source stays non-negative."""
    H = np.stack([e.r for e in healthy])
    mu, sd = H.mean(0), H.std(0) + 1e-6
    for e in episodes:
        z = (e.r - mu) / sd
        e.r = (np.clip(z, 0, None) if clip else z).astype(np.float32)
    return mu, sd

TIME_BINS = 6

def _parts(mag, w, bins=TIME_BINS, global_thresh=None):
    """The three pieces of a per-node summary, kept separate so `residual_field` can assemble more than one
    window into one field without breaking the column contract: the energy G is fit to, the severity-LINEAR
    block that becomes the fidelity channels, and the onset time (a fraction in [0,1], not severity-linear)."""
    energy = np.sqrt((mag ** 2).mean(0))
    signed_mean = w.mean(0)                                  # (n,C) signed: direction of deviation
    t = np.arange(len(w))[:, None, None].astype(np.float32)
    tc = t - t.mean()
    slope = (tc * w).sum(0) / max(1e-9, float((tc[:, 0, 0] ** 2).sum()))   # (n,C) signed trend
    # Equal-width time bins of the signed residual: the transient's SHAPE, at operator-averageable resolution.
    # array_split tolerates a post window shorter than `bins` (a late onset yields a short window) by emitting
    # empty pieces; an empty bin falls back to the window mean rather than a nan.
    binned = np.stack([(p.mean(0) if len(p) else signed_mean) for p in np.array_split(w, bins, axis=0)], -1)
    h = max(1, len(mag) // 3)
    early, mid, late = mag[:h].mean(0), mag[h:2 * h].mean(0), mag[2 * h:].mean(0)
    thr = global_thresh if global_thresh is not None else 0.5 * (mag.max() + 1e-9)
    above = mag > thr
    first = np.where(above.any(0), above.argmax(0), len(mag)) / max(1, len(mag) - 1)  # WHEN each node crossed
    lin = np.concatenate([signed_mean, slope, binned.reshape(w.shape[1], -1),
                          np.stack([early, mid, late], 1)], 1)
    return energy, lin, first

def _reduce(mag, w, C, global_thresh=None, bins=TIME_BINS):
    """Per-node features, laid out as [energy | severity-linear signed features | onset time].

    SIGNED per-channel statistics are essential: every-feature-non-negative makes any two operator columns have
    non-negative inner product, so coherence is pinned near 1 by arithmetic rather than by physics. Onset time
    uses a GLOBAL threshold so it is comparable across nodes (a per-node threshold measures each node's own peak
    position and cannot rank which node moved first).

    `bins` equal-width time bins of the signed residual are the load-bearing addition. Measured on franka_v6
    with a fixed shared-per-node scorer and softmax over nodes, the same head reaches hard_top1 0.344 reading the
    per-node residual TRAJECTORY and only 0.243 reading the old mean+slope+thirds summary -- a 0.10 gap, larger
    than the entire margin any localizer had over the floor. Mean and slope keep only the zeroth and first moment
    of a transient whose SHAPE is what separates a source (deviates, then the controller responds) from a victim
    (inherits the deviation later and smoothly); binning keeps the shape at a resolution the operator can still
    average over. The layout contract matters downstream: column 0 is the energy G is fit to, the LAST column is
    the onset time (not severity-linear, so it must not become an operator channel), and everything between is a
    fidelity channel -- `operator_channels` is the single place that arithmetic lives."""
    energy, lin, first = _parts(mag, w, bins, global_thresh)
    return np.concatenate([energy[:, None], lin, first[:, None]], 1)

def operator_channels(field_dim: int) -> int:
    """How many of a field's columns are severity-linear fidelity channels: everything except column 0 (the
    energy the scalar operator G is fit to) and the last column (onset time, which is a fraction in [0,1] and
    does NOT scale with severity, so dividing it by severity to build an operator column is meaningless)."""
    return field_dim - 2

@torch.no_grad()
def residual_field(model, episode, graph, window=48, device="cpu", pre_window=24, onset=None, scale=None):
    """Roll f_theta through the episode. Channel 0 is the combined residual energy (the field G is fit to); the
    remaining channels are the per-signal-channel residual RMS, so the inverse can also read a directly-informative
    channel (e.g. a faulted joint's own clipped torque) rather than only the propagated energy.

    `onset` overrides episode.meta["onset_step"] and MUST be supplied for healthy episodes. Healthy episodes carry
    onset_step=0, so without an override they skip the pre-onset self-reference that every faulted episode
    receives -- healthy and faulted fields then come out of two different formulas and the difference alone
    separates them. That asymmetry, not the detector, produced the ANYmal detection recall of 0.997; treating
    both symmetrically gives 0.185. Callers should pass a pseudo-onset drawn from the fault onset distribution."""
    model.eval()
    A_hat = torch.tensor(norm_adj(graph.adjacency()), device=device)
    A_nosl = torch.tensor(norm_adj_noself(graph.adjacency()), device=device)
    nf = torch.tensor(graph.node_features(), device=device)
    y, u, H = _chan(episode.y), episode.u, model.hist
    hz, nch, T0 = model.horizons, y.shape[-1], len(y)
    X = torch.tensor(np.stack([y[t - H:t].transpose(1, 0, 2) for t in range(H, T0)]).astype(np.float32), device=device)
    U = torch.tensor(u[H:].astype(np.float32), device=device)
    pred = model(X, U, A_hat, nf, A_nosl).cpu().numpy()          # (T0-H, n, len(hz)*nch)
    pred = pred.reshape(len(pred), y.shape[1], len(hz), nch)     # window ending at t=H+k predicts y[t+h-1]
    # Align every horizon by TARGET time: horizon h's prediction of y[tau] was made from the window ending at
    # tau-h+1, so each horizon contributes its own residual channel at the same tau. Persistence decays with h,
    # so the longer horizons carry more fault-sensitive evidence than the 1-step residual alone.
    res = [np.zeros_like(y) for _ in hz]
    for j, h in enumerate(hz):
        tau = np.arange(H + h - 1, T0)                            # target times reachable at this horizon
        res[j][tau] = y[tau] - pred[tau - (H + h - 1), :, j, :]
    resid = np.concatenate(res, -1)[H:]                           # (T0-H, n, len(hz)*nch)
    # Keep the 1-step prediction on the episode, aligned to y's full length (the first H steps have no prediction,
    # so they are copied from y and read as zero residual). Time-series baselines such as RCD consume y - yhat
    # directly; without this they silently receive NaN and degrade to naming node 0 on every episode.
    episode.yhat = (y - res[0]).astype(np.float32)
    # DE-NORMALIZE back to physical units before reducing. f_theta needs a z-scored target (raw channels differ by
    # orders of magnitude, so an unnormalized loss models torque well and position badly), but the residual FIELD
    # must not stay in z-units: healthy per-channel std spans 36-80x within a single joint, so z-scoring
    # re-weights a faulted joint's own torque evidence DOWN relative to propagated position/velocity -- exactly
    # the near-diagonal channel that separates cause from victim. Measured on a perfect oracle residual, keeping
    # the field z-scored costs fsi_rbc 0.124 top1 (0.831 -> 0.707): the single largest term in the whole
    # oracle-to-real ladder, larger than the forward-model gap. Normalize for training, de-normalize for the field.
    if scale is not None:
        resid = resid * np.tile(np.asarray(scale, np.float32), len(hz))   # (n, nch) -> (n, len(hz)*nch)
    C, T = resid.shape[-1], len(resid)
    onset = int(episode.meta.get("onset_step", 0) if onset is None else onset) - H
    # ONSET-RELATIVE, NON-OVERLAPPING windows. The old code took a fixed tail `post = resid[-window:]` together
    # with `pre = resid[onset-window:onset]`. With randomized onsets those two windows overlap, so the
    # self-reference subtracted away part of the fault itself, and the "post-onset" window carried pre-onset
    # samples that dilute the fault by duty cycle. The invariant that matters: post NEVER starts before onset.
    # A late onset therefore yields a shorter post window rather than a contaminated one -- `_reduce`'s features
    # are per-sample statistics (means, a least-squares slope, a fractional crossing time), so they stay
    # comparable across lengths.
    if 5 < onset < T - 8:
        post, pre = resid[onset:onset + window], resid[max(0, onset - pre_window):onset]
    else:
        post, pre = resid[-window:], None
    e_post, lin_post, first = _parts(np.sqrt((post ** 2).sum(-1)), post)
    # Within-episode self-reference: the SAME episode's pre-onset segment is a healthy baseline for that exact
    # command, so subtracting it cancels command-specific model error no cross-episode statistic can reach.
    # Keep BOTH the post-onset level and the self-referenced difference. Returning only the difference (what the
    # code did before) throws away how large the deviation is in absolute terms, and the two answer different
    # questions: the difference says "this node changed when the fault started", the level says "this node is
    # far from healthy". A victim inherits a large level with a small difference when it was already loud, and
    # the source often shows the reverse. Measured with a fixed probe on franka_v6, the same head scored 0.344
    # reading the raw trajectory against 0.243 reading the difference-only summary.
    # Column contract, preserved: column 0 is the energy G is fit to, the LAST column is the onset time (a
    # fraction, not severity-linear), and everything between is a fidelity channel -- so `operator_channels`
    # stays field_dim - 2 however many blocks are concatenated here.
    # The SELF-REFERENCED block goes first, so column 0 stays exactly the quantity it has always been. Column 0
    # is not just another feature: `Episode.energy()` reads it, and `is_hard` and `largest_residual` are defined
    # through it, so reordering the blocks would silently redefine which episodes count as hard and make every
    # table incomparable with the ones already published.
    if pre is not None and len(pre) >= 5:
        e_pre, lin_pre, _ = _parts(np.sqrt((pre ** 2).sum(-1)), pre)
        block = [(e_post - e_pre)[:, None], lin_post - lin_pre, e_post[:, None], lin_post]
    else:
        block = [e_post[:, None], lin_post, e_post[:, None], lin_post]
    return np.concatenate(block + [first[:, None]], 1).astype(np.float32)
