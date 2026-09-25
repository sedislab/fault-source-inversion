"""The amortized inverse h_psi: unrolled proximal-gradient on the KNOWN operator G, with a graph message-passing
prior per layer and dual support/magnitude heads. The data-fidelity gradient keeps the physics of G in every
layer, so the net inverts propagation rather than learning a black-box field->source map; conditioning the prior
on the graph is what lets one h_psi serve many embodiments."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def norm_adj(A: np.ndarray) -> np.ndarray:
    """Symmetric-normalized adjacency with self loops: D^-1/2 (A+I) D^-1/2."""
    A = A + np.eye(A.shape[0], dtype=A.dtype)
    d = A.sum(1)
    dinv = 1.0 / np.sqrt(np.clip(d, 1e-8, None))
    return (dinv[:, None] * A * dinv[None, :]).astype(np.float32)

class DenseGCN(nn.Module):
    """One graph-conv over a dense normalized adjacency, batched: H' = act(A_hat H W_n + H W_s + b)."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.wn = nn.Linear(in_dim, out_dim, bias=False)
        self.ws = nn.Linear(in_dim, out_dim)
    def forward(self, H, A_hat):
        return F.gelu(torch.einsum("ij,bjf->bif", A_hat, self.wn(H)) + self.ws(H))

def matched_filter(Gm, rct, eps=1e-8):
    """Closed-form inversion evidence per node, the part of the physics that CANNOT be optimized away.

    For each fidelity channel c and candidate node j, project the field onto that node's own operator column:

        cos_cj = <g_cj, r_c> / (||g_cj|| ||r_c||)        the matched filter / directional-residual angle test
        rbc_cj = <g_cj, r_c>^2 / <g_cj, g_cj>            reconstruction-based contribution (Alcala & Qin 2009)

    Ranking by residual magnitude is a contribution plot and provably smears a fault onto correlated healthy
    nodes; RBC carries a Cauchy-Schwarz guarantee that the faulty node scores highest for a large enough single
    fault. Computing these directly, rather than hoping an unrolled solver rediscovers them, is what makes the
    G-dependence survive training -- `probe_magw` measured std(mag_skip*s) at 0.0000 on Franka and 0.0056 on
    ANYmal against a logit std of 1.65 / 12.5, i.e. the entire unrolled proximal-gradient path was contributing
    nothing to the ranking on either embodiment, at every mag_w. Returns (B, n, 5): the energy channel's cos and
    rbc, then the mean / max / argmax-frequency of cos over the fidelity channels.

    Gm:(C,n,n)  rct:(B,C,n)."""
    g_norm = Gm.pow(2).sum(1).clamp_min(eps).sqrt()                     # (C,n) column norms
    proj = torch.einsum("cmn,bcm->bcn", Gm, rct)                        # (B,C,n) <g_cj, r_c>
    r_norm = rct.pow(2).sum(-1).clamp_min(eps).sqrt()                   # (B,C)
    cos = proj / (g_norm[None] * r_norm[..., None] + eps)               # (B,C,n)
    # RBC_j = <g_j,r>^2 / <g_j,g_j> = (proj/||g_j||)^2 = (cos_j * ||r||)^2. Because it is immediately rescaled
    # by its max over nodes, the ||r||^2 factor cancels -- so compute it from `cos`, which is bounded in [-1,1],
    # rather than from `proj`, which is not. Squaring an unnormalized projection overflows float32 whenever a
    # field row is large relative to the energy channel it was scaled by (FSILocalizer.predict divides by
    # ||r[:,0]|| only), and an inf here becomes a nan score: on the ANYmal sensor corpus this produced
    # "non-finite scores on 640/640 episodes" and the harness -- correctly -- refused the whole run.
    rbc = cos.pow(2)
    rbc = rbc / (rbc.amax(-1, keepdim=True) + eps)                      # scale-free within episode
    hit = F.one_hot(cos.argmax(-1), cos.shape[-1]).to(cos.dtype).mean(1)  # (B,n) how often j wins a channel
    out = torch.stack([cos[:, 0], rbc[:, 0], cos.mean(1), cos.amax(1), hit], -1)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


class CrossNode(nn.Module):
    """One round of competition between candidate nodes: attention over all nodes plus a graph convolution.

    Localization is a decision AMONG nodes -- "is J3 the cause given what J2 and J4 look like" -- but the old
    readout was `nn.Sequential` applied per node, so the only cross-node information reaching the score was the
    (dead) unrolled `s`. A per-node MLP cannot represent "this node deviated first" or "this node's response is
    unexplained by its neighbours", which is exactly the evidence a hard episode turns on."""
    def __init__(self, hidden, heads=4):
        super().__init__()
        self.heads = heads
        self.q, self.k, self.v, self.o = (nn.Linear(hidden, hidden) for _ in range(4))
        self.gcn = DenseGCN(hidden, hidden)
        self.mlp = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, A_hat):
        B, n, d = h.shape
        dk = d // self.heads
        shape = lambda x: x.view(B, n, self.heads, dk).transpose(1, 2)
        Q, K, V = shape(self.q(h)), shape(self.k(h)), shape(self.v(h))
        att = torch.softmax(Q @ K.transpose(-2, -1) / dk ** 0.5, -1)
        a = self.o((att @ V).transpose(1, 2).reshape(B, n, d))
        return self.norm(h + self.mlp(torch.cat([a, self.gcn(h, A_hat)], -1)))


class AmortizedInverse(nn.Module):
    def __init__(self, node_feat_dim: int, r_dim: int = 3, hidden: int = 64, layers: int = 10, cross: int = 2,
                 heads: int = 4):
        super().__init__()
        self.L = layers
        self.r_dim = r_dim
        self.eta = nn.Parameter(torch.full((layers,), 0.2))
        self.tau = nn.Parameter(torch.full((layers,), -4.0))
        prior_in = 2 + r_dim + node_feat_dim
        self.priors = nn.ModuleList([DenseGCN(prior_in, hidden) for _ in range(layers)])
        self.prior_out = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(layers)])
        # 1 unrolled magnitude + 5 matched-filter statistics + r + its within-episode z-score + node features.
        # The z-score is over the OTHER nodes of the same episode, so "loud for this episode" and "loud for this
        # node" stop being the same number -- the per-node MLP could not tell them apart.
        self.read = nn.Sequential(nn.Linear(1 + 5 + 2 * r_dim + node_feat_dim, hidden), nn.GELU(),
                                  nn.Linear(hidden, hidden), nn.GELU())
        self.cross = nn.ModuleList([CrossNode(hidden, heads) for _ in range(cross)])
        self.support_head = nn.Linear(hidden, 1)
        self.mag_skip = nn.Parameter(torch.tensor(3.0))
        # Input standardization fitted on the training fields, held as buffers so it travels with the model.
        # The field is already z-scored against healthy episodes, but that division is by `sd + 1e-6` per
        # node/channel: a channel a healthy robot never varies (a foot's swing timer, a contact flag) has
        # sd ~ 0, so its z runs to 1e5-1e6 while its neighbours sit at O(1). Feeding that straight into an MLP
        # is what makes the readout diverge; the residual FIELD must stay honest, but the model's view of it
        # does not have to be unscaled.
        self.register_buffer("r_mu", torch.zeros(r_dim))
        self.register_buffer("r_sd", torch.ones(r_dim))

    def norm_r(self, r):
        return torch.clamp((r - self.r_mu) / self.r_sd, -10.0, 10.0)

    def forward(self, r, Gm, rc, A_hat, node_feats):
        """r:(B,n,r_dim) all residual features; Gm:(C,n,n) per-channel operators; rc:(B,n,C) the matching
        fidelity channels. The gradient sums each channel's own inverse-source residual, so a near-diagonal
        (direct) channel and a propagating channel both pull on s."""
        B, n, _ = r.shape
        nf = node_feats.unsqueeze(0).expand(B, -1, -1)
        rct = rc.permute(0, 2, 1)
        rn = self.norm_r(r)          # every learned block reads the standardized field, never the raw one
        s = torch.zeros(B, n, device=r.device)
        iters = []
        # Proximal gradient is stable only for a step below 2/L, L the largest eigenvalue of sum_c G_c^T G_c.
        # A FIXED eta = 0.2 is not a step size, it is a bet on the operator's scale -- and the bet fails in both
        # directions. With the old clipped, non-negative, 21-channel operator the iteration collapsed and the
        # physics branch was measured dead (std(mag_skip*s) = 0.0000 on Franka). With a signed 199-channel
        # operator the same 0.2 is far above 2/L and the iteration diverges geometrically: measured s reaching
        # 1.8e23 after 10 layers on the ANYmal sensor corpus, which became a non-finite logit and cost the whole
        # run. ||Gm||_F^2 upper-bounds L, so eta/L keeps a learned eta in the stable region for any operator.
        lip = Gm.pow(2).sum().clamp_min(1e-8)
        for k in range(self.L):
            resid = torch.einsum("cmn,bn->bcm", Gm, s) - rct
            grad = torch.einsum("bcm,cmn->bn", resid, Gm) / lip     # in step units, so it is O(1) like s
            z = s - self.eta[k] * grad
            # Both iterate features the prior sees are already in step units, so they are O(1) whatever the
            # operator's scale -- feeding the raw gradient here was enough on its own to drive the untrained
            # iterate to the clamp.
            feats = torch.cat([z.unsqueeze(-1), grad.unsqueeze(-1), rn, nf], -1)
            delta = self.prior_out[k](self.priors[k](feats, A_hat)).squeeze(-1)
            z = z + delta
            # The GNN prior's delta is unbounded and is added straight onto the iterate, so a divergent layer
            # can still be manufactured downstream of a correct step size. Clamp the iterate to a generous
            # multiple of the field scale: `_tensors` normalizes r so a well-posed s is O(1).
            s = torch.clamp(F.relu(z - F.softplus(self.tau[k])), max=1e2)
            iters.append(s)
        phys = matched_filter(Gm, rct)
        rz = torch.clamp((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6), -10.0, 10.0)
        h = self.read(torch.cat([s.unsqueeze(-1), phys, rn, rz, nf], -1))
        for blk in self.cross:
            h = blk(h, A_hat)
        support = self.support_head(h).squeeze(-1) + self.mag_skip * s
        return support, s, iters

def _tensors(episodes, graph, device):
    R = np.stack([e.r if e.r.ndim == 2 else e.r[:, None] for e in episodes]).astype(np.float32)
    Sm = np.stack([e.source_vector() for e in episodes]).astype(np.float32)
    Sup = (Sm > 0).astype(np.float32)
    scale = np.linalg.norm(R[..., 0], axis=1, keepdims=True) + 1e-6
    R = R / scale[..., None]
    Sm = Sm / scale
    return (torch.tensor(R, device=device), torch.tensor(Sm, device=device), torch.tensor(Sup, device=device))

def fidelity(operator, R, device):
    """(Gm (C,n,n), rc (N,n,C)): per-channel operators when available, else the single energy operator."""
    if getattr(operator, "G_multi", None) is not None:
        Gm = torch.tensor(operator.G_multi, dtype=torch.float32, device=device)
        return Gm, R[..., 1:1 + Gm.shape[0]]
    Gm = torch.tensor(operator.G, dtype=torch.float32, device=device)[None]
    return Gm, R[..., 0:1]

def train_inverse(model, episodes, graph, operator, epochs=60, lr=2e-3, batch=128, mag_w=50.0,
                  deep=0.3, device="cpu", pos_weight=None, verbose=False, rank_w=1.0, sup_w=0.3):
    """mag_w defaults high on purpose. `_tensors` divides the source magnitude by ||r[:,0]|| to make the net
    scale-invariant (if r = G s then scaling r scales s, so the ratio must be preserved), but that leaves the
    normalized target at ~0.14: a constant s=0 predictor already scores mag-MSE 0.0032 per node against a total
    loss of ~0.39, so at mag_w=1 the magnitude term is under 1% of the objective. The optimizer duly drove s to
    zero -- measured std of `mag_skip * s` across candidates was 0.001 versus 1.433 for the full logit -- which
    silently switched OFF the unrolled proximal-gradient path. With the physics branch dead, `self.read`
    collapses to a per-node MLP over that node's own features: a cross-node-blind energy ranker, which is exactly
    the behaviour observed (per-episode Spearman with residual energy +0.33, argmax equals the loudest node 60%
    of the time). Weighting the magnitude term back to parity is what keeps G in the ranking. That reweighting
    did not in fact revive it -- a later sweep found std(mag_skip*s) at 0.0000/0.0056 for every mag_w on both
    embodiments -- which is why `matched_filter` now supplies the G-dependence in closed form instead.

    `rank_w` weights a LISTWISE loss: soft cross-entropy of the score distribution over candidates against the
    true support. Per-node BCE (`sup_w`) asks "is this node a source", one node at a time, and is already nearly
    satisfied by calling every node healthy -- it never asks the model to rank the cause ABOVE the victim, which
    is the only question a hard episode poses. The listwise term makes the loudest victim an explicit negative
    for the source's probability mass, and it generalizes to k>1 through the uniform-on-support target, so
    multi-fault support recovery is not sacrificed to get it."""
    model.to(device).train()
    A_hat = torch.tensor(norm_adj(graph.adjacency()), device=device)
    nf = torch.tensor(graph.node_features(), device=device)
    R, Sm, Sup = _tensors(episodes, graph, device)
    flat = R.reshape(-1, R.shape[-1])
    model.r_mu.copy_(flat.mean(0))
    model.r_sd.copy_(flat.std(0).clamp_min(1e-3))
    Gm, RC = fidelity(operator, R, device)
    N = R.shape[0]
    fail = torch.tensor(graph.fail_nodes, device=device)
    pw = torch.tensor(pos_weight if pos_weight else max(1.0, graph.n / 3.0), device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        perm = torch.randperm(N, device=device)
        tot = 0.0
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            sup_logit, s, iters = model(R[idx], Gm, RC[idx], A_hat, nf)
            tgt = Sm[idx][:, fail]
            lsup = F.binary_cross_entropy_with_logits(sup_logit[:, fail], Sup[idx][:, fail], pos_weight=pw)
            lmag = ((s[:, fail] - tgt) ** 2).mean()
            ldeep = sum(((it[:, fail] - tgt) ** 2).mean() for it in iters[-3:]) / 3
            # Listwise: uniform over the true support, skipping fault-free episodes (no support to rank towards).
            q = Sup[idx][:, fail]
            hasq = q.sum(1) > 0
            if bool(hasq.any()):
                qn = q[hasq] / q[hasq].sum(1, keepdim=True)
                lrank = -(qn * F.log_softmax(sup_logit[:, fail][hasq], dim=1)).sum(1).mean()
            else:
                lrank = sup_logit.sum() * 0.0
            loss = sup_w * lsup + rank_w * lrank + mag_w * lmag + deep * ldeep
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d} loss {tot / N:.4f}")
    return model
