"""FSI as a Localizer: fit trains the amortized inverse on labeled (source, field) pairs from simulation; predict
runs one forward pass and returns per-node source probabilities. Detection reuses the same residual field."""
from __future__ import annotations
import numpy as np
import torch
from ..localize.base import Localizer, Prediction
from ..core.types import Operator
from .inverse import AmortizedInverse, train_inverse, norm_adj, fidelity

class FSILocalizer(Localizer):
    name = "fsi"
    needs_operator = True

    def __init__(self, hidden=64, layers=10, epochs=80, lr=2e-3, batch=128, device="cpu", seed=0, mag_w=50.0,
                 cross=2, rank_w=1.0, sup_w=0.3):
        self.cfg = dict(hidden=hidden, layers=layers, epochs=epochs, lr=lr, batch=batch, mag_w=mag_w,
                        cross=cross, rank_w=rank_w, sup_w=sup_w)
        self.device = device
        self.seed = seed
        self.model = None
        self.operator = None

    def fit(self, episodes, graph, operator: Operator | None = None):
        assert operator is not None, "FSI needs the operator G"
        torch.manual_seed(self.seed)
        r_dim = episodes[0].r.shape[1] if episodes[0].r.ndim == 2 else 1
        self.model = AmortizedInverse(graph.node_features().shape[1], r_dim,
                                      self.cfg["hidden"], self.cfg["layers"], cross=self.cfg["cross"])
        train_inverse(self.model, episodes, graph, operator, epochs=self.cfg["epochs"],
                      lr=self.cfg["lr"], batch=self.cfg["batch"], mag_w=self.cfg["mag_w"],
                      rank_w=self.cfg["rank_w"], sup_w=self.cfg["sup_w"], device=self.device)
        self.operator = operator
        self._G = torch.tensor(operator.G, dtype=torch.float32, device=self.device)
        self._A = torch.tensor(norm_adj(graph.adjacency()), device=self.device)
        self._nf = torch.tensor(graph.node_features(), device=self.device)
        return self

    @torch.no_grad()
    def predict(self, episode, graph, operator=None) -> Prediction:
        self.model.eval()
        r = episode.r if episode.r.ndim == 2 else episode.r[:, None]
        scale = np.linalg.norm(r[:, 0]) + 1e-6
        rt = torch.tensor((r / scale)[None].astype(np.float32), device=self.device)
        Gm, rc = fidelity(self.operator, rt, self.device)
        logit, s, _ = self.model(rt, Gm, rc, self._A, self._nf)
        probs = torch.sigmoid(logit)[0].cpu().numpy()
        mag = s[0].cpu().numpy() * scale
        return Prediction(scores=probs, magnitudes=mag, meta={"kind": "amortized_inverse"})
