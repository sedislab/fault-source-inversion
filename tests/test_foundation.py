"""Foundation checks: graph contracts, synthetic propagation, cause-victim gap, operator recovery."""
import numpy as np
from fsi.core import StructuralGraph, NodeType
from fsi.synth import SyntheticSystem, chain_graph, tree_graph, star_graph

def _sys(g, **kw):
    return SyntheticSystem(g, **kw)

def test_graph_contracts():
    for g in (chain_graph(4), tree_graph(4), star_graph(4)):
        assert [n.id for n in g.nodes] == list(range(g.n))
        D = g.hop_distance()
        assert D.shape == (g.n, g.n) and (D >= 0).all() and (np.diag(D) == 0).all()
        assert len(g.fail_nodes) > 0
        assert StructuralGraph.from_dict(g.to_dict()).n == g.n

def _e(n, j):
    s = np.zeros(n, np.float32); s[j] = 1.0; return s

def test_residual_matches_field():
    g = chain_graph(4); sys = _sys(g, noise=0.0, seed=1)
    rng = np.random.default_rng(0)
    j = g.fail_nodes[2]
    ep = sys.episode(rng, [j], [1.0])
    field = sys._field(_e(g.n, j))
    assert np.corrcoef(ep.energy(), field)[0, 1] > 0.98

def test_cause_victim_gap():
    for gf in (chain_graph(5), tree_graph(4), star_graph(4)):
        sys = _sys(gf, seed=2)
        rng = np.random.default_rng(3)
        miss = 0; total = 200
        for _ in range(total):
            ep = sys.sample(rng, 1, keep_series=False)
            src = ep.support()[0]
            if int(np.argmax(ep.energy())) != src:
                miss += 1
        gap = miss / total
        assert gap > 0.35, f"{gf.name}: cause-victim gap too small ({gap:.2f}) to test the thesis"

def test_operator_recovery():
    g = chain_graph(5); sys = _sys(g, noise=0.03, seed=4)
    Gt = sys.operator_true().G
    Gs = sys.sweep_operator(reps=16).G
    fail = g.fail_nodes
    cos = [float(Gt[:, j] @ Gs[:, j] / (np.linalg.norm(Gt[:, j]) * np.linalg.norm(Gs[:, j]) + 1e-9)) for j in fail]
    assert min(cos) > 0.95, f"swept columns diverge from analytic: min cos {min(cos):.3f}"

def test_inversion_beats_loudest():
    """The testbed must exhibit the thesis: solving r=Gs finds the cause where the loudest node fails."""
    from scipy.optimize import nnls
    g = tree_graph(4); sys = _sys(g, seed=7)
    G = sys.sweep_operator(reps=24).G
    rng = np.random.default_rng(8)
    inv, loud = [], []
    for _ in range(200):
        ep = sys.sample(rng, 1, keep_series=False); src = ep.support()[0]
        shat, _ = nnls(G, ep.energy())
        inv.append(int(np.argmax(shat)) == src)
        loud.append(int(np.argmax(ep.energy())) == src)
    assert np.mean(inv) > 0.9 and np.mean(inv) - np.mean(loud) > 0.4

def test_coherence_and_multifault():
    g = tree_graph(4); sys = _sys(g, seed=5)
    op = sys.operator_true()
    assert 0.0 <= op.coherence() <= 1.0
    rng = np.random.default_rng(6)
    ep = sys.sample(rng, 2, keep_series=False)
    assert ep.k == 2 and len(set(ep.support())) == 2

if __name__ == "__main__":
    import sys, traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for fn in fns:
        try:
            fn(); ok += 1; print(f"PASS {fn.__name__}")
        except Exception:
            print(f"FAIL {fn.__name__}"); traceback.print_exc()
    # report the cause-victim gap explicitly, it is the project's gating number
    for gf in (chain_graph(5), tree_graph(4), star_graph(4)):
        sys_ = SyntheticSystem(gf, seed=7); rng = np.random.default_rng(8)
        eps = [sys_.sample(rng, 1, keep_series=False) for _ in range(300)]
        gap = np.mean([int(np.argmax(e.energy())) != e.support()[0] for e in eps])
        print(f"cause-victim gap [{gf.name}]: {gap:.3f}")
    print(f"{ok}/{len(fns)} tests passed")
