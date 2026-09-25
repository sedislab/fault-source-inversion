"""Metrics validation: hard_top1 must separate a cause-finder (direct solve) from a victim rule (largest
residual), and W1G / ranking / detection must behave. Uses the synthetic testbed as ground truth.

hard_top1 replaces CvV (retired 2026-07-30: a scorer ranking nodes by NEGATIVE residual energy scored CvV 1.000
while localizing almost nothing). hard_top1 is top-1 restricted to episodes where the loudest candidate is not
the source, so chance is 1/|candidates| and the loudest-node rule scores exactly 0 by construction."""
import numpy as np
from fsi.synth import SyntheticSystem, tree_graph, chain_graph
from fsi.baselines import LargestResidual, DirectSolve
from fsi.metrics import evaluate_localization, evaluate_detection, graph_wasserstein

def _data(gf, n=300, seed=7):
    sys = SyntheticSystem(gf, seed=seed)
    op = sys.sweep_operator(reps=24)
    rng = np.random.default_rng(11)
    eps = [sys.sample(rng, 1, keep_series=True) for _ in range(n)]
    heal = [sys.sample(rng, 0, keep_series=True) for _ in range(80)]
    return sys, op, eps, heal

def test_hard_top1_separates_cause_from_victim():
    for gf in (chain_graph(5), tree_graph(4)):
        sys, op, eps, _ = _data(gf)
        loud = LargestResidual(); inv = DirectSolve(lam=0.02)
        m_loud = evaluate_localization(loud.predict_many(eps, gf, op), eps, gf, op)
        m_inv = evaluate_localization(inv.predict_many(eps, gf, op), eps, gf, op)
        assert m_loud["n_hard"] > 0, "no cause-over-victim episodes: hard_top1 would be vacuous"
        # the victim rule cannot win a hard episode by construction; the cause-finder must win most of them
        assert m_loud["hard_top1"] <= 1e-9, f"loudest hard_top1 must be 0, got {m_loud['hard_top1']:.2f}"
        assert m_inv["hard_top1"] > 0.85, f"inversion hard_top1 should be high, got {m_inv['hard_top1']:.2f}"
        assert m_inv["top1"] > m_loud["top1"] + 0.3
        assert m_inv["w1g"] < m_loud["w1g"]
        assert m_loud["hard_frac"] > 0.3, "too few hard episodes for the metric to mean anything"

def test_wasserstein_basics():
    D = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], float)
    assert graph_wasserstein([1, 0, 0], [1, 0, 0], D) == 0.0
    assert abs(graph_wasserstein([1, 0, 0], [0, 0, 1], D) - 2.0) < 1e-6
    assert abs(graph_wasserstein([1, 0, 0], [0, 1, 0], D) - 1.0) < 1e-6

def test_detection_honest():
    sys, op, eps, heal = _data(tree_graph(4))
    cal, test_h = heal[:40], heal[40:]
    d = evaluate_detection(eps + test_h, cal, target_far=0.05)
    assert d["f1"] > 0.8 and d["far"] <= 0.2

if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for fn in fns:
        try:
            fn(); ok += 1; print(f"PASS {fn.__name__}")
        except Exception:
            print(f"FAIL {fn.__name__}"); traceback.print_exc()
    sys_, op_, eps_, _ = _data(tree_graph(4))
    for L in (LargestResidual(), DirectSolve(lam=0.02)):
        m = evaluate_localization(L.predict_many(eps_, tree_graph(4), op_), eps_, tree_graph(4), op_)
        print(f"{L.name:16s} hard_top1={m['hard_top1']:.2f} (chance={m['chance']:.2f}, n_hard={m['n_hard']}) "
              f"top1={m['top1']:.2f} hit@3={m['hit@3']:.2f} "
              f"mrr={m['mrr']:.2f} w1g={m['w1g']:.2f} hard_frac={m['hard_frac']:.2f}")
    print(f"{ok}/{len(fns)} tests passed")
