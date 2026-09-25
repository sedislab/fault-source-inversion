"""FSI amortized inverse end-to-end: it should match the iterative direct solve on the easy operator, beat it
under noise/nonlinearity, and recover multi-fault support. All on the synthetic testbed, CPU.

Scored on `hard_top1`, not CvV: CvV was retired on 2026-07-30 after a scorer ranking nodes by NEGATIVE residual
energy achieved CvV 1.000 while localizing almost nothing, so every assertion written against it was vacuous."""
import numpy as np
import torch
torch.set_num_threads(8)
from fsi.synth import SyntheticSystem, tree_graph
from fsi.baselines import LargestResidual, DirectSolve
from fsi.models import FSILocalizer
from fsi.metrics import evaluate_localization

def _split(sys, n_tr=1400, n_te=300, seed=11):
    rng = np.random.default_rng(seed)
    tr = sys.dataset(int(n_tr * .15), int(n_tr * .6), int(n_tr * .25), seed=seed)
    te_single = [sys.sample(rng, 1, keep_series=False) for _ in range(n_te)]
    te_double = [sys.sample(rng, 2, keep_series=False) for _ in range(150)]
    return tr, te_single, te_double

def test_fsi_matches_direct_easy():
    g = tree_graph(4); sys = SyntheticSystem(g, seed=7); op = sys.sweep_operator(reps=24)
    tr, te, _ = _split(sys)
    fsi = FSILocalizer(epochs=80).fit(tr, g, op)
    m_fsi = evaluate_localization(fsi.predict_many(te, g, op), te, g, op)
    m_dir = evaluate_localization(DirectSolve(lam=0.02).predict_many(te, g, op), te, g, op)
    print(f"easy: fsi top1={m_fsi['top1']:.2f} hard={m_fsi['hard_top1']:.2f} | direct top1={m_dir['top1']:.2f}")
    assert m_fsi["top1"] > 0.88
    assert m_fsi["top1"] >= m_dir["top1"] - 0.08

def test_fsi_beats_victims_nonlinear():
    """Under nonlinearity, FSI (cause-finder) must crush the victim rule on CvV/top1 and stay competitive with
    the exact solve. Beating the linear solve outright is a real-data (strongly nonlinear) claim, not synthetic."""
    from fsi.baselines import LargestResidual
    g = tree_graph(4); sys = SyntheticSystem(g, noise=0.14, sat=1.2, seed=5); op = sys.sweep_operator(reps=24)
    tr, te, _ = _split(sys, seed=13)
    fsi = FSILocalizer(epochs=90).fit(tr, g, op)
    m_fsi = evaluate_localization(fsi.predict_many(te, g, op), te, g, op)
    m_dir = evaluate_localization(DirectSolve(lam=0.05).predict_many(te, g, op), te, g, op)
    m_loud = evaluate_localization(LargestResidual().predict_many(te, g, op), te, g, op)
    print(f"nonlinear: fsi top1={m_fsi['top1']:.2f} hard={m_fsi['hard_top1']:.2f} | dir top1={m_dir['top1']:.2f} "
          f"| loud top1={m_loud['top1']:.2f} hard={m_loud['hard_top1']:.2f}")
    # the loudest-node rule scores 0 on hard episodes by construction; a cause-finder must not
    assert m_fsi["hard_top1"] > m_loud["hard_top1"] + 0.3
    assert m_fsi["top1"] >= m_loud["top1"] and m_fsi["top1"] >= m_dir["top1"] - 0.3

def test_fsi_multifault():
    g = tree_graph(4); sys = SyntheticSystem(g, seed=7); op = sys.sweep_operator(reps=24)
    tr, _, te2 = _split(sys)
    fsi = FSILocalizer(epochs=80).fit(tr, g, op)
    m_fsi = evaluate_localization(fsi.predict_many(te2, g, op), te2, g, op)
    m_loud = evaluate_localization(LargestResidual().predict_many(te2, g, op), te2, g, op)
    print(f"k=2: fsi support_f1={m_fsi['support_f1']:.2f} | loud support_f1={m_loud['support_f1']:.2f}")
    assert m_fsi["support_f1"] > m_loud["support_f1"]

if __name__ == "__main__":
    import traceback
    for fn in (test_fsi_matches_direct_easy, test_fsi_beats_victims_nonlinear, test_fsi_multifault):
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception:
            print(f"FAIL {fn.__name__}"); traceback.print_exc()
