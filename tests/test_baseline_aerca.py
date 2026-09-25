"""AERCA baseline: trains unsupervised on healthy series, localizes the cause by exogenous z-score,
must clearly beat random top1 on held-out single faults. Fast CPU config."""
import numpy as np
from fsi.synth import SyntheticSystem, tree_graph
from fsi.baselines.aerca import AERCA
from fsi.metrics import evaluate_localization

def _data(gf, seed=7):
    sys = SyntheticSystem(gf, seed=seed)
    op = sys.sweep_operator(reps=16)
    rng = np.random.default_rng(3)
    heal = [sys.sample(rng, 0, keep_series=True) for _ in range(150)]
    test = [sys.sample(rng, 1, keep_series=True) for _ in range(150)]
    return sys, op, gf, heal, test

def test_aerca_beats_random():
    sys, op, gf, heal, test = _data(tree_graph(4))
    loc = AERCA(epochs=12, hidden=32).fit(heal, gf, op)
    m = evaluate_localization(loc.predict_many(test, gf, op), test, gf, op)
    rand = 1.0 / len(gf.fail_nodes)
    print(f"aerca hard_top1={m['hard_top1']:.2f} (chance={m['chance']:.3f}, n_hard={m['n_hard']}) "
          f"top1={m['top1']:.2f} hit@3={m['hit@3']:.2f} mrr={m['mrr']:.2f} (random top1={rand:.3f})")
    assert not np.isnan(m["top1"])
    assert m["top1"] > 2 * rand or m["top1"] > 0.25, f"top1 {m['top1']:.3f} not above random {rand:.3f}"

if __name__ == "__main__":
    import traceback
    try:
        test_aerca_beats_random(); print("PASS")
    except Exception:
        print("FAIL"); traceback.print_exc()
