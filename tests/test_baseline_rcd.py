"""RCD (causal-RCA) baseline: fits a normal reference, scores held-out single faults, must beat random top1."""
import numpy as np
from fsi.synth import SyntheticSystem, chain_graph
from fsi.baselines.rcd import RCD
from fsi.metrics import evaluate_localization

def _data(seed=7, n_train=100, n_test=150):
    gf = chain_graph(5)
    sys = SyntheticSystem(gf, seed=seed)
    op = sys.sweep_operator(reps=16)
    rng = np.random.default_rng(seed + 1)
    train = [sys.sample(rng, 0, keep_series=True) for _ in range(n_train)]
    test = [sys.sample(rng, 1, keep_series=True) for _ in range(n_test)]
    return gf, op, train, test

def test_rcd_beats_random():
    gf, op, train, test = _data()
    loc = RCD().fit(train, gf, op)
    m = evaluate_localization(loc.predict_many(test, gf, op), test, gf, op)
    random = 1.0 / len(gf.fail_nodes)
    print(f"rcd hard_top1={m['hard_top1']:.3f} (chance={m['chance']:.3f}, n_hard={m['n_hard']}) "
          f"top1={m['top1']:.3f} hit@3={m['hit@3']:.3f} random={random:.3f}")
    assert not np.isnan(m["top1"])
    assert m["top1"] > 2 * random or m["top1"] > 0.25, f"top1 {m['top1']:.3f} vs random {random:.3f}"

def test_rcd_needs_series():
    gf, op, train, test = _data(n_test=1)
    loc = RCD().fit(train, gf, op)
    dry = test[0]
    dry.y = None
    try:
        loc.predict(dry, gf, op); raised = False
    except ValueError:
        raised = True
    assert raised, "RCD must raise a clear error without time-series"

if __name__ == "__main__":
    test_rcd_beats_random()
    test_rcd_needs_series()
    print("PASS")
