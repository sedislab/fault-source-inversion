"""GDN baseline: trains unsupervised on healthy series, localizes by normalized deviation, and must beat random
top1. It is victim-style, so on cause-over-victim episodes (hard_top1, which replaced the retired CvV) it should
stay down at chance -- it has no mechanism for inverting propagation."""
import numpy as np
from fsi.synth import SyntheticSystem, tree_graph
from fsi.baselines.gdn import GDN
from fsi.metrics import evaluate_localization

def _data(seed=7):
    sys = SyntheticSystem(tree_graph(4), seed=seed)
    op = sys.sweep_operator(reps=16)
    rng = np.random.default_rng(3)
    train = [sys.sample(rng, 0, keep_series=True) for _ in range(150)]
    test = [sys.sample(rng, 1, keep_series=True) for _ in range(150)]
    return sys, op, train, test

def test_gdn_beats_random():
    sys, op, train, test = _data()
    gdn = GDN(dim=32, epochs=12, seed=0).fit(train, sys.g, op)
    m = evaluate_localization(gdn.predict_many(test, sys.g, op), test, sys.g, op)
    random = 1.0 / len(sys.g.fail_nodes)
    print(f"gdn hard_top1={m['hard_top1']:.2f} (chance={m['chance']:.3f}, n_hard={m['n_hard']}) "
          f"top1={m['top1']:.2f} hit@3={m['hit@3']:.2f} mrr={m['mrr']:.2f} (random={random:.3f})")
    assert not np.isnan(m["top1"])
    assert m["top1"] > 2 * random, f"top1 {m['top1']:.3f} should beat 2*random {2*random:.3f}"
    assert m["n_hard"] > 0, "no cause-over-victim episodes: hard_top1 would be vacuous"
    assert m["hard_top1"] <= 2 * m["chance"], (
        f"victim-style method should stay near chance on hard episodes, got hard_top1 "
        f"{m['hard_top1']:.3f} vs chance {m['chance']:.3f}")

if __name__ == "__main__":
    test_gdn_beats_random()
    print("PASS")
