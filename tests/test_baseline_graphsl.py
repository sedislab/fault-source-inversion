"""GraphSL (learned field->source inverse) must clearly beat random top1 on held-out single-fault episodes."""
import numpy as np
from fsi.synth import SyntheticSystem, tree_graph, chain_graph
from fsi.baselines.graphsl import GraphSL
from fsi.metrics import evaluate_localization

def _data(gf, n_train=400, n_test=150, seed=7):
    sys = SyntheticSystem(gf, seed=seed)
    op = sys.sweep_operator(reps=16)
    rng = np.random.default_rng(11)
    train = [sys.sample(rng, 1, keep_series=False) for _ in range(n_train // 2)]
    train += [sys.sample(rng, 2, keep_series=False) for _ in range(n_train // 2)]
    test = [sys.sample(rng, 1, keep_series=False) for _ in range(n_test)]
    return sys, op, train, test

def test_graphsl_beats_random():
    for gf in (tree_graph(4), chain_graph(5)):
        sys, op, train, test = _data(gf)
        loc = GraphSL(hidden=32, epochs=12).fit(train, gf, op)
        m = evaluate_localization(loc.predict_many(test, gf, op), test, gf, op)
        rand = 1.0 / len(gf.fail_nodes)
        print(f"{gf.name}: hard_top1={m['hard_top1']:.2f} (chance={m['chance']:.2f}, n_hard={m['n_hard']}) "
              f"top1={m['top1']:.2f} hit@3={m['hit@3']:.2f} rand={rand:.2f}")
        assert np.isfinite(m["top1"]) and m["n_hard"] > 0 and np.isfinite(m["hard_top1"])
        assert m["top1"] > max(2 * rand, 0.25), f"{gf.name}: top1 {m['top1']:.2f} not clearly above random {rand:.2f}"

if __name__ == "__main__":
    import traceback
    try:
        test_graphsl_beats_random(); print("PASS test_graphsl_beats_random")
    except Exception:
        print("FAIL test_graphsl_beats_random"); traceback.print_exc()
