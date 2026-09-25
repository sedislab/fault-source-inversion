"""TranAD baseline: unsupervised transformer reconstruction localizes by per-channel deviation over the fault
window. A non-graph victim-style ablation -- it need not win cause-over-victim episodes (hard_top1, which replaced the
retired CvV), but must clearly beat random top-1."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import torch; torch.set_num_threads(2)
from fsi.synth import SyntheticSystem, tree_graph
from fsi.baselines.tranad import TranAD
from fsi.metrics import evaluate_localization

def _data(seed=7):
    g = tree_graph(4)
    sys = SyntheticSystem(g, seed=seed)
    op = sys.sweep_operator(reps=16)
    rng = np.random.default_rng(0)
    train = [sys.sample(rng, 0, keep_series=True) for _ in range(160)]
    test = [sys.sample(rng, 1, keep_series=True) for _ in range(150)]
    return g, op, train, test

def test_tranad_beats_random():
    g, op, train, test = _data()
    loc = TranAD(epochs=12, hidden=16, seed=0).fit(train, g, op)
    m = evaluate_localization(loc.predict_many(test, g, op), test, g, op)
    rand = 1.0 / len(g.fail_nodes)
    print(f"tranad hard_top1={m['hard_top1']:.2f} (chance={m['chance']:.3f}, n_hard={m['n_hard']}) "
          f"top1={m['top1']:.2f} hit@3={m['hit@3']:.2f} random={rand:.3f}")
    assert not np.isnan(m["top1"]) and not np.isnan(m["hit@3"])
    assert m["top1"] > 2 * rand and m["top1"] > 0.15
    assert m["n_hard"] > 0 and 0.0 <= m["hard_top1"] <= 1.0

def test_tranad_needs_series():
    g = tree_graph(4); sys = SyntheticSystem(g, seed=1)
    ep = sys.sample(np.random.default_rng(2), 1, keep_series=False)
    try:
        TranAD(epochs=1).predict(ep, g); assert False, "expected ValueError on missing series"
    except ValueError:
        pass

if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for fn in fns:
        try:
            fn(); ok += 1; print(f"PASS {fn.__name__}")
        except Exception:
            print(f"FAIL {fn.__name__}"); traceback.print_exc()
    print(f"{ok}/{len(fns)} tests passed")
