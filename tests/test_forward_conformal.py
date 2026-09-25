"""Forward model f_theta must make the residual field small on healthy data and track the true fault field on
faulted data; conformal sets must actually cover at the nominal rate. Both on the synthetic testbed, CPU."""
import numpy as np
import torch
torch.set_num_threads(8)
from fsi.synth import SyntheticSystem, tree_graph
from fsi.models import ForwardModel, train_forward, residual_field, ConformalSupport
from fsi.baselines import DirectSolve

def test_forward_residual_tracks_fault():
    g = tree_graph(4); sys = SyntheticSystem(g, seed=7)
    rng = np.random.default_rng(1)
    heal = [sys.sample(rng, 0, keep_series=True) for _ in range(140)]
    fm = ForwardModel(g.node_features().shape[1], du=sys.du, hist=8, hidden=48)
    train_forward(fm, heal, g, epochs=30)
    he = [np.linalg.norm(residual_field(fm, e, g, window=sys.window)[:, 0]) for e in heal[:40]]
    faulted = [sys.sample(rng, 1, keep_series=True) for _ in range(60)]
    fe, corr = [], []
    for e in faulted:
        rf = residual_field(fm, e, g, window=sys.window)[:, 0]
        fe.append(np.linalg.norm(rf))
        corr.append(np.corrcoef(rf, e.energy())[0, 1])
    print(f"forward: healthy_energy={np.mean(he):.3f} faulted_energy={np.mean(fe):.3f} corr={np.median(corr):.2f}")
    assert np.mean(fe) > 2.5 * np.mean(he)
    assert np.median(corr) > 0.5

def test_conformal_covers():
    g = tree_graph(4); sys = SyntheticSystem(g, seed=7); op = sys.sweep_operator(reps=24)
    rng = np.random.default_rng(2)
    cal = [sys.sample(rng, 1, keep_series=False) for _ in range(400)]
    test = [sys.sample(rng, 1, keep_series=False) for _ in range(300)]
    loc = DirectSolve(lam=0.03)
    aps = ConformalSupport(loc, mode="aps", alpha=0.1).calibrate(cal, g, op)
    ma = aps.evaluate(test, g, op)
    crc = ConformalSupport(loc, mode="crc", alpha=0.1).calibrate(cal, g, op)
    mc = crc.evaluate(test, g, op)
    print(f"conformal: aps set_cov={ma.get('set_coverage'):.3f} size={ma['avg_set_size']:.2f} | "
          f"crc cov={mc['coverage']:.3f} size={mc['avg_set_size']:.2f}")
    assert ma["set_coverage"] >= 0.85
    assert mc["coverage"] >= 0.85

if __name__ == "__main__":
    import traceback
    for fn in (test_forward_residual_tracks_fault, test_conformal_covers):
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception:
            print(f"FAIL {fn.__name__}"); traceback.print_exc()
