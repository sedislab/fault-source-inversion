"""Characterize a dataset for the dataset paper and the Gate-G1 pilot: the cause-victim gap (the number the
whole thesis rests on), how far faults propagate, how identifiable the operator is, and the residual floor. If
the cause-victim gap is small, the benchmark cannot reveal the contribution and the fault mix must be rebalanced."""
from __future__ import annotations
import numpy as np
from ..core.graph import StructuralGraph
from ..core.types import Operator

def characterize(episodes, graph: StructuralGraph, operator: Operator | None = None) -> dict:
    D = graph.hop_distance()
    faults = [e for e in episodes if not e.is_healthy]
    healthy = [e for e in episodes if e.is_healthy]
    gap, depth, by_type = [], [], {}
    for e in faults:
        en = e.energy(); src = e.support()[0]
        loud = int(np.argmax(en))
        is_gap = int(loud != src)
        gap.append(is_gap)
        depth.append(float(min(D[loud, s] for s in e.support())))
        ft = e.faults[0].fault_type
        by_type.setdefault(ft, []).append(is_gap)
    out = {
        "n_episodes": len(episodes), "n_fault": len(faults), "n_healthy": len(healthy),
        "cause_victim_gap": float(np.mean(gap)) if gap else float("nan"),
        "mean_propagation_hops": float(np.mean(depth)) if depth else float("nan"),
        "k_distribution": {k: sum(1 for e in faults if e.k == k) for k in sorted({e.k for e in faults})},
        "healthy_energy": float(np.mean([np.linalg.norm(e.energy()) for e in healthy])) if healthy else float("nan"),
        "fault_energy": float(np.mean([np.linalg.norm(e.energy()) for e in faults])) if faults else float("nan"),
        "gap_by_fault_type": {k: round(float(np.mean(v)), 3) for k, v in sorted(by_type.items())},
    }
    if operator is not None:
        M = operator.coherence_matrix()
        out["coherence"] = float(M.max())
        out["mean_coherence"] = float(M[np.triu_indices_from(M, 1)].mean())
        out["ambiguous_pairs"] = int((M > 0.99).sum() // 2)
    return out

def report_md(stats: dict, title="Dataset characterization") -> str:
    lines = [f"# {title}", ""]
    for k, v in stats.items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", f"> cause-victim gap = {stats['cause_victim_gap']:.2f}: fraction of fault episodes where the "
              "loudest node is not the source. This must be large for FSI's contribution to be testable."]
    return "\n".join(lines)
