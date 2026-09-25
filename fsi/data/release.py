"""Package a versioned dataset release: per-embodiment episode shards, the propagation operator G, the structural
graph, a datasheet, and the scoring harness so anyone can reproduce CvV/W1G numbers. This is the dataset-paper
artifact — the first cross-embodiment, propagation-labeled fault-SOURCE benchmark (source, not just fault type)."""
from __future__ import annotations
import json
from pathlib import Path
from .store import save_episodes
from .splits import make_splits, fault_free
from ..eval.stats import characterize, report_md

DATASHEET = """# FSI-Bench v{version}

The first cross-embodiment, simulator-generated, propagation-labeled fault-**source** benchmark for robotic
cyber-physical systems. Every episode carries the ground-truth source support S* (which component(s) actually
faulted), the source magnitude, the fault type, the onset (abrupt/incipient) and time, and the residual field r.
Each embodiment ships its propagation operator G (column v = residual fingerprint of a unit fault at v), which is
what makes causal **localization** — not just detection — evaluable. Unlike real fault datasets (ALFA, propeller,
BASiC), simulation hands us S* and G, so we can score whether a method finds the cause versus a propagated victim.

## Embodiments
{embodiments}

## Splits
- in-distribution: train / cal / test disjoint at the episode level
- cross-embodiment holdout: train on {trained}, test on {heldout}
- cross-fault-type holdout: >=1 fault type per embodiment excluded from training
- conformal calibration: disjoint held-out split
- fault-free: healthy-only, for false-alarm-rate measurement

## Scoring
Use `fsi.metrics.evaluate_localization` (CvV primary, W1G, hop-Hit@k, RCA canon) and `fsi.metrics.evaluate_detection`
(event-F1, AUPRC, detection delay, false alarms/hour — no point adjustment). Load with `fsi.data.load_episodes`.

## Per-embodiment characterization
{stats}
"""

def package(release_dir, version, per_embodiment, trained, heldout):
    """per_embodiment: {name: (episodes, graph, operator)}. Writes shards, G, graph, splits, datasheet."""
    root = Path(release_dir) / f"v{version}"
    root.mkdir(parents=True, exist_ok=True)
    stats_md, emb_lines = [], []
    for name, (episodes, graph, operator) in per_embodiment.items():
        d = root / name; d.mkdir(exist_ok=True)
        splits = make_splits(episodes)
        splits["fault_free"] = fault_free(episodes)
        for sname, eps in splits.items():
            if eps:
                save_episodes(eps, d / sname)
        operator.save(d / "operator_G.npz")
        graph.save(d / "graph.json")
        st = characterize(episodes, graph, operator)
        (d / "characterization.md").write_text(report_md(st, f"{name} characterization"))
        stats_md.append(f"### {name}\n" + report_md(st, "").split("\n", 2)[2])
        emb_lines.append(f"- **{name}**: {graph.n} nodes, {len(graph.fail_nodes)} failable, "
                         f"{st['n_episodes']} episodes, cause-victim gap {st['cause_victim_gap']:.2f}")
    (root / "DATASHEET.md").write_text(DATASHEET.format(
        version=version, embodiments="\n".join(emb_lines), trained=", ".join(trained),
        heldout=", ".join(heldout), stats="\n\n".join(stats_md)))
    (root / "manifest.json").write_text(json.dumps(
        {"version": version, "embodiments": list(per_embodiment), "trained": trained, "heldout": heldout}, indent=2))
    return root
