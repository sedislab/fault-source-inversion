"""Run every localizer through the identical harness and tabulate. Each localizer's fit takes what it needs
(unsupervised methods filter healthy episodes internally, supervised ones use the labels); we time fit and
per-episode inference so the control-loop-latency claim is measured, not asserted.

Two protocol rules are enforced here rather than left to the caller, because violating either silently produces a
publishable-looking table that is wrong:
  * scores must be finite -- a baseline emitting all-NaN does not crash, it gets stably argsorted into naming
    node 0 on every episode, which reads as a plausible weak accuracy instead of a dead method;
  * a single seed is not a result -- across seeds the learned baselines move by more than the gaps being claimed,
    so `seeds` runs each method repeatedly and the table carries the spread."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ..metrics import evaluate_localization

_COLS = ["localizer", "top1", "top1_sd", "hard_top1", "hard_top1_sd", "hit@3", "mrr", "w1g", "support_f1",
         "n_hard", "n_fault", "chance", "n_seeds", "fit_s", "pred_ms", "error"]
_SPREAD = ("top1", "hard_top1")

def run_comparison(localizers, train, test, graph, operator, threads=8, verbose=True, seeds=(0,), candidates=None,
                   keep=None):
    """`localizers` may hold Localizer instances or zero-argument factories; factories are required for a clean
    multi-seed run, since an instance carries state from its previous fit. `keep`, if given, collects the last
    successfully fitted instance per name for downstream use (conformal calibration reuses the fitted FSI)."""
    rows = []
    for spec in localizers:
        torch.set_num_threads(threads)
        runs, fit_s, pred_ms, err, name = [], [], [], "", None
        for sd in seeds:
            loc = spec() if callable(spec) else spec
            name = name or loc.name
            if hasattr(loc, "seed"):
                loc.seed = sd
            torch.manual_seed(sd)
            np.random.seed(sd)
            try:
                t0 = time.time(); loc.fit(train, graph, operator); fit_s.append(time.time() - t0)
                t1 = time.time(); preds = loc.predict_many(test, graph, operator)
                pred_ms.append(1000 * (time.time() - t1) / max(1, len(test)))
                bad = sum(1 for p in preds if not np.isfinite(p.scores).all())
                if bad:
                    raise ValueError(f"non-finite scores on {bad}/{len(preds)} episodes")
                runs.append(evaluate_localization(preds, test, graph, operator, candidates=candidates))
                if keep is not None:
                    keep[loc.name] = loc
            except Exception as e:
                err = f"{type(e).__name__}: {e}"[:140]
                break
        row = {"localizer": name, "error": err, "n_seeds": len(runs)}
        if runs:
            for k in runs[0]:
                row[k] = float(np.mean([r[k] for r in runs]))
            for k in _SPREAD:
                row[f"{k}_sd"] = float(np.std([r[k] for r in runs], ddof=1)) if len(runs) > 1 else float("nan")
            for k in ("n_hard", "n_fault"):
                row[k] = int(runs[0][k])
            row["fit_s"] = round(float(np.mean(fit_s)), 1)
            row["pred_ms"] = round(float(np.mean(pred_ms)), 3)
        rows.append(row)
        if verbose:
            if err:
                print(f"  !! {name:16s} FAILED  {err}", flush=True)
            else:
                sd = f" +-{row['hard_top1_sd']:.3f}" if len(runs) > 1 else ""
                print(f"  {name:16s} top1={row['top1']:.3f} hard_top1={row['hard_top1']:.3f}{sd} "
                      f"w1g={row['w1g']:.2f} pred_ms={row['pred_ms']}", flush=True)
    df = pd.DataFrame(rows)
    # Keep the canonical columns in their canonical order, then APPEND whatever else the metric suite produced
    # (the per-family `top1_<family>` breakdown). Dropping unknown columns is how a per-family split would
    # silently vanish from every table the moment sensor faults enter the corpus, leaving one pooled accuracy
    # that moves with the fault mix rather than with the method.
    known = [c for c in _COLS if c in df.columns]
    extra = [c for c in df.columns if c not in known and c != "error"]
    return df[[c for c in known if c != "error"] + sorted(extra) + (["error"] if "error" in df.columns else [])]

def fitted(localizers, name):
    return next(l for l in localizers if l.name == name)

def save_table(df, path, title="Localization comparison"):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=False)
    md = [f"# {title}", "", df.round(3).to_markdown(index=False), "",
          "`hard_top1` = top-1 restricted to episodes where the loudest node is not the source "
          "(chance = 1/|candidates|; the loudest-node rule scores 0 there by construction). The `zz_*` rows are "
          "degenerate reference scorers carrying no information about the source: they mark the floor of every "
          "column, so a margin can be read as a margin."]
    path.with_suffix(".md").write_text("\n".join(md))
    return path
