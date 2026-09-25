# Figures

Paper figures are generated from the evaluation results by the scripts in `scripts/paper/` and written here, as a
PDF for the paper and a PNG for the web:

| figure | script |
| --- | --- |
| `fig1_main_results`, `fig1b_top1` | `scripts/paper/fig_main.py` |
| `fig_ablation`, `fig_forward_vs_localize` | `scripts/paper/fig_ablation.py` |
| `fig_metric_validity` | `scripts/paper/fig_metric_validity.py` |
| `fig_identifiability`, `fig_multifault`, `fig_severity`, `fig_transfer` | `scripts/paper/fig_rest.py` |

Each script reads its inputs from `$FSI_ROOT/results/` (see [docs/CONFIGURATION.md](../docs/CONFIGURATION.md)), so
run the matching evaluation first. [docs/REPRODUCE.md](../docs/REPRODUCE.md) lists the order.
