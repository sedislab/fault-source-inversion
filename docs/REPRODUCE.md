# Reproducing the claims

Every number in [CLAIMS.md](CLAIMS.md) comes from one of three tiers. The tiers are ordered by what they need, so you can stop at whichever one your hardware allows.

- **The CPU tier** needs only a CPU and the Python dependencies. It covers the correctness claims and the synthetic benchmark, and it takes a few minutes.
- **The data tier** needs the processed corpora. It covers every localization, detection, ablation and severity number in the paper. A CPU is enough; a GPU makes training the forward model faster.
- **The simulator tier** needs an NVIDIA GPU and Isaac Lab 2.3.x. It regenerates the corpora from scratch.

All commands run from the repository root. Paths are resolved under `FSI_ROOT`, which defaults to the checkout; [CONFIGURATION.md](CONFIGURATION.md) explains how to move it.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`torch_geometric` is used only by the GraphSL baseline. If it will not install on your platform, every other test and method still runs.

## The CPU tier

```bash
python -m pytest tests
python scripts/run_synthetic_benchmark.py
```

The tests check the contracts, the metric, the inverse model, every baseline and the conformal sets on small synthetic problems. The benchmark runs the full comparison on the real embodiment topologies (Franka, ANYmal, a multirotor and a vehicle) with synthetic propagation, and writes per-embodiment tables and a summary to `results/synthetic/`. It is the same harness the real corpora go through.

The benchmark trains every baseline on four embodiments and needs more than 4 GB of memory; it is stopped by the out-of-memory killer on a 4 GB machine. The tests fit comfortably in 4 GB.

The paper's synthetic studies (multi-fault, transfer, the comparison table) have their own drivers:

```bash
python scripts/paper/synth_comparison.py
python scripts/paper/synth_multifault.py
python scripts/paper/synth_transfer.py
python scripts/paper/synth_aggregate.py
```

## The data tier


### Building a corpus from raw rollouts

Stage B turns raw rollouts into the FSI dataset. It trains the forward model f_θ on healthy rollouts, computes the residual field of every episode, and estimates the propagation operator G from the unit-fault sweep minus its matched healthy reference.

| corpus | robot and faults | `--graph` | `--channels` |
| --- | --- | --- | --- |
| `franka_v9` | Franka, actuator faults | `franka_joint` | 3 |
| `anymal_v7` | ANYmal-C, actuator faults | `anymal_sim` | 3 |
| `franka_sensor` | Franka, actuator and encoder faults | `franka_sensor` | 3 |
| `anymal_sensor` | ANYmal-C, actuator, encoder, IMU and foot-contact faults | `anymal_sensor` | 4 |

```bash
python scripts/build_embodiment.py --raw "$FSI_ROOT/raw/anymal_sensor" --out "$FSI_ROOT/data/anymal_sensor" \
    --graph anymal_sensor --channels 4
```

### Evaluating

`eval_embodiment.py` runs every localizer through one harness, with detection, conformal sets and latency, and repeats each localizer over `--seeds` seeds. Write each corpus's results to `results/<corpus>/` with the corpus name as the tag, because the summary script finds tables by that layout:

```bash
for c in franka_v9 anymal_v7 franka_sensor anymal_sensor; do
  python scripts/eval_embodiment.py --data "$FSI_ROOT/data/$c" --out "$FSI_ROOT/results/$c" --tag "$c" --seeds 5
done
python scripts/paper/results_summary.py                  # ungated tables  -> results/paper/SUMMARY.md
python scripts/paper/results_summary.py --split DETECTED # detection-gated tables
```

Each run writes `<tag>_in_distribution.csv` (ungated), `<tag>_DETECTED.csv` (detection-gated) and `<tag>_detection_conformal.json`, each table with a Markdown copy. Add `--heldout_task` for the held-out-task split on multi-task corpora.

### The paper's analyses

The component sweep behind the README's ablation table:

```bash
python scripts/paper/fsi_sweep.py --data "$FSI_ROOT/data/franka_v8" --out "$FSI_ROOT/results/paper/fsi_sweep" --seeds 3
```

The detection, severity, identifiability and metric-validity analyses, and the per-component ablation, run on the `franka_v6` and `anymal_v4` builds (the ablation and detection-seed scripts rebuild from the raw `franka_v4` and `anymal_v4` rollouts). The SLURM job files next to each script show the settings used for the paper:

| analysis | command | job file |
| --- | --- | --- |
| detection and conformal | `python scripts/paper/detection_conformal.py` then `detection_conformal_summary.py` | `detection_conf.sbatch` |
| detection over seeds | `python scripts/paper/detection_seeds.py --emb franka --seed 0` for each embodiment and seed, then `detection_merge.py` | `detection_run.sbatch` |
| severity | `python scripts/paper/severity_run.py --split_seeds 0 1 2 3 4 --model_seeds 5`, then `severity_agg.py` | `severity_job.sbatch` (GPU) or `severity_job_cpu.sbatch` |
| identifiability | `python scripts/paper/identifiability_confusion.py --seeds 5 --perms 2000` | `identifiability_job.sbatch` |
| component ablation | `bash scripts/paper/ablation_submit.sh franka` (and `anymal`), then `ablation_collect.py` | submits one job per arm |

On a SLURM cluster, set `SBATCH_ACCOUNT` and `SBATCH_PARTITION` first (see [CONFIGURATION.md](CONFIGURATION.md)) and submit from the repository root, for example `sbatch scripts/paper/severity_job.sbatch`.

### Figures

Once the results above exist, the figure scripts write to `figures/`; [figures/README.md](../figures/README.md) maps each figure to its script.

```bash
python scripts/paper/fig_main.py
python scripts/paper/fig_ablation.py
python scripts/paper/fig_metric_validity.py
python scripts/paper/fig_rest.py
```

## The simulator tier

This tier regenerates the raw rollouts. It needs an NVIDIA GPU with RT cores for the video recorder (the corpus generators run headless and do not), Apptainer, and an Isaac Lab 2.3.x container image. Point `FSI_ISAAC_SIF` at the image; `run_in_container.sh` handles the writable directories Kit needs.

Run each generator's bind gate first. It checks, before hours of simulation, that every fault kind actually changes the robot's behaviour at its source, and the generators refuse to write a corpus that fails it ([sensor_faults.md](sensor_faults.md) explains the criterion):

```bash
export FSI_ISAAC_SIF=/path/to/isaac-lab.sif
scripts/isaac/run_in_container.sh scripts/isaac/gen_anymal_sensor.py --headless --check_only
scripts/isaac/run_in_container.sh scripts/isaac/gen_anymal_sensor.py --headless
```

| generator | writes by default | notes |
| --- | --- | --- |
| `scripts/isaac/gen_franka_v2.py` | `raw/franka_v2` | multi-task Franka, actuator faults |
| `scripts/isaac/gen_anymal_v2.py` | `raw/anymal_v4` | ANYmal-C, actuator faults through the ANYdrive actuator model |
| `scripts/isaac/gen_franka_sensor.py` | `raw/franka_sensor` | adds encoder faults inside the servo loop |
| `scripts/isaac/gen_anymal_sensor.py` | `raw/anymal_sensor` | adds encoder, IMU and foot-contact faults |

Pass `--out` to write a differently named corpus. **TODO (author): record the exact generator and arguments that produced `franka_v9`, `anymal_v7` and `franka_v8`.**

`scripts/isaac/probes/` holds the short diagnostics used while building the generators (environment inspection, actuator and sensor binding probes). `scripts/legacy/` holds the first Franka and ANYmal pipeline, kept for reference; its generators used fault mechanisms that do not bind on ANYmal-C, which is why the v2 generators exist.

## Data format

**Raw rollouts** (`raw/<corpus>/`). One compressed NumPy shard per batch, named `<task or robot>_<kind>_<batch>.npz`, where the kind is `healthy`, `fault`, `sweep` or `sweepref` (the matched healthy reference for a sweep). A `manifest.json` lists every shard with its kind and episode count.

- `signals`: float array, episodes × time steps × graph nodes × channels
- `actions`: float array, episodes × time steps × action dimension (the commanded joint targets)
- `labels` (fault and sweep shards): a JSON string with one record per episode, holding the fault `kind`, the source node index (`joint`), `severity`, `onset` step, and for the sensor corpora the fault `family`

**Processed corpora** (`data/<corpus>/`). Episode shards are stored as an `.npz` with a matching `.json` and are read with `fsi.data.load_episodes`.

- `r`: the residual field per episode, nodes × features
- `src`: the ground-truth source vector, holding the fault magnitude at each faulted node and zero elsewhere
- `y`, `u`, `yhat` (when stored): the measured signals, commands, and f_θ's prediction
- the `.json` holds, per episode, the embodiment and each fault's `node`, `fault_type`, `magnitude`, `onset` profile, `onset_step` and `end_step`
- `operator_G.npz`: the propagation operator, one column per candidate source
- `graph.json`: the structural graph (nodes, types, edges)
- `characterization.md`: the corpus's cause-victim gap, propagation depth and related statistics

## Tolerances

The value you get needs to fall within the tolerance stated in [CLAIMS.md](CLAIMS.md) for a claim to hold. If it does not on your machine, please open an issue with your hardware, versions and the command you ran.
