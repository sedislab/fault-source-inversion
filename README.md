<div align="center">

# FSI

#### Fault Source Inversion: finding the cause of a robot fault

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#try-it)

[Reproduce the claims](docs/REPRODUCE.md) | [Claims](docs/CLAIMS.md) | [Configuration](docs/CONFIGURATION.md) 

</div>
This repository contains the code for the paper *Amortized Inverse Source Recovery for Fault Isolation in Robotic Systems*.
A robot fault does not always show up where it started.Fault source inversion seeks to find the root cause of faults in robotic systems by treating fault localization as an inverse problem over the robot's structural graph. 



## Try it

Everything below runs on a CPU, with no simulator:

```bash
git clone https://github.com/sedislab/fault-source-inversion
cd fault-source-inversion
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests                      # the correctness claims, a few minutes
python scripts/run_synthetic_benchmark.py   # the full comparison on synthetic propagation, needs > 4 GB RAM
```

The benchmark writes its tables to `results/synthetic/`. Generating the real corpora needs Isaac Lab 2.3.x on an NVIDIA GPU; [docs/REPRODUCE.md](docs/REPRODUCE.md) covers every tier.

## What you can do

You can use this repository to :
- Localize a fault source from a residual field with `fsi.models.FSILocalizer`, and get a calibrated candidate set from `fsi.models.ConformalSupport`.
- Compare FSI against GDN, TranAD, GraphSL, RCD, AERCA, a direct sparse solve, the largest-residual rule, and FSI-RBC, alongside three reference floors (`zz_constant`, `zz_anti_energy`, `zz_random`) .
- Build structural graphs for Franka, ANYmal, a multirotor and a vehicle, or from a URDF file, with `fsi.graph`.
- Generate your own fault corpora in Isaac Lab, with physical actuator faults and sensor faults injected inside the control loop ([docs/sensor_faults.md](docs/sensor_faults.md)).

## The metric

`hard_top1` is top-1 accuracy restricted to the episodes where the loudest candidate is not the source, so a high `hard_top1` means a method found the cause rather than the symptom. It is reported alongside plain top-1 and the graph-Wasserstein error, which measures how far a wrong answer lands from the true source in graph hops.


## Results

Four Isaac Lab corpora, 5 seeds, detection-gated. `over floor` is `hard_top1` minus `zz_constant` on the same episodes.

| corpus | candidates | chance | floor | FSI `hard_top1` | best baseline | over floor |
| --- | --- | --- | --- | --- | --- | --- |
| `franka_v9`: 7 joints, actuator faults | 7 | 0.143 | 0.150 | **0.474 ± 0.019** | TranAD 0.444 ± 0.029 | +0.324 |
| `anymal_v7`: 12 leg joints, actuator faults | 12 | 0.083 | 0.069 | **0.832 ± 0.011** | FSI-RBC 0.431 | +0.763 |
| `franka_sensor`: 7 joints and 7 encoders | 14 | 0.071 | 0.012 | **0.854 ± 0.008** | GraphSL 0.555 ± 0.055 | +0.842 |
| `anymal_sensor`: 12 joints, IMU and 4 feet | 17 | 0.059 | 0.056 | **0.877 ± 0.010** | TranAD 0.354 ± 0.049 | +0.821 |

FSI also has the best plain top-1 and the best graph-Wasserstein error on every corpus.


## Reproduce the claims

[docs/CLAIMS.md](docs/CLAIMS.md) maps each claim to the command that regenerates it and the value to expect, and [docs/REPRODUCE.md](docs/REPRODUCE.md) walks the three tiers: the CPU tier (tests and synthetic benchmark), the data tier (every number in the paper, from the processed corpora), and the simulator tier (regenerating the corpora in Isaac Lab).

Nothing in the code depends on a particular machine. Data and results live under `FSI_ROOT`, which defaults to the checkout, and cluster settings come from environment variables; [docs/CONFIGURATION.md](docs/CONFIGURATION.md) lists them.


## Citation

If FSI is useful in your work, please cite the paper:

```bibtex
@misc{aning2026fsi,
    title={Amortized Inverse Source Recovery for Fault Isolation in Robotic Systems},
    author={Abena Aning and Ernest Bonnah},
    year={2026}
}
```

## Contributing

Contributions are welcome, from a fixed typo to a new baseline or embodiment. Run the test suite before opening a pull request:

```bash
python -m pytest tests
```

[CONTRIBUTING.md](CONTRIBUTING.md) has the pull-request flow and the house style.

## License

FSI is the work of Abena Aning, Ernest Bonnah, and the SeDIS Lab at Baylor University. It is dual licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
