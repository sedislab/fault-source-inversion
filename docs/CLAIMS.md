# Claims

In writing the paper we make claims about the correctness of the method, the validity of its metric, and how it compares with prior work. To make them easy to check, this document gathers every claim with the command that reproduces it, the value to expect, and the tolerance it holds to.

The claims are grouped into the three tiers of [REPRODUCE.md](REPRODUCE.md): the CPU tier (a CPU and the Python dependencies), the data tier (the processed corpora), and the simulator tier (an NVIDIA GPU and Isaac Lab 2.3.x). CPU-tier claims are checked by the test suite and hold exactly as stated. Data-tier numbers are means and standard deviations over seeds from our reference runs.



## Metric validity claims

1. On episodes where the loudest node is not the source, the loudest-node rule scores exactly zero `hard_top1`, while inverting propagation scores above 0.85. At least 30% of episodes are hard, so the metric is not vacuous, and inversion also has the lower graph-Wasserstein error.
Command: `python -m pytest tests/test_metrics.py::test_hard_top1_separates_cause_from_victim`.
Expected: pass.
Tolerance: loudest-node `hard_top1` ≤ 1e-9; inversion `hard_top1` > 0.85; inversion top-1 above loudest-node top-1 by more than 0.3.
Tier: CPU.

2. The graph-Wasserstein error measures how far a wrong answer lands from the true source, in graph hops.
Command: `python -m pytest tests/test_metrics.py::test_wasserstein_basics`.
Expected: 0 for the correct node, 1 for a one-hop neighbour, 2 for a two-hop node on a path graph.
Tolerance: 1e-6.
Tier: CPU.

3. Detection is scored without point adjustment and still separates faulted from healthy episodes.
Command: `python -m pytest tests/test_metrics.py::test_detection_honest`.
Expected: with the threshold calibrated on healthy episodes for a 0.05 false-alarm target, F1 above 0.8 and a measured false-alarm rate of at most 0.2.
Tolerance: as stated.
Tier: CPU.

## Correctness claims

4. The structural graph type satisfies its contracts: contiguous node ids, a valid hop-distance matrix, at least one failable node, and a lossless round trip through serialization.
Command: `python -m pytest tests/test_foundation.py::test_graph_contracts`.
Expected: pass on the chain, tree and star test graphs.
Tolerance: exact.
Tier: CPU.

5. The synthetic testbed has a genuine cause-victim gap on every embodiment topology, so the thesis can be tested on it.
Command: `python -m pytest tests/test_foundation.py::test_cause_victim_gap`.
Expected: the fraction of episodes whose loudest node is not the source exceeds 0.35 on each graph.
Tolerance: as stated.
Tier: CPU.

6. Estimating the propagation operator by sweeping unit faults recovers the analytic operator.
Command: `python -m pytest tests/test_foundation.py::test_operator_recovery`.
Expected: every swept column has cosine similarity above 0.95 with the analytic column.
Tolerance: as stated.
Tier: CPU.

7. Inverting the operator finds the cause where the loudest-node rule finds a victim.
Command: `python -m pytest tests/test_foundation.py::test_inversion_beats_loudest`.
Expected: over 200 single-fault episodes on a tree graph, a non-negative least-squares inversion names the source more than 90% of the time, and beats the loudest-node rule by more than 0.4.
Tolerance: as stated.
Tier: CPU.

8. The amortized inverse h_ψ matches the exact sparse solve on easy problems, beats the loudest victim under nonlinear propagation, and handles two simultaneous faults.
Command: `python -m pytest tests/test_inverse.py`.
Expected: top-1 above 0.88 and within 0.08 of direct-l1 on the easy case; `hard_top1` above largest-residual by more than 0.3 on the nonlinear case; higher support F1 than largest-residual on two-fault episodes.
Tolerance: as stated.
Tier: CPU.

9. The forward model's residual rises on faulted episodes and follows the injected fault.
Command: `python -m pytest tests/test_forward_conformal.py::test_forward_residual_tracks_fault`.
Expected: mean residual energy on faulted episodes more than 2.5 times that on healthy ones, and a median correlation above 0.5 with each episode's true residual energy field.
Tolerance: as stated.
Tier: CPU.

10. The conformal candidate sets reach their coverage target.
Command: `python -m pytest tests/test_forward_conformal.py::test_conformal_covers`.
Expected: at α = 0.1 (target 0.90), both the APS set and the conformal risk control set cover the true source on at least 85% of 300 held-out episodes.
Tolerance: coverage ≥ 0.85, allowing for finite-sample variation around the 0.90 target.
Tier: CPU.

11. Every baseline is a working implementation rather than a stub. GDN, TranAD, GraphSL, RCD and AERCA each beat random on their test problem, and the time-series methods raise a clear error when given no time series.
Command: `python -m pytest tests/test_baseline_gdn.py tests/test_baseline_tranad.py tests/test_baseline_graphsl.py tests/test_baseline_rcd.py tests/test_baseline_aerca.py`.
Expected: pass.
Tolerance: the thresholds in each test.
Tier: CPU.

## Localization claims

Claims 12 to 15 are detection-gated (the localizer runs only when detection fires), use 5 seeds, and report `hard_top1` with its margin over the `zz_constant` floor on the same episodes.
Command, for all four: `python scripts/eval_embodiment.py --data "$FSI_ROOT/data/<corpus>" --out "$FSI_ROOT/results/<corpus>" --tag <corpus> --seeds 5` for each corpus, then `python scripts/paper/results_summary.py --split DETECTED`.
Tier: data.

12. Franka, actuator faults (`franka_v9`, 7 candidates, chance 0.143, floor 0.150): FSI 0.474 ± 0.019, best baseline TranAD 0.444 ± 0.029, margin over floor +0.324.
Tolerance: TODO (author). Note: TranAD's own score moves between dataset builds (0.444 to 0.484), because each build retrains f_θ from a different seed, so FSI and TranAD are close on this gated protocol. See claim 16.

13. ANYmal, actuator faults (`anymal_v7`, 12 candidates, chance 0.083, floor 0.069): FSI 0.832 ± 0.011, best baseline FSI-RBC 0.431, margin +0.763.
Tolerance: TODO (author).

14. Franka, actuator and encoder faults (`franka_sensor`, 14 candidates, chance 0.071, floor 0.012): FSI 0.854 ± 0.008, best baseline GraphSL 0.555 ± 0.055, margin +0.842.
Tolerance: TODO (author).

15. ANYmal, actuator, encoder, IMU and foot faults (`anymal_sensor`, 17 candidates, chance 0.059, floor 0.056): FSI 0.877 ± 0.010, best baseline TranAD 0.354 ± 0.049, margin +0.821.
Tolerance: TODO (author).

16. Ungated, on the same builds, FSI also leads: Franka 0.359 ± 0.009 against GDN 0.298 ± 0.026 and TranAD 0.286 ± 0.022; ANYmal 0.822 ± 0.006 against GraphSL 0.322.
Command: as above, then `python scripts/paper/results_summary.py`.
Tolerance: TODO (author).
Tier: data.

17. FSI has the best plain top-1 and the best graph-Wasserstein error on every corpus, so its wrong answers land closer to the true source (Franka: 0.50 hops against TranAD's 1.37).
Command: as for claims 12 to 15; read the `top1` and `w1g` columns.
Tolerance: TODO (author).
Tier: data.

18. Split by fault family (hard episodes, detection-gated), FSI is the only method strong on every family.

| method | Franka actuator | Franka encoder | ANYmal actuator | ANYmal encoder | ANYmal foot | ANYmal IMU |
| --- | --- | --- | --- | --- | --- | --- |
| FSI | 0.830 | 0.865 | 0.930 | 0.894 | 0.533 | 0.814 |
| TranAD | 0.858 | 0.092 | 0.082 | 0.606 | 0.364 | 0.457 |
| GraphSL | 0.093 | 0.753 | 0.153 | 0.091 | 0.523 | 0.664 |
| FSI-RBC | 0.685 | 0.359 | 0.433 | 0.110 | 0.128 | 0.607 |
| GDN | 0.611 | 0.116 | 0.053 | 0.313 | 0.179 | 0.736 |
| `zz_constant` | 0.041 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |

Command: as for claims 12 to 15, on `franka_sensor` and `anymal_sensor`; the per-family split is `fsi.metrics.localization.per_family`.
Tolerance: TODO (author).
Tier: data.
Note: the `zz_constant` score of 1.000 on the ANYmal IMU column is an artefact, not a measurement. The IMU is a single node and the lowest-indexed candidate, so a constant scorer that always names it is right on every hard IMU episode by construction. It does not contaminate the totals because the generators draw the fault node uniformly before drawing the kind, which holds the IMU to 6.2% of episodes.

19. In the component sweep on `franka_v8` (3 seeds), the listwise loss is the consistent win: `hard_top1` rises from 0.325 ± 0.004 (per-node loss, no cross-node readout) to 0.351 ± 0.001 with the listwise loss, and to 0.360 ± 0.010 with both changes trained for 140 epochs.
Command: `python scripts/paper/fsi_sweep.py --data "$FSI_ROOT/data/franka_v8" --out "$FSI_ROOT/results/paper/fsi_sweep" --seeds 3`.
Tolerance: TODO (author).
Tier: data.

## Data generation claims

20. Every fault kind in the sensor corpora binds: it changes the robot's behaviour at its source by more than a typical node deviates, and the generators refuse to write a corpus that fails this check.
Command: `scripts/isaac/run_in_container.sh scripts/isaac/gen_franka_sensor.py --headless --check_only` and the same for `gen_anymal_sensor.py`.
Expected: the gate passes for every kind and writes `bind_check*.json` next to the corpus.
Tolerance: the realized local effect ratio criterion in [sensor_faults.md](sensor_faults.md).
Tier: simulator.
