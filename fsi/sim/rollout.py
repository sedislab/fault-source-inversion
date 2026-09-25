"""Rollout collection in Isaac Lab: run many parallel envs headless, inject one FaultSpec per env at its onset,
record each node's signal and the command, then post-process sensor faults and compute the residual field with
f_theta. This is the GPU data-generation entry point; the per-embodiment BINDINGS map graph nodes to the loaded
articulation's joint/body/sensor handles and are finalized by inspecting the robot on a GPU node (see the runbook).
"""
from __future__ import annotations
import numpy as np
from .faults import FaultCatalogue, CATALOGUE
from .inject import PhysicalInjector, corrupt_signal
from ..core.types import Episode, FaultLabel

BINDINGS = {
    "franka": {"tasks": "Isaac-Repose-Cube-Franka-v0 or Isaac-Lift-Cube-Franka-v0",
               "node_signal": "actuator J_i -> joint effort; sensor tau_i -> joint effort; wrist_ft -> ee wrench",
               "map": "J{i}->joint_id i-1, tau{i}->same joint effort channel, hand->ee body"},
    "anymal": {"tasks": "Isaac-Velocity-Flat-Anymal-C-v0",
               "node_signal": "actuator -> joint vel/effort; imu -> base ang/lin acc; foot -> contact force",
               "map": "{leg}_{HAA,HFE,KFE}->joint ids; imu->base IMU; {leg}_foot->contact sensor"},
    "crazyflie": {"tasks": "Isaac-Quadcopter-Direct-v0 (+ custom 4-rotor mixer env.rotor_scale)",
                  "node_signal": "rotor M_i -> commanded rpm; imu -> body acc/gyro; baro -> body z",
                  "map": "M{i}->mixer column i-1; imu/baro->body state"},
    "vehicle": {"sim": "CARLA (held out); node_signal per stratum: sensor readings, module outputs, actuator cmds"},
}

class RolloutCollector:
    """One embodiment. Call collect() to produce labeled Episodes. Requires a GPU + the Isaac Lab container."""
    def __init__(self, embodiment: str, forward_model, graph, catalogue: FaultCatalogue = CATALOGUE,
                 n_envs=1024, horizon=200, window=96, device="cuda"):
        self.embodiment = embodiment
        self.fm = forward_model
        self.graph = graph
        self.cat = catalogue
        self.n_envs = n_envs
        self.horizon = horizon
        self.window = window
        self.device = device
        self.binding = BINDINGS[embodiment]

    def _load_env(self):
        """Launch Isaac Lab headless and build the env for this embodiment. Imported lazily. See runbook."""
        from isaaclab.app import AppLauncher  # noqa: F401  (GPU-only)
        raise NotImplementedError("finalized on GPU: launch app, gym.make(task), attach PhysicalInjector")

    def _record_node_signals(self, env) -> np.ndarray:
        """Per-step (n_envs, n_nodes) primary signal per node from robot.data / sensors. Finalized on GPU."""
        raise NotImplementedError("finalized on GPU using BINDINGS[self.embodiment]")

    def collect(self, n_episodes, rng, k_dist=(0.34, 0.5, 0.16)) -> list[Episode]:
        """Flow (finalized on GPU): for each batch of n_envs, reset with domain randomization; sample k in
        {0,1,2} per env and a FaultSpec per fault; roll out horizon steps recording (u, y); apply physical
        faults at onset via PhysicalInjector, sensor/software faults via corrupt_signal; compute r with f_theta;
        emit Episode(embodiment, n, r, faults, y, yhat, u)."""
        raise NotImplementedError("GPU entry point; structure above, see docs/REPRODUCE.md")

    def _build_episode(self, y, yhat, u, faults) -> Episode:
        resid = y - yhat
        w = resid[-self.window:]
        r = np.stack([np.sqrt((w ** 2).mean(0)), np.abs(w).max(0), np.abs(w).mean(0)], 1).astype(np.float32)
        return Episode(self.embodiment, self.graph.n, r, faults, y=y, yhat=yhat, u=u,
                       meta={"onset_step": faults[0].onset_step if faults else 0})
