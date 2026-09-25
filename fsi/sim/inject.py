"""Turning a FaultSpec into an actual intervention. Sensor and software faults corrupt the reported signal and
are realized here in pure numpy (a bias really is an added constant, a freeze really is a held value), so they
are exact and testable. Actuator, structural, and wrench faults intervene in physics through Isaac Lab; those
calls live in PhysicalInjector and are exercised on a GPU node (the API names are marked to verify against the
running container)."""
from __future__ import annotations
import numpy as np
from .faults import FaultKind, FaultSpec
from ..core.types import Onset

def corrupt_signal(y: np.ndarray, spec: FaultSpec, base_std: float, rng) -> np.ndarray:
    """Apply a sensor/software fault to one node's recorded time-series y:(T,). Faithful by construction."""
    y = y.copy(); t0 = spec.onset_step; T = len(y); k = spec.kind; sev = spec.severity
    if spec.onset == Onset.INCIPIENT:
        w = np.zeros(T); w[t0:] = np.linspace(0, 1, T - t0)
    else:
        w = np.zeros(T); w[t0:] = 1.0
    if k == FaultKind.SENSOR_BIAS:
        y += sev * base_std * w
    elif k == FaultKind.SENSOR_NOISE:
        y += rng.normal(0, sev * base_std, T) * w
    elif k == FaultKind.SENSOR_DRIFT:
        y += sev * base_std * np.cumsum(w) / max(1, T - t0)
    elif k == FaultKind.SENSOR_FREEZE:
        y[t0:] = y[t0 - 1] if t0 > 0 else y[0]
    elif k == FaultKind.SENSOR_DROPOUT:
        mask = (rng.random(T) < sev) & (w > 0)
        last = y[t0 - 1] if t0 > 0 else 0.0
        for t in range(T):
            if mask[t]:
                y[t] = last
            else:
                last = y[t]
    elif k == FaultKind.SENSOR_SCALE:
        y = y * (1 + (sev - 1) * w)
    elif k in (FaultKind.SENSOR_SPOOF, FaultKind.PERCEPTION_OOD, FaultKind.PREDICTION_ERROR,
               FaultKind.PLANNER_FAILURE, FaultKind.DETECTION_MISS):
        y += (sev * base_std) * w * np.sin(np.linspace(0, 6 * np.pi, T))
    return y

class PhysicalInjector:
    """Actuator/structural/wrench faults intervened in the Isaac Lab sim. `binding` maps a graph node to its
    sim handles (joint index, body index). Isaac Lab imported lazily so this module loads on CPU. API names
    tagged VERIFY are checked once against the running 2.3.1 container on a GPU node."""
    def __init__(self, env, binding: dict):
        self.env = env
        self.binding = binding
        self.robot = env.scene["robot"]

    def apply(self, spec: FaultSpec, env_ids):
        import torch
        b = self.binding[spec.node_name]
        r, jid, bid = self.robot, b.get("joint_id"), b.get("body_id")
        k, sev = spec.kind, spec.severity
        if k == FaultKind.TORQUE_LOSS:
            lim = r.actuators[b["actuator"]].effort_limit.clone()  # VERIFY: actuator effort_limit tensor
            lim[env_ids, jid] *= (1 - sev)
            r.write_joint_effort_limit_to_sim(lim, env_ids=env_ids)  # VERIFY method name
        elif k == FaultKind.LOCKED_JOINT:
            q = r.data.joint_pos[env_ids, jid].clone()
            r.write_joint_stiffness_to_sim(torch.full_like(q, 1e5), joint_ids=[jid], env_ids=env_ids)  # VERIFY
            r.set_joint_position_target(q.unsqueeze(-1), joint_ids=[jid], env_ids=env_ids)
        elif k == FaultKind.FREE_SWING:
            r.write_joint_effort_limit_to_sim(torch.zeros(len(env_ids), 1), env_ids=env_ids)  # VERIFY
        elif k in (FaultKind.FRICTION, FaultKind.DAMPING):
            d = r.data.joint_damping[env_ids, jid].clone() * sev
            r.write_joint_damping_to_sim(d.unsqueeze(-1), joint_ids=[jid], env_ids=env_ids)  # VERIFY
        elif k == FaultKind.MASS_CHANGE:
            m = r.root_physx_view.get_masses().clone()
            m[env_ids, bid] *= sev
            r.root_physx_view.set_masses(m, env_ids)  # VERIFY indexing
        elif k == FaultKind.COM_SHIFT:
            self._shift_com(bid, sev, env_ids)
        elif k == FaultKind.EXTERNAL_WRENCH:
            f = torch.zeros(len(env_ids), 1, 3); f[:, 0, 0] = sev
            r.set_external_force_and_torque(f, torch.zeros_like(f), body_ids=[bid], env_ids=env_ids)
            r.write_data_to_sim()
        elif k == FaultKind.ROTOR_THRUST_LOSS:
            self.env.rotor_scale[env_ids, b["rotor_id"]] = (1 - sev)  # custom mixer field on the drone env
        elif k == FaultKind.LIMB_LOSS:
            for j in b["limb_joints"]:
                r.write_joint_effort_limit_to_sim(torch.zeros(len(env_ids), 1), joint_ids=[j], env_ids=env_ids)

    def _shift_com(self, bid, sev, env_ids):
        coms = self.robot.root_physx_view.get_coms().clone()  # VERIFY
        coms[env_ids, bid, 0] += sev
        self.robot.root_physx_view.set_coms(coms, env_ids)
