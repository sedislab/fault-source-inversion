"""Discover the exact articulation API on Franka: which methods write effort limits / stiffness, which data attrs
hold measured torque, and confirm we can record per-joint signals and drop a joint's torque. One job -> we know
how to record signals AND inject an actuator fault, which is everything the rollout collector needs."""
import argparse
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=16)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app
print("APP UP", flush=True)

import numpy as np
import torch
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

cfg = parse_env_cfg("Isaac-Reach-Franka-v0", num_envs=args.num_envs)
env = gym.make("Isaac-Reach-Franka-v0", cfg=cfg)
u = env.unwrapped
r = u.scene["robot"]
print("N_JOINTS", r.data.joint_pos.shape[1], "ACTION_DIM", env.action_space.shape, flush=True)
print("WRITE_METHODS", [m for m in dir(r) if m.startswith("write_") and ("joint" in m or "effort" in m or "stiff" in m or "damp" in m)], flush=True)
print("DATA_ATTRS", [m for m in dir(r.data) if any(k in m.lower() for k in ("torque", "effort", "applied", "computed", "joint_vel", "joint_pos"))], flush=True)
print("ACTUATOR_GROUPS", {k: {"joints": list(getattr(v, "joint_names", [])), "eff_lim_shape": tuple(v.effort_limit.shape) if hasattr(v, "effort_limit") and v.effort_limit is not None else None} for k, v in r.actuators.items()}, flush=True)

obs, _ = env.reset()
adim = env.action_space.shape[1]
tau, vel = [], []
for t in range(80):
    act = 0.3 * torch.sin(torch.full((u.num_envs, adim), 0.1 * t, device=u.device))
    env.step(act)
    for name, store in (("applied_torque", tau),):
        val = getattr(r.data, name, None)
        if val is not None:
            store.append(val[0].detach().cpu().numpy().copy())
    vel.append(r.data.joint_vel[0].detach().cpu().numpy().copy())
tau = np.asarray(tau) if tau else None
print("TORQUE_REC_SHAPE", None if tau is None else tau.shape, flush=True)
if tau is not None:
    print("MEAN_ABS_TORQUE_PER_JOINT", np.round(np.abs(tau).mean(0), 3).tolist(), flush=True)
print("MEAN_ABS_VEL_PER_JOINT", np.round(np.abs(np.asarray(vel)).mean(0), 3).tolist(), flush=True)

print("--- try effort-limit fault on joint 2 ---", flush=True)
try:
    lim = r.actuators["panda_shoulder"].effort_limit.clone()
    print("shoulder effort_limit before", lim[0].detach().cpu().numpy().tolist(), flush=True)
    lim[:] = lim * 0.05
    if hasattr(r, "write_joint_effort_limit_to_sim"):
        r.write_joint_effort_limit_to_sim(lim, joint_ids=r.actuators["panda_shoulder"].joint_indices)
        print("write_joint_effort_limit_to_sim OK", flush=True)
    else:
        print("no write_joint_effort_limit_to_sim; alt:", [m for m in dir(r) if "effort" in m.lower()], flush=True)
except Exception as e:
    print("effort-limit attempt error:", type(e).__name__, str(e)[:200], flush=True)

env.close()
app.close()
print("PROBE_FRANKA DONE", flush=True)
