"""Probe ANYmal-C: joints, bodies, sensors (contact/IMU), action dim, and whether zero actions hold a standing
pose (so a hip torque loss buckles the leg and tilts the base -> propagation). Informs the quadruped data-gen."""
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

cfg = parse_env_cfg("Isaac-Velocity-Flat-Anymal-C-v0", num_envs=args.num_envs)
env = gym.make("Isaac-Velocity-Flat-Anymal-C-v0", cfg=cfg)
u = env.unwrapped
r = u.scene["robot"]
print("ACTION_DIM", env.action_space.shape, flush=True)
print("N_JOINTS", r.data.joint_pos.shape[1], "JOINTS", list(r.joint_names), flush=True)
print("BODIES", list(r.body_names), flush=True)
print("ACTUATORS", list(r.actuators.keys()), flush=True)
print("SCENE_SENSORS", list(u.scene.sensors.keys()), flush=True)
print("ROOT_DATA", [a for a in dir(r.data) if any(k in a for k in ("root_lin_vel", "root_ang_vel", "projected_gravity", "root_pos", "root_quat"))], flush=True)

obs, _ = env.reset()
adim = env.action_space.shape[1]
h0 = float(r.data.root_pos_w[0, 2].item()) if hasattr(r.data, "root_pos_w") else None
heights = []
for t in range(60):
    env.step(torch.zeros((u.num_envs, adim), device=u.device))
    if hasattr(r.data, "root_pos_w"):
        heights.append(float(r.data.root_pos_w[:, 2].mean().item()))
print("BASE_HEIGHT_START", round(h0, 3) if h0 else None, "TRACE", [round(x, 3) for x in heights[::10]], flush=True)
print("STANDS_UNDER_ZERO_ACTION", (heights[-1] > 0.3) if heights else "unknown", flush=True)
env.close()
app.close()
print("PROBE_ANYMAL DONE", flush=True)
