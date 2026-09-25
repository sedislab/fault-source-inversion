"""Franka pilot for Gate G1: run matched healthy vs faulted rollouts under identical commands, inject a torque
loss on one joint per env at onset, and measure the cause-victim gap = fraction of envs where the loudest
residual joint is NOT the faulted one. This is the number the thesis rests on, measured on real dynamics."""
import os, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[3])  # repository root (code)
ROOT = os.environ.get("FSI_ROOT", REPO)  # data/results root, see docs/CONFIGURATION.md
import argparse
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=112)
p.add_argument("--steps", type=int, default=140)
p.add_argument("--onset", type=int, default=70)
p.add_argument("--severity", type=float, default=0.1)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app
print("APP UP", flush=True)

import numpy as np
import torch
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

N, T, ONSET = args.num_envs, args.steps, args.onset
cfg = parse_env_cfg("Isaac-Reach-Franka-v0", num_envs=N)
env = gym.make("Isaac-Reach-Franka-v0", cfg=cfg)
u = env.unwrapped
r = u.scene["robot"]
ARM = list(range(7))
adim = env.action_space.shape[1]
dev = u.device
gen = torch.Generator(device=dev).manual_seed(0)
phase = torch.rand((N, adim), generator=gen, device=dev) * 6.283
freq = 0.04 + 0.05 * torch.rand((N, adim), generator=gen, device=dev)
amp = 0.6

def action(t):
    return amp * torch.sin(freq * t + phase)

def rollout(fault_joint=None):
    env.reset(seed=0)
    base = r.data.joint_effort_limits[:, ARM].clone()
    rec = []
    for t in range(T):
        if fault_joint is not None and t == ONSET:
            lim = base.clone()
            lim[torch.arange(N, device=dev), fault_joint] *= args.severity
            r.write_joint_effort_limit_to_sim(lim, joint_ids=ARM)
        env.step(action(t))
        rec.append(torch.stack([r.data.joint_pos[:, ARM], r.data.joint_vel[:, ARM],
                                r.data.applied_torque[:, ARM]], -1).detach().cpu().numpy())
    return np.asarray(rec)  # (T, N, 7, 3)

print("healthy rollout...", flush=True)
healthy = rollout(None)
fault_joint = torch.arange(N, device=dev) % 7
print("faulted rollout...", flush=True)
faulted = rollout(fault_joint)
fj = fault_joint.cpu().numpy()

w = slice(ONSET, T)
h, f = healthy[w], faulted[w]
std = h.std(0) + 1e-6                       # (N,7,3) per-channel healthy scale
resid = np.sqrt((((f - h) / std) ** 2).mean(0))  # (N,7,3)
energy = np.linalg.norm(resid, axis=-1)     # (N,7)
loud = energy.argmax(1)
gap = float(np.mean(loud != fj))
print(f"FRANKA cause-victim gap (torque_loss sev={args.severity}): {gap:.3f}", flush=True)
per = {int(j): float(np.mean(loud[fj == j] != j)) for j in range(7)}
print("gap per fault joint:", {k: round(v, 2) for k, v in per.items()}, flush=True)
print("mean energy at source vs loudest:",
      round(float(np.mean(energy[np.arange(N), fj])), 3), round(float(np.mean(energy.max(1))), 3), flush=True)
out = f"{ROOT}/raw/franka_pilot.npz"
np.savez_compressed(out, healthy=healthy, faulted=faulted, fault_joint=fj, energy=energy)
print("saved", out, flush=True)
env.close()
app.close()
print("PILOT_FRANKA DONE", flush=True)
