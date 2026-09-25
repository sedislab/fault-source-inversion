"""Confirm the ANYmal propagation story and nail the sensor API: record per-node signals (12 joints [pos,vel,
torque], IMU [tilt_x,tilt_y,yawrate], 4 feet [contact]), run matched healthy vs a hip torque-loss, and report
which node carries the loudest residual. If it is the base/foot (not the hip), the source is a hidden cause -> FSI's case."""
import argparse
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--steps", type=int, default=120)
p.add_argument("--onset", type=int, default=60)
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
cfg = parse_env_cfg("Isaac-Velocity-Flat-Anymal-C-v0", num_envs=N)
env = gym.make("Isaac-Velocity-Flat-Anymal-C-v0", cfg=cfg)
u = env.unwrapped
r = u.scene["robot"]
cs = u.scene.sensors["contact_forces"]
print("CONTACT bodies:", list(getattr(cs, "body_names", [])), "net_forces shape", tuple(cs.data.net_forces_w.shape), flush=True)
feet = [i for i, b in enumerate(getattr(cs, "body_names", [])) if "FOOT" in b.upper()]
print("FOOT indices in contact sensor:", feet, flush=True)
adim = env.action_space.shape[1]
dev = u.device
HIP = 4  # LF_HFE

def node_signals():
    jp = r.data.joint_pos; jv = r.data.joint_vel; jt = r.data.applied_torque   # (N,12)
    joints = torch.stack([jp, jv, jt], -1)                                      # (N,12,3)
    grav = r.data.projected_gravity_b                                           # (N,3)
    angv = r.data.root_ang_vel_b                                                # (N,3)
    imu = torch.stack([grav[:, 0], grav[:, 1], angv[:, 2]], -1)[:, None, :]     # (N,1,3)
    cf = cs.data.net_forces_w.norm(dim=-1)[:, feet]                             # (N,4)
    feetn = torch.stack([cf, torch.zeros_like(cf), torch.zeros_like(cf)], -1)   # (N,4,3)
    return torch.cat([joints, imu, feetn], 1).detach().cpu().numpy()            # (N,17,3)

def rollout(fault=False):
    env.reset(seed=0)
    base_eff = r.data.joint_effort_limits[:, :adim].clone()
    g = torch.Generator(device=dev).manual_seed(1)
    ph = torch.rand((N, adim), generator=g, device=dev) * 6.283
    rec = []
    for t in range(T):
        if fault and t == ONSET:
            eff = base_eff.clone(); eff[:, HIP] *= 0.05
            r.write_joint_effort_limit_to_sim(eff, joint_ids=list(range(adim)))
        env.step(0.2 * torch.sin(0.1 * t + ph))
        rec.append(node_signals())
    return np.asarray(rec)  # (T,N,17,3)

print("healthy...", flush=True); h = rollout(False)
print("hip fault...", flush=True); f = rollout(True)
w = slice(ONSET, T)
std = h[w].std(0) + 1e-6
resid = np.sqrt((((f[w] - h[w]) / std) ** 2).mean(0))       # (N,17,3)
energy = np.linalg.norm(resid, axis=-1)                     # (N,17)
names = [f"J{i}" for i in range(12)] + ["imu"] + [f"foot{i}" for i in range(4)]
loud = energy.mean(0).argmax()
print("mean energy per node:", {names[i]: round(float(energy.mean(0)[i]), 2) for i in range(17)}, flush=True)
print(f"SOURCE=J{HIP}(LF_HFE) energy={energy.mean(0)[HIP]:.2f}  LOUDEST={names[loud]} energy={energy.mean(0)[loud]:.2f}", flush=True)
print(f"cause-victim: loudest is source? {loud == HIP}", flush=True)
env.close(); app.close()
print("PROBE_ANYMAL2 DONE", flush=True)
