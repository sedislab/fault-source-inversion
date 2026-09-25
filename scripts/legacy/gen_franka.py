"""Generate the raw Franka corpus: healthy rollouts (varied commands + randomized joint friction) and faulted
rollouts (actuator faults injected at onset), recording per-joint [pos, vel, torque] and the command. Saved raw;
a later stage trains f_theta, computes residual fields, sweeps the operator G, and assembles labeled episodes."""
import os, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])  # repository root (code)
ROOT = os.environ.get("FSI_ROOT", REPO)  # data/results root, see docs/CONFIGURATION.md
import argparse, json, os
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=256)
p.add_argument("--healthy_batches", type=int, default=4)
p.add_argument("--fault_batches", type=int, default=6)
p.add_argument("--steps", type=int, default=140)
p.add_argument("--onset", type=int, default=70)
p.add_argument("--out", default=f"{ROOT}/raw/franka")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app
print("APP UP", flush=True)

import numpy as np
import torch
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

N, T, ONSET, ARM = args.num_envs, args.steps, args.onset, list(range(7))
os.makedirs(args.out, exist_ok=True)
cfg = parse_env_cfg("Isaac-Reach-Franka-v0", num_envs=N)
env = gym.make("Isaac-Reach-Franka-v0", cfg=cfg)
u = env.unwrapped
r = u.scene["robot"]
adim = env.action_space.shape[1]
dev = u.device
FAULT_KINDS = ["torque_loss", "friction"]

def commands(seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    phase = torch.rand((N, adim), generator=g, device=dev) * 6.283
    freq = 0.03 + 0.06 * torch.rand((N, adim), generator=g, device=dev)
    amp = 0.4 + 0.4 * torch.rand((N, 1), generator=g, device=dev)
    acts = torch.stack([amp * torch.sin(freq * t + phase) for t in range(T)])  # (T,N,7)
    return acts

def rollout(acts, seed, faults=None):
    env.reset(seed=seed)
    g = torch.Generator(device=dev).manual_seed(seed + 999)
    base_fr = 0.2 * torch.rand((N, 7), generator=g, device=dev)      # domain-randomized friction, applied post-reset
    r.write_joint_friction_to_sim(base_fr, joint_ids=ARM)
    base_eff = r.data.joint_effort_limits[:, ARM].clone()
    idx = torch.arange(N, device=dev)
    rec = []
    for t in range(T):
        if faults is not None and t == ONSET:
            kind, joint, sev = faults
            if_tl = kind == 0
            eff = base_eff.clone()
            fr = base_fr.clone()
            eff[idx, joint] = torch.where(if_tl, base_eff[idx, joint] * (1 - sev), base_eff[idx, joint])
            fr[idx, joint] = torch.where(if_tl, base_fr[idx, joint], base_fr[idx, joint] + 2.0 + 6.0 * sev)
            r.write_joint_effort_limit_to_sim(eff, joint_ids=ARM)
            r.write_joint_friction_to_sim(fr, joint_ids=ARM)
        env.step(acts[t])
        rec.append(torch.stack([r.data.joint_pos[:, ARM], r.data.joint_vel[:, ARM],
                                r.data.applied_torque[:, ARM]], -1).detach().cpu().numpy())
    return np.asarray(rec, np.float32)  # (T,N,7,3)

manifest = []
for b in range(args.healthy_batches):
    acts = commands(b)
    sig = rollout(acts, seed=b)
    fn = f"{args.out}/healthy_{b:03d}.npz"
    np.savez_compressed(fn, signals=sig.transpose(1, 0, 2, 3), actions=acts.cpu().numpy().transpose(1, 0, 2))
    manifest.append({"file": os.path.basename(fn), "kind": "healthy", "n": N})
    print(f"healthy batch {b} saved", flush=True)

rng = np.random.default_rng(0)
for b in range(args.fault_batches):
    acts = commands(100 + b)
    kind = torch.tensor(rng.integers(0, len(FAULT_KINDS), N), device=dev)
    joint = torch.tensor(rng.integers(0, 7, N), device=dev)
    sev = torch.tensor(rng.uniform(0.4, 0.95, N).astype(np.float32), device=dev)
    sig = rollout(acts, seed=100 + b, faults=(kind, joint, sev))
    labels = [{"kind": FAULT_KINDS[int(kind[e])], "joint": int(joint[e]), "severity": float(sev[e]),
               "onset": ONSET} for e in range(N)]
    fn = f"{args.out}/fault_{b:03d}.npz"
    np.savez_compressed(fn, signals=sig.transpose(1, 0, 2, 3), actions=acts.cpu().numpy().transpose(1, 0, 2),
                        labels=json.dumps(labels))
    manifest.append({"file": os.path.basename(fn), "kind": "fault", "n": N})
    print(f"fault batch {b} saved", flush=True)

json.dump(manifest, open(f"{args.out}/manifest.json", "w"))
print(f"TOTAL episodes: {(args.healthy_batches + args.fault_batches) * N}", flush=True)
env.close()
app.close()
print("GEN_FRANKA DONE", flush=True)
