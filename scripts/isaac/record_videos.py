"""Record labeled videos: one healthy rollout and one per fault type, with the condition, target joint, severity
and onset burned into each frame plus a JSON manifest. Needs --enable_cameras (rendering) so run on an RT-core GPU."""
import os, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])  # repository root (code)
ROOT = os.environ.get("FSI_ROOT", REPO)  # data/results root, see docs/CONFIGURATION.md
import argparse, json, os
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", default="Isaac-Reach-Franka-v0")
p.add_argument("--joint", type=int, default=1)
p.add_argument("--severity", type=float, default=0.9)
p.add_argument("--steps", type=int, default=140)
p.add_argument("--onset", type=int, default=60)
p.add_argument("--num_envs", type=int, default=2)
p.add_argument("--out", default=f"{ROOT}/videos/franka")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app
print("APP UP", flush=True)

import numpy as np
import torch
import imageio.v2 as imageio
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

KINDS = ["torque_loss", "friction", "jammed", "speed_limit"]
ARM = list(range(7))
os.makedirs(args.out, exist_ok=True)
cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
env = gym.make(args.task, cfg=cfg, render_mode="rgb_array")
u = env.unwrapped
r = u.scene["robot"]
adim = env.action_space.shape[1]
dev = u.device
idx = torch.arange(args.num_envs, device=dev)

def annotate(frame, lines):
    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray(frame)
        d = ImageDraw.Draw(im)
        for i, ln in enumerate(lines):
            d.text((10, 10 + 16 * i), ln, fill=(255, 255, 0))
        return np.asarray(im)
    except Exception:
        return frame

def rollout(kind_idx, tag):
    env.reset(seed=3)
    g = torch.Generator(device=dev).manual_seed(11)
    base_eff = r.data.joint_effort_limits[:, ARM].clone()
    base_fr = torch.zeros((args.num_envs, 7), device=dev)
    base_dmp = r.data.joint_damping[:, ARM].clone() if hasattr(r.data, "joint_damping") else None
    base_vel = r.data.joint_vel_limits[:, ARM].clone() if hasattr(r.data, "joint_vel_limits") else None
    phase = torch.rand((args.num_envs, adim), generator=g, device=dev) * 6.283
    freq = 0.05 + 0.03 * torch.rand((args.num_envs, adim), generator=g, device=dev)
    frames = []
    for t in range(args.steps):
        active = kind_idx is not None and t >= args.onset
        if kind_idx is not None:
            w = args.severity if active else 0.0
            eff = base_eff.clone(); fr = base_fr.clone()
            if kind_idx == 0:
                eff[idx, args.joint] = base_eff[idx, args.joint] * (1 - 0.95 * w)
            elif kind_idx == 1:
                fr[idx, args.joint] = base_fr[idx, args.joint] + 8.0 * w
            elif kind_idx == 2:
                eff[idx, args.joint] = base_eff[idx, args.joint] * (1 - 0.9 * w)
            r.write_joint_effort_limit_to_sim(eff, joint_ids=ARM)
            r.write_joint_friction_to_sim(fr, joint_ids=ARM)
            if kind_idx == 2 and base_dmp is not None:
                dmp = base_dmp.clone(); dmp[idx, args.joint] = base_dmp[idx, args.joint] + 60.0 * w
                r.write_joint_damping_to_sim(dmp, joint_ids=ARM)
            if kind_idx == 3 and base_vel is not None:
                vl = base_vel.clone(); vl[idx, args.joint] = base_vel[idx, args.joint] * (1 - 0.9 * w)
                r.write_joint_velocity_limit_to_sim(vl, joint_ids=ARM)
        env.step(0.6 * torch.sin(freq * t + phase))
        f = env.render()
        if f is not None:
            frames.append(annotate(np.asarray(f), [
                f"FSI | {args.task}", f"condition: {tag}",
                "" if kind_idx is None else f"fault: {KINDS[kind_idx]} @ joint J{args.joint+1} sev={args.severity}",
                f"t={t}  " + ("FAULT ACTIVE" if active else "healthy"),
            ]))
    return frames

manifest = []
for kind_idx, tag in [(None, "healthy")] + [(i, KINDS[i]) for i in range(len(KINDS))]:
    frames = rollout(kind_idx, tag)
    if not frames:
        print(f"no frames for {tag} (rendering unavailable)", flush=True)
        continue
    fn = f"{args.out}/{args.task}_{tag}" + ("" if kind_idx is None else f"_J{args.joint+1}") + ".mp4"
    imageio.mimsave(fn, frames, fps=30)
    manifest.append({"file": os.path.basename(fn), "condition": tag, "task": args.task,
                     "joint": None if kind_idx is None else args.joint,
                     "severity": None if kind_idx is None else args.severity,
                     "onset_step": None if kind_idx is None else args.onset, "fps": 30, "steps": args.steps})
    print(f"saved {fn} ({len(frames)} frames)", flush=True)

json.dump(manifest, open(f"{args.out}/manifest.json", "w"), indent=2)
env.close()
app.close()
print("RECORD_VIDEOS DONE", flush=True)
