"""ANYmal-C corpus: the propagating case. The robot holds a stance under small commanded perturbations; a leg
actuator fault buckles that leg, tilts the base and redistributes foot contacts, so the fault's signature lands on
node types the source itself is not (IMU, feet). Records an 18-node field (base, IMU, 12 joints, 4 foot contacts),
randomized onsets (abrupt + incipient), four fault kinds, and matched-pair sweeps for a clean operator."""
import os, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])  # repository root (code)
ROOT = os.environ.get("FSI_ROOT", REPO)  # data/results root, see docs/CONFIGURATION.md
import argparse, json, os
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", default="Isaac-Velocity-Flat-Anymal-C-v0")
p.add_argument("--num_envs", type=int, default=256)
p.add_argument("--healthy_batches", type=int, default=3)
p.add_argument("--fault_batches", type=int, default=8)
p.add_argument("--sweep_batches", type=int, default=3)
p.add_argument("--steps", type=int, default=140)
p.add_argument("--out", default=f"{ROOT}/raw/anymal")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app
print("APP UP", flush=True)

import numpy as np
import torch
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

N, T = args.num_envs, args.steps
KINDS = ["torque_loss", "friction", "jammed", "speed_limit"]
LEGS = ("LF", "LH", "RF", "RH")
NJ, NNODE = 12, 18
os.makedirs(args.out, exist_ok=True)
manifest = []

cfg = parse_env_cfg(args.task, num_envs=N)
env = gym.make(args.task, cfg=cfg)
u = env.unwrapped
r = u.scene["robot"]
dev = u.device
adim = env.action_space.shape[1]
idx = torch.arange(N, device=dev)
JOINTS = list(range(NJ))
PRISTINE_EFF = r.data.joint_effort_limits[:, JOINTS].clone()
PRISTINE_DMP = r.data.joint_damping[:, JOINTS].clone() if hasattr(r.data, 'joint_damping') else None
PRISTINE_VEL = r.data.joint_vel_limits[:, JOINTS].clone() if hasattr(r.data, 'joint_vel_limits') else None
print("joints", list(r.joint_names), flush=True)

cs = None
try:
    cs = u.scene.sensors["contact_forces"]
except Exception as e:
    print("no contact sensor:", e, flush=True)
foot_cols = None
if cs is not None:
    names = list(getattr(cs, "body_names", []) or [])
    print("contact bodies", names, flush=True)
    cand = [next((i for i, n in enumerate(names) if n.upper().startswith(leg) and "FOOT" in n.upper()), None) for leg in LEGS]
    foot_cols = cand if all(c is not None for c in cand) else None
    if foot_cols is None:
        print("foot columns unresolved; feet read zero", flush=True)

def schedule(onset, ramp, dev):
    t = torch.arange(T, device=dev)[:, None].float()
    prog = (t - onset[None, :].float()) / torch.clamp(ramp[None, :].float(), min=1.0)
    return torch.clamp(prog, 0.0, 1.0) * (t >= onset[None, :].float())

def snapshot():
    pg, av = r.data.projected_gravity_b, r.data.root_ang_vel_b
    s = torch.zeros((N, NNODE, 3), device=dev)
    s[:, 0] = torch.stack([pg[:, 2], av[:, 0], av[:, 1]], -1)
    s[:, 1] = torch.stack([pg[:, 0], pg[:, 1], av[:, 2]], -1)
    s[:, 2:2 + NJ] = torch.stack([r.data.joint_pos[:, JOINTS], r.data.joint_vel[:, JOINTS],
                                  r.data.applied_torque[:, JOINTS]], -1)
    if cs is not None and foot_cols is not None:
        s[:, 2 + NJ:, 0] = torch.linalg.norm(cs.data.net_forces_w[:, foot_cols, :], dim=-1)
    return s

def rollout(seed, faults=None):
    env.reset(seed=seed)
    g = torch.Generator(device=dev).manual_seed(seed + 7)
    base_fr = 0.15 * torch.rand((N, NJ), generator=g, device=dev)
    r.write_joint_friction_to_sim(base_fr, joint_ids=JOINTS)
    r.write_joint_effort_limit_to_sim(PRISTINE_EFF.clone(), joint_ids=JOINTS)   # sim props persist across reset()
    if PRISTINE_DMP is not None:
        r.write_joint_damping_to_sim(PRISTINE_DMP.clone(), joint_ids=JOINTS)
    if PRISTINE_VEL is not None:
        r.write_joint_velocity_limit_to_sim(PRISTINE_VEL.clone(), joint_ids=JOINTS)
    base_eff = PRISTINE_EFF.clone()
    base_dmp = None if PRISTINE_DMP is None else PRISTINE_DMP.clone()
    base_vel = None if PRISTINE_VEL is None else PRISTINE_VEL.clone()
    phase = torch.rand((N, adim), generator=g, device=dev) * 6.283
    freq = 0.02 + 0.05 * torch.rand((N, adim), generator=g, device=dev)
    amp = 0.20 + 0.35 * torch.rand((N, 1), generator=g, device=dev)   # load the legs: weak excitation left most faults undetectable
    WRITE_EVERY, sig, cmd, first = 10, [], [], T
    if faults is not None:
        kind, joint, sev, prog = faults
        has_eff = bool(((kind == 0) | (kind == 2)).any()); has_fr = bool((kind == 1).any())
        has_jam = bool((kind == 2).any()); has_spd = bool((kind == 3).any())
        first = int((prog > 0).float().argmax(0).min().item())
    for t in range(T):
        if faults is not None and t >= first and t % WRITE_EVERY == 0:
            w = sev * prog[t]
            eff, fr = base_eff.clone(), base_fr.clone()
            tgt = torch.minimum(torch.clamp(demand_tau[joint] * (1.0 - 0.9 * w), min=0.02), base_eff[idx, joint])
            eff[idx, joint] = torch.where(kind == 0, tgt, eff[idx, joint])
            eff[idx, joint] = torch.where(kind == 2, tgt, eff[idx, joint])
            fr[idx, joint] = torch.where(kind == 1, base_fr[idx, joint] + 8.0 * w, fr[idx, joint])
            if has_eff:
                r.write_joint_effort_limit_to_sim(eff, joint_ids=JOINTS)
            if has_fr:
                r.write_joint_friction_to_sim(fr, joint_ids=JOINTS)
            if has_jam and base_dmp is not None:
                d = base_dmp.clone(); d[idx, joint] = torch.where(kind == 2, base_dmp[idx, joint] + 60.0 * w, d[idx, joint])
                r.write_joint_damping_to_sim(d, joint_ids=JOINTS)
            if has_spd and base_vel is not None:
                v = base_vel.clone()
                vtgt = torch.minimum(torch.clamp(demand_vel[joint] * (1.0 - 0.9 * w), min=0.02), base_vel[idx, joint])
                v[idx, joint] = torch.where(kind == 3, vtgt, v[idx, joint])
                r.write_joint_velocity_limit_to_sim(v, joint_ids=JOINTS)
        env.step(amp * torch.sin(freq * t + phase))
        sig.append(snapshot().detach().cpu().numpy())
        cmd.append(r.data.joint_pos_target[:, JOINTS].detach().cpu().numpy())
    return np.asarray(sig, np.float32), np.asarray(cmd, np.float32)

def save(tag, b, s, c, labels=None):
    fn = f"{args.out}/anymal_{tag}_{b:03d}.npz"
    kw = {"labels": json.dumps(labels)} if labels else {}
    np.savez_compressed(fn, signals=s.transpose(1, 0, 2, 3), actions=c.transpose(1, 0, 2), **kw)
    manifest.append({"file": os.path.basename(fn), "kind": tag, "n": N})
    print(f"anymal {tag} {b}", flush=True)

# Calibrate faults against ACTUAL demand (see franka note): nominal-limit scaling leaves faults non-binding.
_scal, _ = rollout(900)
demand_tau = torch.tensor(np.percentile(np.abs(_scal[:, :, 2:2 + NJ, 2]), 90, axis=(0, 1)), dtype=torch.float32, device=dev)
demand_vel = torch.tensor(np.percentile(np.abs(_scal[:, :, 2:2 + NJ, 1]), 90, axis=(0, 1)), dtype=torch.float32, device=dev)
print("demand p90 torque", np.round(demand_tau.cpu().numpy(), 2).tolist(), flush=True)

rng = np.random.default_rng(0)
for b in range(args.healthy_batches):
    s, c = rollout(b); save("healthy", b, s, c)

for b in range(args.fault_batches):
    kind = torch.full((N,), b % len(KINDS), dtype=torch.long, device=dev)
    joint = torch.tensor(rng.integers(0, NJ, N), device=dev)
    sev = torch.tensor(rng.uniform(0.30, 0.90, N).astype(np.float32), device=dev)  # identifiable band
    onset = torch.tensor(rng.integers(int(0.25 * T), int(0.65 * T), N), device=dev)
    ramp = torch.tensor(np.where(rng.random(N) < 0.5, 0, rng.integers(10, 40, N)), device=dev)
    s, c = rollout(100 + b, faults=(kind, joint, sev, schedule(onset, ramp, dev)))
    save("fault", b, s, c, [{"kind": KINDS[int(kind[e])], "joint": int(joint[e]) + 2, "severity": float(sev[e]),
                             "onset": int(onset[e]), "ramp": int(ramp[e]), "task": args.task} for e in range(N)])

for b in range(args.sweep_batches):
    S = 500 + b
    kind = torch.zeros(N, dtype=torch.long, device=dev)
    joint = torch.arange(N, device=dev) % NJ
    sev = torch.full((N,), 0.70, device=dev)  # ~37% remaining torque: oracle top-1 = 1.000
    onset = torch.full((N,), T // 3, device=dev)
    ramp = torch.zeros(N, device=dev)
    s_ref, c_ref = rollout(S)
    s, c = rollout(S, faults=(kind, joint, sev, schedule(onset, ramp, dev)))
    lab = [{"kind": "torque_loss", "joint": int(joint[e]) + 2, "severity": 0.7, "onset": int(onset[e]),
            "ramp": 0, "task": args.task, "sweep": True} for e in range(N)]
    save("sweep", b, s, c, lab); save("sweepref", b, s_ref, c_ref, lab)

json.dump(manifest, open(f"{args.out}/manifest.json", "w"))
print(f"TOTAL episodes: {sum(m['n'] for m in manifest)}", flush=True)
env.close(); app.close()
print("GEN_ANYMAL DONE", flush=True)
