"""Franka corpus v2: multi-task, randomized fault onset, and a wider actuator-fault catalogue.

Task-agnostic by construction: the recorded command is `joint_pos_target` (the controller's commanded joint
targets), which exists in every task whatever its action space, so f_theta conditions on intent rather than on
a task-specific action vector. Faults ramp in on a per-episode schedule (abrupt step or incipient ramp) at a
randomized onset, written every step, so nothing can key off a fixed injection time."""
import os, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])  # repository root (code)
ROOT = os.environ.get("FSI_ROOT", REPO)  # data/results root, see docs/CONFIGURATION.md
import argparse, json, os
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=256)
p.add_argument("--tasks", default="Isaac-Reach-Franka-v0,Isaac-Lift-Cube-Franka-v0,Isaac-Open-Drawer-Franka-v0")
p.add_argument("--healthy_per_task", type=int, default=3)
p.add_argument("--fault_per_task", type=int, default=5)
p.add_argument("--sweep_per_task", type=int, default=2)
p.add_argument("--steps", type=int, default=140)
p.add_argument("--out", default=f"{ROOT}/raw/franka_v2")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app
print("APP UP", flush=True)

import numpy as np
import torch
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

N, T, ARM = args.num_envs, args.steps, list(range(7))
KINDS = ["torque_loss", "friction", "jammed", "speed_limit"]
os.makedirs(args.out, exist_ok=True)
manifest = []

def schedule(onset, ramp, T, dev):
    """(T,N) fault progress in [0,1]: 0 before onset, then step (ramp=0) or linear ramp."""
    t = torch.arange(T, device=dev)[:, None].float()
    prog = (t - onset[None, :].float()) / torch.clamp(ramp[None, :].float(), min=1.0)
    return torch.clamp(prog, 0.0, 1.0) * (t >= onset[None, :].float())

def run_task(task, seed0, n_healthy, n_fault, rng):
    cfg = parse_env_cfg(task, num_envs=N)
    env = gym.make(task, cfg=cfg)
    u = env.unwrapped
    r = u.scene["robot"]
    adim = env.action_space.shape[1]
    dev = u.device
    idx = torch.arange(N, device=dev)
    PRISTINE_EFF = r.data.joint_effort_limits[:, ARM].clone()
    PRISTINE_DMP = r.data.joint_damping[:, ARM].clone() if hasattr(r.data, 'joint_damping') else None
    PRISTINE_VEL = r.data.joint_vel_limits[:, ARM].clone() if hasattr(r.data, 'joint_vel_limits') else None

    def rollout(seed, faults=None):
        env.reset(seed=seed)
        g = torch.Generator(device=dev).manual_seed(seed + 7)
        base_fr = 0.2 * torch.rand((N, 7), generator=g, device=dev)
        r.write_joint_friction_to_sim(base_fr, joint_ids=ARM)
        r.write_joint_effort_limit_to_sim(PRISTINE_EFF.clone(), joint_ids=ARM)   # reset() restores state, not sim props
        if PRISTINE_DMP is not None:
            r.write_joint_damping_to_sim(PRISTINE_DMP.clone(), joint_ids=ARM)
        if PRISTINE_VEL is not None:
            r.write_joint_velocity_limit_to_sim(PRISTINE_VEL.clone(), joint_ids=ARM)
        base_eff = PRISTINE_EFF.clone()
        base_dmp = None if PRISTINE_DMP is None else PRISTINE_DMP.clone()
        base_vel = None if PRISTINE_VEL is None else PRISTINE_VEL.clone()
        phase = torch.rand((N, adim), generator=g, device=dev) * 6.283
        freq = 0.03 + 0.06 * torch.rand((N, adim), generator=g, device=dev)
        amp = 0.4 + 0.4 * torch.rand((N, 1), generator=g, device=dev)
        import time as _t
        sig, cmd = [], []
        prog_all = faults[3] if faults is not None else None
        WRITE_EVERY = 10                     # each physics write costs ~seconds (CPU API refresh) -> few update points
        has_jam = has_spd = False
        first_onset = T
        t_write = 0.0
        t0 = _t.time()
        if faults is not None:
            kind, joint, sev, _ = faults
            has_jam = bool((kind == 2).any()); has_spd = bool((kind == 3).any())
            has_eff = bool(((kind == 0) | (kind == 2)).any()); has_fric = bool((kind == 1).any())
            first_onset = int((prog_all > 0).float().argmax(0).min().item())
        for t in range(T):
            if faults is not None and t >= first_onset and t % WRITE_EVERY == 0:
                _w0 = _t.time()
                w = (sev * prog_all[t])
                eff = base_eff.clone()
                fr = base_fr.clone()
                tgt = torch.clamp(demand_tau[joint] * (1.0 - 0.9 * w), min=0.02)
                tgt = torch.minimum(tgt, base_eff[idx, joint])
                eff[idx, joint] = torch.where(kind == 0, tgt, eff[idx, joint])
                eff[idx, joint] = torch.where(kind == 2, tgt, eff[idx, joint])
                fr[idx, joint] = torch.where(kind == 1, base_fr[idx, joint] + 8.0 * w, fr[idx, joint])
                if has_eff:
                    r.write_joint_effort_limit_to_sim(eff, joint_ids=ARM)
                if has_fric:
                    r.write_joint_friction_to_sim(fr, joint_ids=ARM)
                if has_jam and base_dmp is not None:
                    dmp = base_dmp.clone()
                    dmp[idx, joint] = torch.where(kind == 2, base_dmp[idx, joint] + 60.0 * w, dmp[idx, joint])
                    r.write_joint_damping_to_sim(dmp, joint_ids=ARM)
                if has_spd and base_vel is not None:
                    vl = base_vel.clone()
                    vtgt = torch.minimum(torch.clamp(demand_vel[joint] * (1.0 - 0.9 * w), min=0.02), base_vel[idx, joint])
                    vl[idx, joint] = torch.where(kind == 3, vtgt, vl[idx, joint])
                    r.write_joint_velocity_limit_to_sim(vl, joint_ids=ARM)
                t_write += _t.time() - _w0
            env.step(amp * torch.sin(freq * t + phase))
            sig.append(torch.stack([r.data.joint_pos[:, ARM], r.data.joint_vel[:, ARM],
                                    r.data.applied_torque[:, ARM]], -1).detach().cpu().numpy())
            cmd.append(r.data.joint_pos_target[:, ARM].detach().cpu().numpy())
        print(f"    rollout {_t.time()-t0:.1f}s (writes {t_write:.1f}s)", flush=True)
        return np.asarray(sig, np.float32), np.asarray(cmd, np.float32)

    # Calibrate faults against ACTUAL demand. Scaling the nominal limit is a no-op when a joint never approaches
    # it (Franka: 87 Nm limit vs ~1-15 Nm used), so the "fault" changed nothing and was unlocalizable by construction.
    s_cal, _ = rollout(seed0 + 900)
    demand_tau = torch.tensor(np.percentile(np.abs(s_cal[..., 2]), 90, axis=(0, 1)), dtype=torch.float32, device=dev)
    demand_vel = torch.tensor(np.percentile(np.abs(s_cal[..., 1]), 90, axis=(0, 1)), dtype=torch.float32, device=dev)
    print(f"{task} demand p90 torque {np.round(demand_tau.cpu().numpy(),2).tolist()}", flush=True)

    for b in range(n_healthy):
        s, c = rollout(seed0 + b)
        fn = f"{args.out}/{task}_healthy_{b:03d}.npz"
        np.savez_compressed(fn, signals=s.transpose(1, 0, 2, 3), actions=c.transpose(1, 0, 2))
        manifest.append({"file": os.path.basename(fn), "kind": "healthy", "task": task, "n": N})
        print(f"{task} healthy {b}", flush=True)

    for b in range(n_fault):
        kind = torch.full((N,), b % len(KINDS), dtype=torch.long, device=dev)  # one kind per batch -> one physics write
        joint = torch.tensor(rng.integers(0, 7, N), device=dev)
        sev = torch.tensor(rng.uniform(0.30, 0.90, N).astype(np.float32), device=dev)  # 40-80% remaining torque: the identifiable (incipient) band
        onset = torch.tensor(rng.integers(int(0.25 * T), int(0.65 * T), N), device=dev)
        ramp = torch.tensor(np.where(rng.random(N) < 0.5, 0, rng.integers(10, 40, N)), device=dev)
        prog = schedule(onset, ramp, T, dev)
        s, c = rollout(seed0 + 100 + b, faults=(kind, joint, sev, prog))
        labels = [{"kind": KINDS[int(kind[e])], "joint": int(joint[e]), "severity": float(sev[e]),
                   "onset": int(onset[e]), "ramp": int(ramp[e]), "task": task} for e in range(N)]
        fn = f"{args.out}/{task}_fault_{b:03d}.npz"
        np.savez_compressed(fn, signals=s.transpose(1, 0, 2, 3), actions=c.transpose(1, 0, 2),
                            labels=json.dumps(labels))
        manifest.append({"file": os.path.basename(fn), "kind": "fault", "task": task, "n": N})
        print(f"{task} fault {b}", flush=True)

    for b in range(args.sweep_per_task):
        # unit-fault sweep: fixed severity, systematic joint, varied commands -> a clean per-node fingerprint
        kind = torch.zeros(N, dtype=torch.long, device=dev)
        joint = torch.arange(N, device=dev) % 7
        sev = torch.full((N,), 0.70, device=dev)  # ~37% remaining torque: oracle top-1 = 1.000
        onset = torch.full((N,), T // 3, device=dev)
        ramp = torch.zeros(N, device=dev)
        S = seed0 + 500 + b
        s_ref, c_ref = rollout(S)                     # matched healthy reference: same seed -> same command+init
        s, c = rollout(S, faults=(kind, joint, sev, schedule(onset, ramp, T, dev)))
        labels = [{"kind": "torque_loss", "joint": int(joint[e]), "severity": 0.7, "onset": int(onset[e]),
                   "ramp": 0, "task": task, "sweep": True} for e in range(N)]
        fn = f"{args.out}/{task}_sweep_{b:03d}.npz"
        np.savez_compressed(fn, signals=s.transpose(1, 0, 2, 3), actions=c.transpose(1, 0, 2),
                            labels=json.dumps(labels))
        np.savez_compressed(f"{args.out}/{task}_sweepref_{b:03d}.npz",
                            signals=s_ref.transpose(1, 0, 2, 3), actions=c_ref.transpose(1, 0, 2),
                            labels=json.dumps(labels))
        manifest.append({"file": os.path.basename(fn), "kind": "sweep", "task": task, "n": N})
        print(f"{task} sweep {b} (+matched ref)", flush=True)
    env.close()

rng = np.random.default_rng(0)
for ti, task in enumerate(args.tasks.split(",")):
    try:
        run_task(task, 1000 * (ti + 1), args.healthy_per_task, args.fault_per_task, rng)
    except Exception as e:
        print(f"TASK {task} FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)

json.dump(manifest, open(f"{args.out}/manifest.json", "w"))
print(f"TOTAL episodes: {sum(m['n'] for m in manifest)}", flush=True)
app.close()
print("GEN_FRANKA_V2 DONE", flush=True)
