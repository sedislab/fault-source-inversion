"""ANYmal-C corpus, v2 -- the version where the faults actually bind.

v1 (gen_anymal.py) injected actuator faults with `write_joint_effort_limit_to_sim` / `write_joint_friction_to_sim`.
Both are no-ops on ANYmal-C: it uses an EXPLICIT ANYdrive actuator model that computes torque in Python and never
consults PhysX's effort limit. Measured at severity 0.8 on LF_HFE (scripts/isaac/probes/probe_actuator.py):

    mechanism                  |tau_src| drop   episodes affected
    physx_effort_limit (v1)         4.0%            0.438      <- a coin flip
    actuator_saturation             4.0%            0.438
    actuator_effort_limit          31.7%            0.688
    compute_hook                   60.6%            0.969      <- the only one that binds

Consequence for v1's corpus: the source joint's post-onset mean |tau| was 17.638 faulted vs 17.570 on its matched
reference, so a fully-supervised deep model with ground-truth labels reached only top1 0.190 (chance 0.083). On the
two kinds that DID bite (speed_limit, jammed-via-damping) the same model reached 0.442.

v2 changes, all three necessary:
  1. Torque faults go through a hook on the actuator's compute(), clamping the returned effort directly.
  2. Sweep pairs share an exact initial state. In v1 the reference and faulted rollouts merely shared a seed and
     diverged from step 0 -- 59% of the "oracle counterfactual" was non-fault divergence (Franka's figure is 7%),
     so the operator G was fit largely on noise.
  3. A generation-time assertion: the source joint's torque (or velocity) must drop materially versus its matched
     reference, per kind. v1 would have been caught immediately by this check."""
import os, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])  # repository root (code)
ROOT = os.environ.get("FSI_ROOT", REPO)  # data/results root, see docs/CONFIGURATION.md
import argparse, json, os, sys
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", default="Isaac-Velocity-Flat-Anymal-C-v0")
p.add_argument("--num_envs", type=int, default=256)
p.add_argument("--healthy_batches", type=int, default=4)
p.add_argument("--fault_batches", type=int, default=10)
p.add_argument("--sweep_batches", type=int, default=3)
p.add_argument("--steps", type=int, default=140)
p.add_argument("--min_drop", type=float, default=0.20, help="unused; kept so old invocations still parse")
p.add_argument("--check_only", action="store_true", help="run the bind check and stop, before spending hours")
p.add_argument("--out", default=f"{ROOT}/raw/anymal_v4")
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
PRISTINE_DMP = r.data.joint_damping[:, JOINTS].clone() if hasattr(r.data, "joint_damping") else None
PRISTINE_VEL = r.data.joint_vel_limits[:, JOINTS].clone() if hasattr(r.data, "joint_vel_limits") else None
print("joints", list(r.joint_names), flush=True)

# ---- the fault hook: clamp the effort the actuator actually returns ----------------------------------------
ACT = next(iter(r.actuators.values()))
print(f"actuator: {type(ACT).__name__}  implicit={getattr(ACT, 'is_implicit_model', 'n/a')}", flush=True)
ACT._fsi_scale = None      # (N,NJ) multiplicative torque retention, or None
ACT._fsi_coulomb = None    # (N,NJ) opposing torque magnitude, or None
ACT._fsi_vlim = None       # (N,NJ) speed past which torque derates, or None
_orig_compute = ACT.compute

def _hooked_compute(control_action, joint_pos, joint_vel, _o=_orig_compute):
    out = _o(control_action, joint_pos, joint_vel)
    if ACT._fsi_scale is not None:
        out.joint_efforts[:, :NJ] = out.joint_efforts[:, :NJ] * ACT._fsi_scale
    if ACT._fsi_coulomb is not None:
        out.joint_efforts[:, :NJ] -= ACT._fsi_coulomb * torch.tanh(10.0 * joint_vel[:, :NJ])
    if ACT._fsi_vlim is not None:
        # A speed-limited actuator does not stop at its limit, it loses torque past it -- a DC motor's available
        # torque falls toward zero at no-load speed. Writing PhysX's joint velocity limit is bypassed by this
        # explicit actuator exactly as the effort limit is (measured: effect ratio 1.33x, only 69% of episodes),
        # so the derate has to happen on the effort this model returns.
        over = torch.clamp(joint_vel[:, :NJ].abs() - ACT._fsi_vlim, min=0.0)
        out.joint_efforts[:, :NJ] = out.joint_efforts[:, :NJ] * torch.clamp(1.0 - over / (ACT._fsi_vlim + 1e-3), 0.0, 1.0)
    return out

ACT.compute = _hooked_compute

cs = None
try:
    cs = u.scene.sensors["contact_forces"]
except Exception as e:
    print("no contact sensor:", e, flush=True)
foot_cols = None
if cs is not None:
    names = list(getattr(cs, "body_names", []) or [])
    cand = [next((i for i, n in enumerate(names) if n.upper().startswith(leg) and "FOOT" in n.upper()), None) for leg in LEGS]
    foot_cols = cand if all(c is not None for c in cand) else None

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

def rollout(seed, faults=None, state=None):
    """One rollout. `state`, if given, is written back after reset so a faulted run starts from a bit-identical
    configuration to its reference -- the whole point of a matched pair. Returns signals, commands, and the
    captured post-reset state so a paired run can reuse it."""
    ACT._fsi_scale = ACT._fsi_coulomb = ACT._fsi_vlim = None
    env.reset(seed=seed)
    if state is not None:
        root, jp, jv = state
        r.write_root_state_to_sim(root.clone())
        r.write_joint_state_to_sim(jp.clone(), jv.clone())
    captured = (r.data.root_state_w.clone(), r.data.joint_pos.clone(), r.data.joint_vel.clone())
    g = torch.Generator(device=dev).manual_seed(seed + 7)
    base_fr = 0.15 * torch.rand((N, NJ), generator=g, device=dev)
    r.write_joint_friction_to_sim(base_fr, joint_ids=JOINTS)
    r.write_joint_effort_limit_to_sim(PRISTINE_EFF.clone(), joint_ids=JOINTS)
    if PRISTINE_DMP is not None:
        r.write_joint_damping_to_sim(PRISTINE_DMP.clone(), joint_ids=JOINTS)
    if PRISTINE_VEL is not None:
        r.write_joint_velocity_limit_to_sim(PRISTINE_VEL.clone(), joint_ids=JOINTS)
    base_dmp = None if PRISTINE_DMP is None else PRISTINE_DMP.clone()
    base_vel = None if PRISTINE_VEL is None else PRISTINE_VEL.clone()
    phase = torch.rand((N, adim), generator=g, device=dev) * 6.283
    freq = 0.02 + 0.05 * torch.rand((N, adim), generator=g, device=dev)
    amp = 0.20 + 0.35 * torch.rand((N, 1), generator=g, device=dev)
    sig, cmd = [], []
    if faults is not None:
        kind, joint, sev, prog = faults
        has_jam = bool((kind == 2).any())
    for t in range(T):
        if faults is not None:
            w = sev * prog[t]                                    # realized severity this step, in [0, sev]
            scale = torch.ones((N, NJ), device=dev)
            coul = torch.zeros((N, NJ), device=dev)
            # torque_loss and the effort half of jammed: retain (1 - 0.9w) of the commanded torque. This is the
            # only mechanism that binds -- PhysX's effort limit is bypassed by the explicit actuator model.
            tl = (kind == 0) | (kind == 2)
            scale[idx, joint] = torch.where(tl, 1.0 - 0.9 * w, scale[idx, joint])
            # friction as an opposing Coulomb torque scaled by the joint's own demand, again inside the actuator.
            coul[idx, joint] = torch.where(kind == 1, 0.9 * w * demand_tau[joint], coul[idx, joint])
            # speed_limit: derate torque past a reduced speed, in the actuator. HUGE default for every other
            # joint so the derate is inert there.
            vlim = torch.full((N, NJ), 1e6, device=dev)
            vlim[idx, joint] = torch.where(kind == 3, torch.clamp(demand_vel[joint] * (1.0 - 0.9 * w), min=0.02),
                                           vlim[idx, joint])
            ACT._fsi_scale, ACT._fsi_coulomb, ACT._fsi_vlim = scale, coul, vlim
            if t % 10 == 0 and has_jam and base_dmp is not None:
                d = base_dmp.clone()
                d[idx, joint] = torch.where(kind == 2, base_dmp[idx, joint] + 60.0 * w, d[idx, joint])
                r.write_joint_damping_to_sim(d, joint_ids=JOINTS)
        env.step(amp * torch.sin(freq * t + phase))
        sig.append(snapshot().detach().cpu().numpy())
        cmd.append(r.data.joint_pos_target[:, JOINTS].detach().cpu().numpy())
    ACT._fsi_scale = ACT._fsi_coulomb = ACT._fsi_vlim = None
    return np.asarray(sig, np.float32), np.asarray(cmd, np.float32), captured

def save(tag, b, s, c, labels=None):
    fn = f"{args.out}/anymal_{tag}_{b:03d}.npz"
    kw = {"labels": json.dumps(labels)} if labels else {}
    np.savez_compressed(fn, signals=s.transpose(1, 0, 2, 3), actions=c.transpose(1, 0, 2), **kw)
    manifest.append({"file": os.path.basename(fn), "kind": tag, "n": N})
    print(f"anymal {tag} {b}", flush=True)

_scal, _, _ = rollout(900)
demand_tau = torch.tensor(np.percentile(np.abs(_scal[:, :, 2:2 + NJ, 2]), 90, axis=(0, 1)), dtype=torch.float32, device=dev)
# MEDIAN, not p90, for the speed reference. A limit set at a fraction of the 90th-percentile speed only engages
# in the top decile of motion, so the fault acts intermittently and lands on ~78% of episodes; referencing the
# median makes it bite whenever the joint is actually moving.
demand_vel = torch.tensor(np.percentile(np.abs(_scal[:, :, 2:2 + NJ, 1]), 50, axis=(0, 1)), dtype=torch.float32, device=dev)
print("demand p90 torque", np.round(demand_tau.cpu().numpy(), 2).tolist(), flush=True)

# ---- GATE: every fault kind must make its OWN joint the standout deviator ----------------------------------
# The criterion is the realized LOCAL EFFECT RATIO, not a torque "drop". A signed drop is the wrong test: adding
# Coulomb friction makes the actuator work harder to track its target, so |tau| RISES (measured -17.2% "drop"),
# and a velocity fault shows up in the velocity channel rather than the torque channel at all. What every kind
# must do, whatever its sign or channel, is move its own joint more than a typical joint moves. That is exactly
# the quantity localization depends on, so gating on it gates on the thing that actually matters.
print("\n=== BIND CHECK (local effect ratio, severity 0.7, joint 4) ===", flush=True)
_on = torch.full((N,), T // 3, device=dev)
_ref, _, _st = rollout(777)
on = int(T // 3)
_sd = _ref[on:].std(axis=(0, 1)) + 1e-6                            # (NNODE,3) per-node/channel scale
bind = {}
for ki, kname in enumerate(KINDS):
    kind = torch.full((N,), ki, dtype=torch.long, device=dev)
    joint = torch.full((N,), 4, dtype=torch.long, device=dev)
    sev = torch.full((N,), 0.70, device=dev)
    s_f, _, _ = rollout(777, faults=(kind, joint, sev, schedule(_on, torch.zeros(N, device=dev), dev)), state=_st)
    z = np.abs(s_f[on:] - _ref[on:]) / _sd                         # (T',N,NNODE,3) scale-free deviation
    per_joint = z[:, :, 2:2 + NJ, :].mean(axis=(0, 3))             # (N,NJ) per-episode effect at each joint
    src = per_joint[:, 4]
    oth = np.median(per_joint, axis=1)
    ratio = src / (oth + 1e-9)
    med, frac = float(np.median(ratio)), float((ratio > 1.0).mean())
    bind[kname] = (med, frac)
    print(f"  {kname:12s} median effect ratio {med:5.2f}x   episodes with source above median joint {frac:.3f}",
          flush=True)
bad = [k for k, (m, f) in bind.items() if m < 1.5 or f < 0.9]
if bad:
    print(f"\nFAIL: these kinds do not localize: {bad}. In v1's corpus the median ratio was ~1.00 for every kind "
          f"-- the source joint deviated no more than a random joint -- which is why a fully-supervised model "
          f"topped out at 0.190. Refusing to generate a corpus whose labels are unrecoverable.", flush=True)
    env.close(); app.close(); sys.exit(2)
print("BIND CHECK PASSED -- every kind makes its own joint the standout deviator\n", flush=True)
if args.check_only:
    env.close(); app.close(); sys.exit(0)

rng = np.random.default_rng(0)
for b in range(args.healthy_batches):
    s, c, _ = rollout(b); save("healthy", b, s, c)

for b in range(args.fault_batches):
    kind = torch.full((N,), b % len(KINDS), dtype=torch.long, device=dev)
    joint = torch.tensor(rng.integers(0, NJ, N), device=dev)
    sev = torch.tensor(rng.uniform(0.30, 0.90, N).astype(np.float32), device=dev)
    onset = torch.tensor(rng.integers(int(0.25 * T), int(0.65 * T), N), device=dev)
    ramp = torch.tensor(np.where(rng.random(N) < 0.5, 0, rng.integers(10, 40, N)), device=dev)
    s, c, _ = rollout(100 + b, faults=(kind, joint, sev, schedule(onset, ramp, dev)))
    save("fault", b, s, c, [{"kind": KINDS[int(kind[e])], "joint": int(joint[e]) + 2, "severity": float(sev[e]),
                             "onset": int(onset[e]), "ramp": int(ramp[e]), "task": args.task} for e in range(N)])

for b in range(args.sweep_batches):
    S = 500 + b
    kind = torch.zeros(N, dtype=torch.long, device=dev)
    joint = torch.arange(N, device=dev) % NJ
    sev = torch.full((N,), 0.70, device=dev)
    onset = torch.full((N,), T // 3, device=dev)
    ramp = torch.zeros(N, device=dev)
    # Reference FIRST, then the faulted twin restored to the reference's exact post-reset state.
    s_ref, c_ref, st = rollout(S)
    s, c, _ = rollout(S, faults=(kind, joint, sev, schedule(onset, ramp, dev)), state=st)
    on = int(T // 3)
    pre = np.abs(s[:on] - s_ref[:on]).mean()
    post = np.abs(s[on:] - s_ref[on:]).mean()
    print(f"  sweep {b}: pre-onset divergence {pre:.4f} vs post-onset {post:.4f} (ratio {post/max(pre,1e-9):.2f}; "
          f"Franka reference is 13.5)", flush=True)
    lab = [{"kind": "torque_loss", "joint": int(joint[e]) + 2, "severity": 0.7, "onset": int(onset[e]),
            "ramp": 0, "task": args.task, "sweep": True} for e in range(N)]
    save("sweep", b, s, c, lab); save("sweepref", b, s_ref, c_ref, lab)

json.dump(manifest, open(f"{args.out}/manifest.json", "w"))
print(f"TOTAL episodes: {sum(m['n'] for m in manifest)}", flush=True)
env.close(); app.close()
print("GEN_ANYMAL_V2 DONE", flush=True)
