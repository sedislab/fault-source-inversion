"""Why doesn't a torque fault bind on ANYmal-C, and what does?

Measured defect: at severity 0.70 the source joint's post-onset mean |tau| is 17.638 faulted vs 17.570 on the
matched reference -- torque drops in 49.7% of episodes, a coin flip. `write_joint_effort_limit_to_sim` and
`write_joint_friction_to_sim` are empirically no-ops here. The likely reason is that ANYmal-C uses an EXPLICIT
actuator model (an ANYdrive network), which computes torque in Python and clips it against the actuator object's
own limits before writing a target to PhysX -- so writing PhysX's effort limit is bypassed entirely.

This probe does not assume that. It enumerates the actuator objects, then tests candidate clamp mechanisms
head-to-head on identical seeds and reports, for each, the realized drop in the source joint's applied torque.
Whatever wins here is what the corpus generator must use."""
import argparse
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--steps", type=int, default=120)
p.add_argument("--onset", type=int, default=50)
p.add_argument("--joint", type=int, default=4)
p.add_argument("--severity", type=float, default=0.8)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app
print("APP UP", flush=True)

import numpy as np
import torch
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

N, T, ON, J, SEV = args.num_envs, args.steps, args.onset, args.joint, args.severity
cfg = parse_env_cfg("Isaac-Velocity-Flat-Anymal-C-v0", num_envs=N)
env = gym.make("Isaac-Velocity-Flat-Anymal-C-v0", cfg=cfg)
u = env.unwrapped
r = u.scene["robot"]
dev, adim = u.device, env.action_space.shape[1]
idx = torch.arange(N, device=dev)
NJ = 12

print("\n=== ACTUATOR INVENTORY ===", flush=True)
for key, act in r.actuators.items():
    print(f"  '{key}': {type(act).__module__}.{type(act).__name__}", flush=True)
    print(f"     is_implicit_model = {getattr(act, 'is_implicit_model', 'n/a')}", flush=True)
    for attr in ("effort_limit", "saturation_effort", "velocity_limit", "stiffness", "damping", "gear_ratio"):
        v = getattr(act, attr, None)
        if v is None:
            continue
        v = v if not torch.is_tensor(v) else v.flatten()[:4]
        print(f"     {attr}: {v}", flush=True)
    print(f"     joint_indices: {getattr(act, 'joint_indices', 'n/a')}", flush=True)
print(f"  joint_names: {list(r.joint_names)}", flush=True)
print(f"  PhysX effort limit [0,:4]: {r.data.joint_effort_limits[0, :4]}", flush=True)

PRISTINE_EFF = r.data.joint_effort_limits[:, :NJ].clone()
ACT = next(iter(r.actuators.values()))
ACT_EFF0 = ACT.effort_limit.clone() if torch.is_tensor(getattr(ACT, "effort_limit", None)) else None
ACT_SAT0 = ACT.saturation_effort.clone() if torch.is_tensor(getattr(ACT, "saturation_effort", None)) else None

def restore():
    r.write_joint_effort_limit_to_sim(PRISTINE_EFF.clone(), joint_ids=list(range(NJ)))
    if ACT_EFF0 is not None:
        ACT.effort_limit[:] = ACT_EFF0
    if ACT_SAT0 is not None:
        ACT.saturation_effort[:] = ACT_SAT0
    if hasattr(ACT, "_fsi_hook"):
        delattr(ACT, "_fsi_hook")

def rollout(mode, seed=1234):
    """One rollout under a named fault mechanism; returns per-step applied torque (T,N,NJ)."""
    restore()
    env.reset(seed=seed)
    g = torch.Generator(device=dev).manual_seed(seed + 7)
    phase = torch.rand((N, adim), generator=g, device=dev) * 6.283
    freq = 0.02 + 0.05 * torch.rand((N, adim), generator=g, device=dev)
    amp = 0.20 + 0.35 * torch.rand((N, 1), generator=g, device=dev)
    # Patch the actuator's compute() once, for the mechanisms that need it.
    if mode == "compute_hook" and not hasattr(ACT, "_fsi_orig"):
        ACT._fsi_orig = ACT.compute
        def hooked(control_action, joint_pos, joint_vel, _o=ACT._fsi_orig):
            out = _o(control_action, joint_pos, joint_vel)
            if getattr(ACT, "_fsi_on", False):
                out.joint_efforts[:, J] *= (1.0 - SEV)
            return out
        ACT.compute = hooked
    ACT._fsi_on = False
    taus = []
    for t in range(T):
        if t == ON:
            if mode == "physx_effort_limit":                       # what the generator does today
                eff = PRISTINE_EFF.clone(); eff[idx, J] *= (1.0 - SEV)
                r.write_joint_effort_limit_to_sim(eff, joint_ids=list(range(NJ)))
            elif mode == "actuator_effort_limit" and ACT_EFF0 is not None:
                ACT.effort_limit[:, J] = ACT_EFF0[:, J] * (1.0 - SEV)
            elif mode == "actuator_saturation" and ACT_SAT0 is not None:
                ACT.saturation_effort[:, J] = ACT_SAT0[:, J] * (1.0 - SEV)
            elif mode == "compute_hook":
                ACT._fsi_on = True
        env.step(amp * torch.sin(freq * t + phase))
        taus.append(r.data.applied_torque[:, :NJ].detach().cpu().numpy().copy())
    restore()
    return np.stack(taus)

print("\n=== MECHANISM COMPARISON ===", flush=True)
print(f"fault: joint {J} ({r.joint_names[J]}), severity {SEV}, onset {ON}, {N} envs", flush=True)
ref = rollout("none")
print(f"{'mechanism':26s} {'|tau_src| post':>14s} {'vs ref':>9s} {'drop%':>7s} {'frac eps dropped':>17s}", flush=True)
base = np.abs(ref[ON:, :, J]).mean()
print(f"{'none (reference)':26s} {base:14.3f} {'--':>9s} {'--':>7s} {'--':>17s}", flush=True)
for mode in ("physx_effort_limit", "actuator_effort_limit", "actuator_saturation", "compute_hook"):
    try:
        out = rollout(mode)
    except Exception as e:
        print(f"{mode:26s} FAILED {type(e).__name__}: {str(e)[:60]}", flush=True)
        continue
    val = np.abs(out[ON:, :, J]).mean()
    per_ep_f = np.abs(out[ON:, :, J]).mean(0)
    per_ep_r = np.abs(ref[ON:, :, J]).mean(0)
    frac = float((per_ep_f < per_ep_r).mean())
    print(f"{mode:26s} {val:14.3f} {val - base:+9.3f} {100*(1-val/max(base,1e-9)):6.1f}% {frac:17.3f}", flush=True)

print("\nPASS CRITERION: drop% > 20 and frac eps dropped > 0.9", flush=True)
env.close()
app.close()
