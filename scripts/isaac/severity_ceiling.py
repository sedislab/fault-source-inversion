"""What localization accuracy is physically achievable, as a function of fault severity? Matched pairs (identical
command+init) at several severities, scored by the simplest possible localizer (argmax of the counterfactual
deviation). This measures the ceiling the experiment design allows, independent of any model."""
import argparse, json, os
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--task", default="Isaac-Reach-Franka-v0")
p.add_argument("--num_envs", type=int, default=224)
p.add_argument("--steps", type=int, default=140)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app
import numpy as np, torch, gymnasium as gym
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
N, T, ARM = args.num_envs, args.steps, list(range(7))
cfg = parse_env_cfg(args.task, num_envs=N); env = gym.make(args.task, cfg=cfg)
u = env.unwrapped; r = u.scene["robot"]; dev = u.device; adim = env.action_space.shape[1]
idx = torch.arange(N, device=dev); ON = T // 3
PRISTINE = r.data.joint_effort_limits[:, ARM].clone()   # captured before any write

def rollout(seed, joint=None, frac=None):
    env.reset(seed=seed)
    r.write_joint_effort_limit_to_sim(PRISTINE.clone(), joint_ids=ARM)   # undo any previous rollout's fault
    g = torch.Generator(device=dev).manual_seed(seed + 7)
    ph = torch.rand((N, adim), generator=g, device=dev) * 6.283
    fr = 0.03 + 0.06 * torch.rand((N, adim), generator=g, device=dev)
    am = 0.4 + 0.4 * torch.rand((N, 1), generator=g, device=dev)
    base = PRISTINE.clone()
    sig = []
    for t in range(T):
        if joint is not None and t == ON:
            eff = base.clone(); eff[idx, joint] = torch.clamp(demand[joint] * frac, min=1e-4)
            r.write_joint_effort_limit_to_sim(eff, joint_ids=ARM)
        env.step(am * torch.sin(fr * t + ph))
        sig.append(torch.stack([r.data.joint_pos[:, ARM], r.data.joint_vel[:, ARM],
                                r.data.applied_torque[:, ARM]], -1).detach().cpu().numpy())
    return np.asarray(sig, np.float32)

cal = rollout(900)
demand = torch.tensor(np.percentile(np.abs(cal[..., 2]), 90, axis=(0, 1)), dtype=torch.float32, device=dev)
print("demand p90:", np.round(demand.cpu().numpy(), 2).tolist(), flush=True)
joint = torch.arange(N, device=dev) % 7
src = joint.cpu().numpy()
print(f"\n{'remaining torque':>18} | oracle argmax top1 | mean |delta| at source / max", flush=True)
for frac in (0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.37):
    S = 4242
    ref = rollout(S); flt = rollout(S, joint=joint, frac=frac)
    d = flt[ON:] - ref[ON:]                       # (t,N,7,3)
    mag = np.sqrt((d ** 2).sum(-1)).mean(0)       # (N,7)
    pred = mag.argmax(1)
    acc = float((pred == src).mean())
    at_src = float(mag[np.arange(N), src].mean()); mx = float(mag.max(1).mean())
    print(f"{frac*100:15.0f}%  |        {acc:.3f}       | {at_src:.3f} / {mx:.3f}", flush=True)
env.close(); app.close()
print("SEVERITY_CEILING DONE", flush=True)
