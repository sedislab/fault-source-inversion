"""Probe the two mechanisms a CLOSED-LOOP sensor fault needs, before spending GPU hours on a corpus.

A sensor fault is not a corruption of the recorded file. The controller reads the same sensor, so a biased
encoder makes the servo drive the joint to the wrong place, and that error propagates through the body exactly
as an actuator fault does. Injecting it only on the recorded array would produce a fault with no physics, whose
"propagation operator" column is the identity -- unlocalizable by construction and dishonest as a benchmark.

Two robots, two mechanisms:

  Franka   implicit actuator, PhysX computes tau = kp(q_des - q_true) + kd(qd_des - qd_true) internally, so
           there is nothing to hook. But a measurement bias b enters the SAME expression as a target shift:
           the servo believes q_meas = q_true + b and drives q_meas -> q_des, i.e. q_true -> q_des - b. So
           writing the action that yields target' = target - b is an EXACT realization. This probe measures the
           action -> joint_pos_target affine map, then checks the identity holds in the sim.

  ANYmal   explicit ActuatorNetLSTM, compute(control_action, joint_pos, joint_vel) is called in Python with the
           measured state as arguments -- so the corruption goes straight in as an argument. This probe checks
           the hook binds (same hook style gen_anymal_v2.py already uses for torque faults).

Run: scripts/isaac/run_in_container.sh scripts/isaac/probes/probe_sensor.py --headless [--robot franka|anymal|both]
"""
import argparse
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--robot", default="both", choices=["franka", "anymal", "both"])
p.add_argument("--num_envs", type=int, default=32)
p.add_argument("--steps", type=int, default=100)
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


def affine_map(env, u, r, joints):
    """Recover (scale, offset) in joint_pos_target = scale * action + offset, per joint, by stepping two known
    actions. Everything about the closed-loop encoder fault depends on this map being affine and stable."""
    adim = env.action_space.shape[1]
    dev = u.device
    a0 = torch.zeros((N, adim), device=dev)
    a1 = torch.zeros((N, adim), device=dev); a1[:, :len(joints)] = 0.1
    env.reset(seed=0)
    env.step(a0); t0 = r.data.joint_pos_target[:, joints].clone()
    env.step(a0); t0 = r.data.joint_pos_target[:, joints].clone()
    env.step(a1); t1 = r.data.joint_pos_target[:, joints].clone()
    scale = (t1 - t0) / 0.1
    offset = t0
    return scale.mean(0).cpu().numpy(), offset.mean(0).cpu().numpy(), scale.std(0).cpu().numpy()


def probe_franka():
    task = "Isaac-Reach-Franka-v0"
    cfg = parse_env_cfg(task, num_envs=N)
    env = gym.make(task, cfg=cfg)
    u = env.unwrapped
    r = u.scene["robot"]
    dev = u.device
    ARM = list(range(7))
    adim = env.action_space.shape[1]
    print(f"\n=== FRANKA {task} | adim {adim} | joints {list(r.joint_names)[:7]} ===", flush=True)
    print("actuators:", {k: type(v).__name__ for k, v in r.actuators.items()}, flush=True)
    for k, v in r.actuators.items():
        print(f"  {k}: implicit={getattr(v, 'is_implicit_model', 'n/a')} joints={list(getattr(v, 'joint_names', []))}",
              flush=True)
    try:
        print("action terms:", list(u.action_manager._terms.keys()), flush=True)
        for nm, term in u.action_manager._terms.items():
            print(f"  {nm}: {type(term).__name__} dim={term.action_dim} "
                  f"scale={getattr(term, '_scale', None) if not torch.is_tensor(getattr(term, '_scale', None)) else term._scale.flatten()[:8].tolist()} "
                  f"offset={'tensor' if torch.is_tensor(getattr(term, '_offset', None)) else getattr(term, '_offset', None)}",
                  flush=True)
    except Exception as e:
        print("action manager introspection failed:", type(e).__name__, str(e)[:200], flush=True)

    sc, off, sd = affine_map(env, u, r, ARM)
    print("ACTION->TARGET  scale", np.round(sc, 4).tolist(), flush=True)
    print("                offset", np.round(off, 4).tolist(), flush=True)
    print("                scale sd across envs", np.round(sd, 6).tolist(), flush=True)

    # --- the identity test: does target' = target - b move the joint by exactly -b? ---------------------------
    JT, BIAS = 3, 0.15

    def run(bias):
        env.reset(seed=1)
        g = torch.Generator(device=dev).manual_seed(7)
        phase = torch.rand((N, adim), generator=g, device=dev) * 6.283
        freq = 0.03 + 0.06 * torch.rand((N, adim), generator=g, device=dev)
        amp = 0.4 + 0.4 * torch.rand((N, 1), generator=g, device=dev)
        qs, taus = [], []
        for t in range(T):
            a = amp * torch.sin(freq * t + phase)
            if bias and t >= T // 3:
                a = a.clone()
                a[:, JT] -= bias / max(float(sc[JT]), 1e-6)      # target' = target - b
            env.step(a)
            qs.append(r.data.joint_pos[:, ARM].detach().cpu().numpy().copy())
            taus.append(r.data.applied_torque[:, ARM].detach().cpu().numpy().copy())
        return np.asarray(qs), np.asarray(taus)

    q0, tau0 = run(0.0)
    q1, tau1 = run(BIAS)
    on = T // 3
    dq = (q1 - q0)[on:].mean(axis=(0, 1))
    print(f"\nCLOSED-LOOP ENCODER BIAS b={BIAS} rad at joint J{JT+1}", flush=True)
    print("  mean dq per joint (expect ~-b at the source, non-zero elsewhere = propagation):",
          np.round(dq, 4).tolist(), flush=True)
    print(f"  source dq {dq[JT]:+.4f} vs -b {-BIAS:+.4f}  -> realization error {abs(dq[JT]+BIAS):.4f}", flush=True)
    dtau = np.abs(tau1 - tau0)[on:].mean(axis=(0, 1))
    print("  mean |dtau| per joint (the actuator is the VICTIM of its own encoder):",
          np.round(dtau, 4).tolist(), flush=True)
    sd_ref = np.abs(q0[on:]).std(axis=(0, 1)) + 1e-9
    z = np.abs(q1 - q0)[on:].mean(axis=(0, 1)) / sd_ref
    print("  scale-free effect ratio (source / median joint):", round(float(z[JT] / (np.median(z) + 1e-9)), 2),
          flush=True)
    loud = int(np.argmax(dtau))
    print(f"  LOUDEST TORQUE NODE = J{loud+1}, source = J{JT+1} -> "
          f"{'HARD (cause != loudest)' if loud != JT else 'easy (cause == loudest)'}", flush=True)

    # --- freeze: measurement latched, so the servo sees a constant error and the joint runs away -------------
    def run_freeze(sev):
        env.reset(seed=1)
        g = torch.Generator(device=dev).manual_seed(7)
        phase = torch.rand((N, adim), generator=g, device=dev) * 6.283
        freq = 0.03 + 0.06 * torch.rand((N, adim), generator=g, device=dev)
        amp = 0.4 + 0.4 * torch.rand((N, 1), generator=g, device=dev)
        qs, held = [], None
        for t in range(T):
            a = amp * torch.sin(freq * t + phase)
            if t >= T // 3:
                qtrue = r.data.joint_pos[:, JT]
                if held is None:
                    held = qtrue.clone()
                qmeas = (1 - sev) * qtrue + sev * held           # severity blends true and latched
                a = a.clone()
                a[:, JT] -= (qmeas - qtrue) / max(float(sc[JT]), 1e-6)
            env.step(a)
            qs.append(r.data.joint_pos[:, ARM].detach().cpu().numpy().copy())
        return np.asarray(qs)

    for sev in (0.5, 1.0):
        qf = run_freeze(sev)
        d = np.abs(qf - q0)[on:].mean(axis=(0, 1))
        print(f"  FREEZE sev={sev}: mean |dq| per joint {np.round(d, 4).tolist()}", flush=True)
    env.close()


def probe_anymal():
    task = "Isaac-Velocity-Flat-Anymal-C-v0"
    cfg = parse_env_cfg(task, num_envs=N)
    env = gym.make(task, cfg=cfg)
    u = env.unwrapped
    r = u.scene["robot"]
    dev = u.device
    NJ = 12
    adim = env.action_space.shape[1]
    print(f"\n=== ANYMAL {task} | adim {adim} ===", flush=True)
    ACT = next(iter(r.actuators.values()))
    print(f"actuator {type(ACT).__name__} implicit={getattr(ACT, 'is_implicit_model', 'n/a')}", flush=True)
    print("sensors:", list(u.scene.sensors.keys()), flush=True)
    try:
        cs = u.scene.sensors["contact_forces"]
        print("contact bodies:", list(getattr(cs, "body_names", []))[:20], flush=True)
        print("net_forces_w shape:", tuple(cs.data.net_forces_w.shape), flush=True)
    except Exception as e:
        print("contact sensor n/a:", type(e).__name__, str(e)[:150], flush=True)
    for attr in ("root_lin_vel_b", "root_ang_vel_b", "projected_gravity_b", "root_pos_w"):
        v = getattr(r.data, attr, None)
        print(f"  data.{attr}: {None if v is None else tuple(v.shape)}", flush=True)

    # the encoder hook: corrupt the measured joint state the actuator model consumes
    ACT._fsi_qbias = None
    _orig = ACT.compute

    def hooked(control_action, joint_pos, joint_vel, _o=_orig):
        if ACT._fsi_qbias is not None:
            joint_pos = joint_pos.clone()
            joint_pos[:, :NJ] = joint_pos[:, :NJ] + ACT._fsi_qbias
        return _o(control_action, joint_pos, joint_vel)

    ACT.compute = hooked

    def run(bias_joint=None, b=0.0):
        ACT._fsi_qbias = None
        env.reset(seed=2)
        g = torch.Generator(device=dev).manual_seed(7)
        phase = torch.rand((N, adim), generator=g, device=dev) * 6.283
        freq = 0.02 + 0.05 * torch.rand((N, adim), generator=g, device=dev)
        amp = 0.20 + 0.35 * torch.rand((N, 1), generator=g, device=dev)
        qs, taus = [], []
        for t in range(T):
            if bias_joint is not None and t >= T // 3:
                qb = torch.zeros((N, NJ), device=dev); qb[:, bias_joint] = b
                ACT._fsi_qbias = qb
            env.step(amp * torch.sin(freq * t + phase))
            qs.append(r.data.joint_pos[:, :NJ].detach().cpu().numpy().copy())
            taus.append(r.data.applied_torque[:, :NJ].detach().cpu().numpy().copy())
        ACT._fsi_qbias = None
        return np.asarray(qs), np.asarray(taus)

    q0, tau0 = run()
    JT, B = 4, 0.20
    q1, tau1 = run(JT, B)
    on = T // 3
    dq = (q1 - q0)[on:].mean(axis=(0, 1))
    dtau = np.abs(tau1 - tau0)[on:].mean(axis=(0, 1))
    print(f"\nANYMAL ENCODER BIAS b={B} rad at joint {JT} (via compute() hook)", flush=True)
    print("  mean dq per joint:", np.round(dq, 4).tolist(), flush=True)
    print(f"  source dq {dq[JT]:+.4f} vs -b {-B:+.4f}", flush=True)
    print("  mean |dtau| per joint:", np.round(dtau, 3).tolist(), flush=True)
    sd = np.abs(q0[on:]).std(axis=(0, 1)) + 1e-9
    z = np.abs(q1 - q0)[on:].mean(axis=(0, 1)) / sd
    print("  effect ratio source/median:", round(float(z[JT] / (np.median(z) + 1e-9)), 2), flush=True)
    print(f"  BINDS: {'YES' if abs(dq[JT]) > 0.02 else 'NO'}", flush=True)
    env.close()


if args.robot in ("franka", "both"):
    try:
        probe_franka()
    except Exception as e:
        import traceback; traceback.print_exc()
        print("FRANKA PROBE FAILED:", type(e).__name__, str(e)[:300], flush=True)
if args.robot in ("anymal", "both"):
    try:
        probe_anymal()
    except Exception as e:
        import traceback; traceback.print_exc()
        print("ANYMAL PROBE FAILED:", type(e).__name__, str(e)[:300], flush=True)

app.close()
print("PROBE_SENSOR DONE", flush=True)
