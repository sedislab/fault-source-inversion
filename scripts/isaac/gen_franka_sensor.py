"""Franka corpus with SENSOR faults alongside actuator faults, on the split actuation/sensing graph.

The point of this corpus is that a sensor fault here is a real fault, not a corrupted file. The servo reads the
same encoder the diagnosis system reads, so a biased encoder makes the controller drive the joint to the wrong
place and that error propagates through the arm exactly as an actuator fault does. Injecting it only into the
recorded array would give it an identity operator column -- a "fault" with no physics, trivially localizable and
dishonest as a benchmark.

Mechanism. Franka's actuators are ImplicitActuator: PhysX computes tau = kp(q_des - q_true) + kd(qd_des - qd_true)
internally, so there is no compute() to hook (unlike ANYmal's explicit ANYdrive). But a measurement error enters
that expression in exactly the same place as a target shift -- the servo believes q_meas = q_true + delta and
drives q_meas -> q_des, i.e. q_true -> q_des - delta. So commanding target' = target - delta is an EXACT
realization of the encoder fault, not an approximation. `joint_pos_target = 0.5 * action + default_q` was
measured on the running container (scripts/isaac/probes/probe_sensor.py: scale 0.5 on all 7 joints, sd 0 across envs),
and the probe confirmed a 0.15 rad bias at J4 moves J4 by -0.077 rad, raises |dtau| at every other joint
(J2 1.28, J3 0.41, J1 0.30 Nm), and gives a scale-free local effect ratio of 16.5x.

Node layout (fsi.graph.franka_sensor_graph), 14 nodes x 3 channels:
    0-6    J1..J7  ACTUATOR  (applied_torque, target - measured_pos, measured_vel * applied_torque)
    7-13   E1..E7  SENSOR    (measured_pos, measured_vel, measured_acc)

Everything the diagnosis system sees is a MEASUREMENT: the recorded encoder channels carry the corruption, and
so does the tracking error the actuator node reports. `u` is the NOMINAL target the trajectory generator asked
for, not the shifted one we write -- the shift is our way of putting the lie inside the servo loop, and a
deployed system would not know about it.
"""
import os, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])  # repository root (code)
ROOT = os.environ.get("FSI_ROOT", REPO)  # data/results root, see docs/CONFIGURATION.md
import argparse, json, os, sys
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=256)
p.add_argument("--tasks", default="Isaac-Reach-Franka-v0,Isaac-Lift-Cube-Franka-v0,Isaac-Open-Drawer-Franka-v0")
p.add_argument("--healthy_per_task", type=int, default=3)
p.add_argument("--fault_per_task", type=int, default=10)
p.add_argument("--sweep_per_task", type=int, default=2)
p.add_argument("--steps", type=int, default=140)
p.add_argument("--check_only", action="store_true", help="run the bind gate and stop, before spending hours")
p.add_argument("--out", default=f"{ROOT}/raw/franka_sensor")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app
print("APP UP", flush=True)

import numpy as np
import torch
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

NJ = 7
NNODE = 2 * NJ                       # J1..J7 then E1..E7, matching fsi.graph.franka_sensor_graph
N, T, ARM = args.num_envs, args.steps, list(range(NJ))
# `friction` is deliberately absent. On the SPLIT graph the actuator node carries (torque, tracking error,
# power) and the position channels live on the encoder node, and the gate measured joint friction at effect
# ratio 1.14x with the source moving 0.28 sigma and the loudest node NEVER being the source (1.000) -- even
# after rescaling the injection from a fixed +8.0 coefficient to 0.9*w*demand_tau, which is the formulation
# that binds on ANYmal. Its evidence lands entirely on the encoder node and its neighbours, so at joint-level
# resolution the label is not recoverable at its own node. Excluding a structurally non-isolable fault and
# saying so is standard practice (Tennessee Eastman faults 3, 9 and 15 are routinely dropped and papers report
# 18 classes instead of 22). It remains in the actuator-only corpora, where the joint node carries position.
ACT_KINDS = ["torque_loss", "jammed", "speed_limit"]
SEN_KINDS = ["enc_bias", "enc_drift", "enc_freeze", "enc_noise", "enc_scale", "enc_dropout"]
KINDS = ACT_KINDS + SEN_KINDS
KI = {k: i for i, k in enumerate(KINDS)}
# severity ranges: actuator kinds are a retained fraction of demand, sensor kinds are in units of the joint's own
# healthy position spread (sigma) except freeze/dropout/scale which are fractions.
SEV = {"torque_loss": (0.30, 0.90), "jammed": (0.30, 0.90), "speed_limit": (0.30, 0.90),
       "enc_bias": (0.6, 4.0), "enc_drift": (1.0, 6.0), "enc_freeze": (0.35, 1.0), "enc_noise": (1.5, 6.0),
       "enc_scale": (0.25, 0.80), "enc_dropout": (0.15, 0.65)}
ALPHA = 0.3                          # first-order low-pass on the differentiated encoder reading
# Secondary guard only: a fault can clear the RATIO test against a very quiet baseline while moving
# its own node by almost nothing in absolute terms. 0.3 sigma, not 0.5 -- speed_limit measured 0.47
# sigma at effect ratio 2.26x with the source above the median node on 0.906 of episodes, which is a
# real and attributable effect; the two ratio criteria are the ones inherited from the established
# gate and they carry the decision.
ABS_SIGMA_FLOOR = 0.3
os.makedirs(args.out, exist_ok=True)
manifest = []


def schedule(onset, ramp, T, dev):
    """(T,N) fault progress in [0,1]: 0 before onset, then step (ramp=0) or linear ramp."""
    t = torch.arange(T, device=dev)[:, None].float()
    prog = (t - onset[None, :].float()) / torch.clamp(ramp[None, :].float(), min=1.0)
    return torch.clamp(prog, 0.0, 1.0) * (t >= onset[None, :].float())


K_BIAS, K_DRIFT, K_FREEZE, K_NOISE, K_SCALE, K_DROP = (KINDS.index(k) for k in SEN_KINDS)


class Encoder:
    """Per-env encoder state: turns a fault spec into the measurement error delta = q_meas - q_true, one step at
    a time, and carries the state the history-dependent kinds need (the value latched at onset, the last good
    reading, the accumulated drift)."""

    def __init__(self, dev, sigma, q_ref):
        self.dev, self.sigma, self.q_ref = dev, sigma, q_ref     # sigma:(NJ,) healthy position spread
        self.latch = None            # q at the moment a freeze began
        self.last_good = None
        self.acc = torch.zeros(N, device=dev)                    # integrated drift, in units of sigma

    def delta(self, q_true, kind, joint, sev, w, gen):
        """(N,NJ) measurement error this step. `kind` (N,) indexes KINDS, `w` (N,) is the onset schedule in
        [0,1]; the realized severity is sev*w."""
        idx = torch.arange(N, device=self.dev)
        sig = self.sigma[joint]                                   # (N,) the source joint's own scale
        if self.latch is None:
            self.latch = q_true.clone()
            self.last_good = q_true.clone()
        started = w > 0
        self.latch = torch.where(started[:, None], self.latch, q_true)   # latch the value at onset, not before
        qs = q_true[idx, joint]                                   # (N,) the source joint's true position
        zero = torch.zeros(N, device=self.dev)
        # drift integrates the schedule over time, so it is a genuine ramp rather than a rescaled step. Driving
        # it off w alone would make an abrupt-onset drift (w jumps to 1) identical to a bias.
        self.acc = self.acc + w / T
        v = torch.where(kind == K_BIAS, sev * sig * w, zero)                    # constant offset
        v = torch.where(kind == K_DRIFT, sev * sig * self.acc, v)               # accumulating offset
        v = torch.where(kind == K_FREEZE, sev * w * (self.latch[idx, joint] - qs), v)   # blend toward latched
        eps = torch.randn(N, device=self.dev, generator=gen)
        v = torch.where(kind == K_NOISE, sev * sig * w * eps, v)                # measurement noise
        v = torch.where(kind == K_SCALE, sev * w * (qs - self.q_ref[joint]), v)  # gain error about the posture
        # dropout: with probability sev the reading is the last good one instead of the current one
        # scaled by the schedule, not merely gated by it: with `< sev` alone an incipient (ramped) dropout
        # switched on at full miss probability the instant it started, which is an abrupt fault wearing a ramp.
        drop = (torch.rand(N, device=self.dev, generator=gen) < sev * w) & started & (kind == K_DROP)
        v = torch.where(drop, self.last_good[idx, joint] - qs, v)
        keep = ~drop
        self.last_good[idx[keep], joint[keep]] = qs[keep]
        at = torch.zeros((N, NJ), device=self.dev)                # only the source joint's reading is corrupted
        at[idx, joint] = torch.where((kind >= len(ACT_KINDS)) & started, v, torch.zeros_like(v))
        return at


def run_task(task, seed0, rng, gate=True):
    cfg = parse_env_cfg(task, num_envs=N)
    env = gym.make(task, cfg=cfg)
    u = env.unwrapped
    r = u.scene["robot"]
    adim = env.action_space.shape[1]
    dev = u.device
    idx = torch.arange(N, device=dev)
    dt = float(u.step_dt) if hasattr(u, "step_dt") else float(u.physics_dt) * u.cfg.decimation
    PRISTINE_EFF = r.data.joint_effort_limits[:, ARM].clone()
    PRISTINE_DMP = r.data.joint_damping[:, ARM].clone() if hasattr(r.data, "joint_damping") else None
    PRISTINE_VEL = r.data.joint_vel_limits[:, ARM].clone() if hasattr(r.data, "joint_vel_limits") else None

    # --- the action -> target affine map, measured not assumed -------------------------------------------------
    env.reset(seed=seed0)
    a0 = torch.zeros((N, adim), device=dev)
    a1 = torch.zeros((N, adim), device=dev); a1[:, :NJ] = 0.1
    env.step(a0); env.step(a0); t0 = r.data.joint_pos_target[:, ARM].clone()
    env.step(a1); t1 = r.data.joint_pos_target[:, ARM].clone()
    ASCALE = ((t1 - t0) / 0.1).mean(0)                       # (NJ,)
    AOFF = t0.mean(0)
    print(f"{task} action->target scale {np.round(ASCALE.cpu().numpy(),4).tolist()} "
          f"offset {np.round(AOFF.cpu().numpy(),3).tolist()}", flush=True)
    assert float(ASCALE.abs().min()) > 1e-3, "degenerate action->target map: the encoder fault cannot be realized"

    def rollout(seed, faults=None, state=None):
        """faults = (kind, joint, sev, prog) with kind indexing KINDS and joint in 0..NJ-1 (the JOINT, not the
        node: an actuator kind attributes to node `joint`, a sensor kind to node `NJ + joint`)."""
        env.reset(seed=seed)
        if state is not None:
            root, jp, jv = state
            r.write_root_state_to_sim(root.clone()); r.write_joint_state_to_sim(jp.clone(), jv.clone())
        captured = (r.data.root_state_w.clone(), r.data.joint_pos.clone(), r.data.joint_vel.clone())
        g = torch.Generator(device=dev).manual_seed(seed + 7)
        base_fr = 0.2 * torch.rand((N, NJ), generator=g, device=dev)
        r.write_joint_friction_to_sim(base_fr, joint_ids=ARM)
        r.write_joint_effort_limit_to_sim(PRISTINE_EFF.clone(), joint_ids=ARM)
        if PRISTINE_DMP is not None:
            r.write_joint_damping_to_sim(PRISTINE_DMP.clone(), joint_ids=ARM)
        if PRISTINE_VEL is not None:
            r.write_joint_velocity_limit_to_sim(PRISTINE_VEL.clone(), joint_ids=ARM)
        base_eff, base_dmp = PRISTINE_EFF.clone(), (None if PRISTINE_DMP is None else PRISTINE_DMP.clone())
        base_vel = None if PRISTINE_VEL is None else PRISTINE_VEL.clone()
        phase = torch.rand((N, adim), generator=g, device=dev) * 6.283
        freq = 0.03 + 0.06 * torch.rand((N, adim), generator=g, device=dev)
        amp = 0.4 + 0.4 * torch.rand((N, 1), generator=g, device=dev)
        # Built only for a faulted rollout: SIGMA is measured BY the first (calibration) rollout, so it does not
        # exist yet on that call.
        enc = Encoder(dev, SIGMA, AOFF) if faults is not None else None
        WRITE_EVERY = 10                     # each physics write costs seconds of CPU API refresh
        first_onset, has_jam, has_spd, has_eff, has_fric = T, False, False, False, False
        if faults is not None:
            kind, joint, sev, prog_all = faults
            has_jam = bool((kind == KINDS.index("jammed")).any())
            has_spd = bool((kind == KINDS.index("speed_limit")).any())
            has_eff = bool(((kind == KINDS.index("torque_loss")) | (kind == KINDS.index("jammed"))).any())
            has_fric = bool((kind == KI.get("friction", -1)).any())
            first_onset = int((prog_all > 0).float().argmax(0).min().item())
        sig, cmd = [], []
        prev_dq = torch.zeros((N, NJ), device=dev)
        prev_qd_meas = r.data.joint_vel[:, ARM].clone()
        prev_qdd_meas = torch.zeros((N, NJ), device=dev)
        for t in range(T):
            q_true = r.data.joint_pos[:, ARM]
            # ---- the measurement the servo and the recorder both read --------------------------------------
            if faults is not None and t >= first_onset:
                w = sev * prog_all[t]
                dq = enc.delta(q_true, kind, joint, sev, prog_all[t], g)
            else:
                w = None
                dq = torch.zeros((N, NJ), device=dev)
            q_meas = q_true + dq
            # Differentiate the MEASUREMENT, then low-pass it, as any real joint controller does. A step in the
            # reading (a bias switching on, a dropout sample) differentiates to delta/dt, which at dt ~ 1/60 s is
            # a 60x spike that would otherwise swamp every other channel and make the fault trivially loud for
            # the wrong reason. ALPHA is applied identically on healthy and faulted rollouts, so nothing about
            # the filter separates the classes.
            qd_raw = r.data.joint_vel[:, ARM] + (dq - prev_dq) / dt
            qd_meas = ALPHA * qd_raw + (1 - ALPHA) * prev_qd_meas
            qdd_meas = ALPHA * (qd_meas - prev_qd_meas) / dt + (1 - ALPHA) * prev_qdd_meas
            prev_dq, prev_qd_meas, prev_qdd_meas = dq, qd_meas, qdd_meas
            # ---- actuator-side physics writes (unchanged mechanism from gen_franka_v2) -----------------------
            if faults is not None and t >= first_onset and t % WRITE_EVERY == 0:
                eff, fr = base_eff.clone(), base_fr.clone()
                tgt = torch.clamp(demand_tau[joint] * (1.0 - 0.9 * w), min=0.02)
                tgt = torch.minimum(tgt, base_eff[idx, joint])
                for kk in ("torque_loss", "jammed"):
                    eff[idx, joint] = torch.where(kind == KINDS.index(kk), tgt, eff[idx, joint])
                # Friction scaled to the joint's OWN measured demand, as gen_anymal_v2 does, not a fixed +8.0.
                # The fixed constant was calibrated against `franka_joint_graph`, whose actuator node carried
                # (pos, vel, torque) -- a friction fault showed up mostly in the POSITION channel. On the split
                # graph position lives on the encoder node, so the actuator node sees only torque, tracking
                # error and power, and the gate measured the source at 0.11 sigma with effect ratio 0.99x:
                # the fault's evidence had moved to a different candidate node and the label was no longer
                # recoverable at its own.
                fr[idx, joint] = torch.where(kind == KI.get("friction", -1),
                                             base_fr[idx, joint] + 0.9 * w * demand_tau[joint], fr[idx, joint])
                if has_eff:
                    r.write_joint_effort_limit_to_sim(eff, joint_ids=ARM)
                if has_fric:
                    r.write_joint_friction_to_sim(fr, joint_ids=ARM)
                if has_jam and base_dmp is not None:
                    d = base_dmp.clone()
                    d[idx, joint] = torch.where(kind == KINDS.index("jammed"), base_dmp[idx, joint] + 60.0 * w,
                                                d[idx, joint])
                    r.write_joint_damping_to_sim(d, joint_ids=ARM)
                if has_spd and base_vel is not None:
                    vl = base_vel.clone()
                    vt = torch.minimum(torch.clamp(demand_vel[joint] * (1.0 - 0.9 * w), min=0.02),
                                       base_vel[idx, joint])
                    vl[idx, joint] = torch.where(kind == KINDS.index("speed_limit"), vt, vl[idx, joint])
                    r.write_joint_velocity_limit_to_sim(vl, joint_ids=ARM)
            # ---- act: the NOMINAL command, then the encoder lie folded into the servo's target ---------------
            a_nom = amp * torch.sin(freq * t + phase)
            tgt_nom = ASCALE[None] * a_nom[:, :NJ] + AOFF[None]
            a = a_nom.clone()
            a[:, :NJ] = a_nom[:, :NJ] - dq / ASCALE[None]      # target' = target - delta  (exact for a PD servo)
            env.step(a)
            tau = r.data.applied_torque[:, ARM]
            node = torch.zeros((N, NNODE, 3), device=dev)
            node[:, :NJ] = torch.stack([tau, tgt_nom - q_meas, qd_meas * tau], -1)
            node[:, NJ:] = torch.stack([q_meas, qd_meas, qdd_meas], -1)
            sig.append(node.detach().cpu().numpy())
            cmd.append(tgt_nom.detach().cpu().numpy())
        return np.asarray(sig, np.float32), np.asarray(cmd, np.float32), captured

    # --- calibration: what the joints actually demand, and how much they actually move -------------------------
    s_cal, _, _ = rollout(seed0 + 900)
    demand_tau = torch.tensor(np.percentile(np.abs(s_cal[..., :NJ, 0]), 90, axis=(0, 1)), dtype=torch.float32,
                              device=dev)
    demand_vel = torch.tensor(np.percentile(np.abs(s_cal[..., NJ:, 1]), 90, axis=(0, 1)), dtype=torch.float32,
                              device=dev)
    SIGMA = torch.tensor(s_cal[..., NJ:, 0].std(axis=(0, 1)) + 1e-4, dtype=torch.float32, device=dev)
    print(f"{task} demand p90 torque {np.round(demand_tau.cpu().numpy(),2).tolist()}", flush=True)
    print(f"{task} position sigma   {np.round(SIGMA.cpu().numpy(),4).tolist()}", flush=True)

    # --- GATE: every kind must leave an attributable signature on its OWN node --------------------------------
    # The realized local effect ratio against a matched reference, scale-free (gen_anymal_v2's criterion: a
    # signed torque "drop" is the wrong test, since Coulomb friction makes torque RISE), measured over the
    # 14-node array so a sensor kind is checked at its encoder node rather than at the joint it measures.
    # MAX over channels, not mean: a sensor fault concentrates in the channel it corrupts, and averaging over
    # the node's channels divides its signature while the victims' broad-band response survives.
    # The gate does NOT require the source to be the loudest node -- `loudest!=source` is the cause-victim gap,
    # and a high value is the point of the project rather than a defect.
    on = T // 3
    _on = torch.full((N,), on, device=dev)
    zero_ramp = torch.zeros(N, device=dev)
    # The gate runs once, on the first task: it validates the injection MECHANISM, which every task shares, and
    # one reference plus ten faulted rollouts per task would be ~40% of the whole run.
    if not gate:
        print(f"\n(bind check skipped for {task}; it ran on the first task)\n", flush=True)
        bad, stats = [], {}
    else:
        print(f"\n=== BIND CHECK {task} (severity mid-range, joint 3) ===", flush=True)
        ref, _, st = rollout(seed0 + 777)
        sd_ref = ref[on:].std(axis=(0, 1)) + 1e-6                # (NNODE,3)
        bad, stats = [], {}
        for ki, kname in enumerate(KINDS):
            lo, hi = SEV[kname]
            kind = torch.full((N,), ki, dtype=torch.long, device=dev)
            joint = torch.full((N,), 3, dtype=torch.long, device=dev)
            sev = torch.full((N,), 0.5 * (lo + hi), device=dev)
            s_f, _, _ = rollout(seed0 + 777,
                                faults=(kind, joint, sev, schedule(_on, zero_ramp, T, dev)), state=st)
            z = np.abs(s_f[on:] - ref[on:]) / sd_ref                 # (T',N,NNODE,3)
            per_node = z.max(axis=3).mean(axis=0)                    # (N,NNODE) best-channel effect at each node
            src = 3 + (NJ if ki >= len(ACT_KINDS) else 0)
            ratio = per_node[:, src] / (np.median(per_node, axis=1) + 1e-9)
            med, frac = float(np.median(ratio)), float((ratio > 1.0).mean())
            absz = float(np.median(per_node[:, src]))
            hardfrac = float((per_node.argmax(1) != src).mean())
            stats[kname] = dict(ratio=med, above_median=frac, source_sigma=absz, cause_victim_gap=hardfrac)
            print(f"  {kname:12s} effect ratio {med:8.2f}x  above-median {frac:.3f}  source {absz:6.2f} sigma  "
                  f"loudest!=source {hardfrac:.3f}", flush=True)
            if med < 1.5 or frac < 0.9 or absz < ABS_SIGMA_FLOOR:
                bad.append(kname)
    if bad:
        print(f"\nFAIL {task}: these kinds leave no attributable signature: {bad}. Refusing to write a corpus "
              f"whose labels are unrecoverable -- that is exactly what produced ANYmal v1's top1 0.190 with "
              f"ground-truth labels.", flush=True)
        env.close(); app.close(); sys.exit(2)
    if gate:
        json.dump(stats, open(f"{args.out}/bind_check_{task}.json", "w"), indent=2)
        print(f"BIND CHECK PASSED for {task}\n", flush=True)
        if args.check_only:
            # Stop the process, not just this task: the gate is about the mechanism, and one task exercises it.
            env.close(); app.close(); sys.exit(0)

    def save(tag, b, s, c, labels=None):
        fn = f"{args.out}/{task}_{tag}_{b:03d}.npz"
        kw = {"labels": json.dumps(labels)} if labels else {}
        np.savez_compressed(fn, signals=s.transpose(1, 0, 2, 3), actions=c.transpose(1, 0, 2), **kw)
        manifest.append({"file": os.path.basename(fn), "kind": tag, "task": task, "n": N})
        print(f"{task} {tag} {b}", flush=True)

    for b in range(args.healthy_per_task):
        s, c, _ = rollout(seed0 + b); save("healthy", b, s, c)

    for b in range(args.fault_per_task):
        # Draw the NODE uniformly over all 14 candidates, then the kind from that node's family. A round robin
        # over KINDS instead puts 3 of 9 batches on the 7 actuator nodes and 6 of 9 on the 7 encoder nodes, so
        # the encoder family would carry two thirds of the corpus purely because it has more kinds -- a class
        # prior a method can exploit and a reader cannot see. Mixing kinds within a batch is safe: every
        # mechanism is masked per environment, and the batch-level flags below only decide WHETHER a physics
        # write happens at all.
        node_np = rng.integers(0, NNODE, N)
        is_sen_np = node_np >= NJ
        kname_np = np.array([(SEN_KINDS if s_ else ACT_KINDS)[rng.integers(len(SEN_KINDS if s_ else ACT_KINDS))]
                             for s_ in is_sen_np])
        kind = torch.tensor([KINDS.index(k) for k in kname_np], dtype=torch.long, device=dev)
        joint = torch.tensor(node_np % NJ, device=dev)
        lo = np.array([SEV[k][0] for k in kname_np]); hi = np.array([SEV[k][1] for k in kname_np])
        sev_np = (lo + (hi - lo) * rng.random(N)).astype(np.float32)
        sev = torch.tensor(sev_np, device=dev)
        onset = torch.tensor(rng.integers(int(0.25 * T), int(0.65 * T), N), device=dev)
        ramp = torch.tensor(np.where(rng.random(N) < 0.5, 0, rng.integers(10, 40, N)), device=dev)
        s, c, _ = rollout(seed0 + 100 + b, faults=(kind, joint, sev, schedule(onset, ramp, T, dev)))
        save("fault", b, s, c, [{"kind": str(kname_np[e]), "family": "encoder" if is_sen_np[e] else "actuator",
                                 "joint": int(node_np[e]), "severity": float(sev_np[e]),
                                 "onset": int(onset[e]), "ramp": int(ramp[e]), "task": task} for e in range(N)])

    for b in range(args.sweep_per_task):
        # Unit-fault sweep -> one operator column per candidate node. Node j < NJ gets the canonical actuator
        # fault, node j >= NJ the canonical encoder fault, both at a fixed severity, with a matched healthy twin
        # started from the same post-reset state so the difference is the fault and nothing else.
        node = torch.arange(N, device=dev) % NNODE
        joint = node % NJ
        is_sen = node >= NJ
        kind = torch.where(is_sen, torch.full_like(node, KINDS.index("enc_bias")),
                           torch.full_like(node, KINDS.index("torque_loss")))
        sev = torch.where(is_sen, torch.full((N,), 2.5, device=dev), torch.full((N,), 0.70, device=dev))
        onset = torch.full((N,), T // 3, device=dev)
        S = seed0 + 500 + b
        s_ref, c_ref, st = rollout(S)
        s, c, _ = rollout(S, faults=(kind, joint, sev, schedule(onset, torch.zeros(N, device=dev), T, dev)),
                          state=st)
        pre = np.abs(s[:on] - s_ref[:on]).mean(); post = np.abs(s[on:] - s_ref[on:]).mean()
        print(f"  sweep {b}: pre-onset divergence {pre:.4f} vs post {post:.4f} (ratio {post/max(pre,1e-9):.2f})",
              flush=True)
        lab = [{"kind": KINDS[int(kind[e])], "joint": int(node[e]), "severity": float(sev[e]),
                "onset": T // 3, "ramp": 0, "task": task, "sweep": True} for e in range(N)]
        save("sweep", b, s, c, lab); save("sweepref", b, s_ref, c_ref, lab)
    env.close()


rng = np.random.default_rng(0)
for ti, task in enumerate(args.tasks.split(",")):
    try:
        run_task(task, 1000 * (ti + 1), rng, gate=(ti == 0))
    except SystemExit:
        raise
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"TASK {task} FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)

json.dump(manifest, open(f"{args.out}/manifest.json", "w"))
print(f"TOTAL episodes: {sum(m['n'] for m in manifest if m['kind'] != 'sweepref')} (+{sum(m['n'] for m in manifest if m['kind'] == 'sweepref')} matched healthy references)",
      flush=True)
app.close()
print("GEN_FRANKA_SENSOR DONE", flush=True)
