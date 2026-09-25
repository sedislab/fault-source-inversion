"""ANYmal-C corpus with SENSOR faults, on the 18-node graph whose sensor nodes finally become injectable.

`anymal_v4` injects at 12 of 18 nodes: the imu and the four feet are in the graph, are typed SENSOR, and are
never a fault source -- so `eval_embodiment.py` has to shrink the candidate set to the 12 leg joints, and the
published chance level is 1/12 rather than the 1/17 the figure implies. This corpus injects at all 17 failable
nodes and carries the two families of sensor fault the FDI literature separates, because they pose opposite
problems:

  IN-LOOP (encoder).  ANYmal-C's ANYdrive is an EXPLICIT ActuatorNetLSTM whose `compute(control_action,
  joint_pos, joint_vel)` runs in Python with the measured joint state as arguments -- so a corrupted encoder
  goes straight in as an argument and the actuator net computes a torque from a lie. The error then propagates
  through the body like any actuator fault. Because the ANYdrive packages motor and encoder in ONE physical
  unit, the encoder fault is labelled at the joint's own node; it is a different KIND at the same node, not a
  different node, which is what the hardware actually looks like.

  OBSERVATION-LAYER (imu, foot contact).  These corrupt what is reported without changing the physics -- the
  rollout is driven by a scripted command, so nothing reads the observation back. This is the classical sensor
  fault, and it is the complementary hard case: the source IS the loudest node, so every method that assumes a
  fault must propagate (direct l1 against a propagating G, the loudest-victim story) is wrong in a new way.
  These episodes are therefore mostly EASY by the `hard_top1` definition and are reported per family; the
  headline hard-episode number still comes from the propagating kinds.

Node layout (fsi.graph.anymal_sensor_graph), 18 nodes x 4 channels:
    0      base   BODY      (lin_vel_x, lin_vel_y, lin_vel_z, height)
    1      imu    SENSOR    (proj_grav_x, proj_grav_y, ang_vel_x, ang_vel_y)
    2-13   joints ACTUATOR  (joint_pos_meas, joint_vel_meas, applied_torque, target - joint_pos_meas)
    14-17  feet   SENSOR    (|F|, contact, stance_time, swing_time)
"""
import os, sys
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])  # repository root (code)
ROOT = os.environ.get("FSI_ROOT", REPO)  # data/results root, see docs/CONFIGURATION.md
import argparse, json, os, sys
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", default="Isaac-Velocity-Flat-Anymal-C-v0")
p.add_argument("--num_envs", type=int, default=256)
p.add_argument("--healthy_batches", type=int, default=5)
p.add_argument("--fault_batches", type=int, default=22)
p.add_argument("--sweep_batches", type=int, default=3)
p.add_argument("--steps", type=int, default=140)
p.add_argument("--check_only", action="store_true")
p.add_argument("--out", default=f"{ROOT}/raw/anymal_sensor")
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
LEGS = ("LF", "LH", "RF", "RH")
NJ, NNODE, NCH = 12, 18, 4
J0, F0 = 2, 14                        # first joint node, first foot node
ACT_KINDS = ["torque_loss", "friction", "jammed", "speed_limit"]
# `enc_freeze` is deliberately absent on this robot. Raising its severity floor from 0.35 to 0.75 moved the
# gate only from 1.33x/0.750 to 1.43x/0.797, so the shortfall is structural rather than sub-MDF: a frozen
# ANYdrive encoder propagates so strongly (loudest node is not the source on 0.836 of episodes, the highest in
# the catalogue) that the source itself never stands out against the victims it creates. It stays in the Franka
# catalogue, where the same kind gates at 2.34x with the source at 0.78 sigma.
ENC_KINDS = ["enc_bias", "enc_noise"]                                    # in-loop, at the joint node
IMU_KINDS = ["imu_bias", "imu_drift", "imu_noise", "imu_freeze"]          # observation layer, node 1
FOOT_KINDS = ["contact_dropout", "contact_stuck", "contact_bias"]         # observation layer, nodes 14-17
KINDS = ACT_KINDS + ENC_KINDS + IMU_KINDS + FOOT_KINDS
SEV = {"torque_loss": (0.30, 0.90), "friction": (0.30, 0.90), "jammed": (0.30, 0.90), "speed_limit": (0.30, 0.90),
       "enc_bias": (0.8, 4.0), "enc_noise": (1.5, 6.0),
       "imu_bias": (1.0, 5.0), "imu_drift": (1.5, 7.0), "imu_noise": (2.0, 8.0), "imu_freeze": (0.4, 1.0),
       "contact_dropout": (0.2, 0.8), "contact_stuck": (0.3, 1.0), "contact_bias": (1.0, 5.0)}
FAMILY = ({k: "actuator" for k in ACT_KINDS} | {k: "encoder" for k in ENC_KINDS} |
          {k: "imu" for k in IMU_KINDS} | {k: "foot" for k in FOOT_KINDS})
ALPHA = 0.3
# Secondary guard only: a fault can clear the RATIO test against a very quiet baseline while moving
# its own node by almost nothing in absolute terms. 0.3 sigma, not 0.5 -- speed_limit measured 0.47
# sigma at effect ratio 2.26x with the source above the median node on 0.906 of episodes, which is a
# real and attributable effect; the two ratio criteria are the ones inherited from the established
# gate and they carry the decision.
ABS_SIGMA_FLOOR = 0.3
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
dt = float(u.step_dt) if hasattr(u, "step_dt") else float(u.physics_dt) * u.cfg.decimation
PRISTINE_DMP = r.data.joint_damping[:, JOINTS].clone() if hasattr(r.data, "joint_damping") else None
print("joints", list(r.joint_names), flush=True)

ACT = next(iter(r.actuators.values()))
print(f"actuator: {type(ACT).__name__}  implicit={getattr(ACT, 'is_implicit_model', 'n/a')}", flush=True)
ACT._fsi_scale = None      # (N,NJ) multiplicative torque retention
ACT._fsi_coulomb = None    # (N,NJ) opposing torque magnitude
ACT._fsi_vlim = None       # (N,NJ) speed past which torque derates
ACT._fsi_qerr = None       # (N,NJ) ENCODER error fed to the actuator model -- the in-loop sensor fault
_orig_compute = ACT.compute


def _hooked_compute(control_action, joint_pos, joint_vel, _o=_orig_compute):
    if ACT._fsi_qerr is not None:
        joint_pos = joint_pos.clone()
        joint_pos[:, :NJ] = joint_pos[:, :NJ] + ACT._fsi_qerr      # the model computes torque from the lie
    out = _o(control_action, joint_pos, joint_vel)
    if ACT._fsi_scale is not None:
        out.joint_efforts[:, :NJ] = out.joint_efforts[:, :NJ] * ACT._fsi_scale
    if ACT._fsi_coulomb is not None:
        out.joint_efforts[:, :NJ] -= ACT._fsi_coulomb * torch.tanh(10.0 * joint_vel[:, :NJ])
    if ACT._fsi_vlim is not None:
        over = torch.clamp(joint_vel[:, :NJ].abs() - ACT._fsi_vlim, min=0.0)
        out.joint_efforts[:, :NJ] = out.joint_efforts[:, :NJ] * torch.clamp(1.0 - over / (ACT._fsi_vlim + 1e-3),
                                                                           0.0, 1.0)
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
    cand = [next((i for i, n in enumerate(names) if n.upper().startswith(leg) and "FOOT" in n.upper()), None)
            for leg in LEGS]
    foot_cols = cand if all(c is not None for c in cand) else None
assert foot_cols is not None, "no per-foot contact columns: the foot sensor nodes would carry nothing to fault"
CONTACT_THR = 1.0                                    # N; above this the foot is reported in contact


def schedule(onset, ramp, dev):
    t = torch.arange(T, device=dev)[:, None].float()
    prog = (t - onset[None, :].float()) / torch.clamp(ramp[None, :].float(), min=1.0)
    return torch.clamp(prog, 0.0, 1.0) * (t >= onset[None, :].float())


class Corruption:
    """State for the history-dependent sensor kinds (latched values, accumulated drift, last good reading)."""

    def __init__(self):
        self.latch_q = self.latch_imu = self.last_foot = None
        self.acc = torch.zeros(N, device=dev)

    def encoder(self, q_true, kind, joint, sev, w, gen):
        """(N,NJ) error added to the measured joint position, both in the loop and in the record."""
        if self.latch_q is None:
            self.latch_q = q_true.clone()
        started = w > 0
        self.latch_q = torch.where(started[:, None], self.latch_q, q_true)
        qs = q_true[idx, joint]
        sig = QSIG[joint]
        zero = torch.zeros(N, device=dev)
        v = torch.where(kind == KINDS.index("enc_bias"), sev * sig * w, zero)
        v = torch.where(kind == KINDS.index("enc_noise"),
                        sev * sig * w * torch.randn(N, device=dev, generator=gen), v)
        out = torch.zeros((N, NJ), device=dev)
        is_enc = (kind >= len(ACT_KINDS)) & (kind < len(ACT_KINDS) + len(ENC_KINDS))
        out[idx, joint] = torch.where(is_enc & started, v, torch.zeros_like(v))
        return out

    def imu(self, meas, kind, sev, w, gen):
        """(N,4) corrupted IMU reading. Observation layer: the scripted command never reads it back."""
        if self.latch_imu is None:
            self.latch_imu = meas.clone()
        started = w > 0
        self.latch_imu = torch.where(started[:, None], self.latch_imu, meas)
        self.acc = self.acc + w / T
        s = (sev * w)[:, None] * IMUSIG[None]
        out = meas.clone()
        hit = (kind == KINDS.index("imu_bias")) & started
        out = torch.where(hit[:, None], meas + s, out)
        hit = (kind == KINDS.index("imu_drift")) & started
        out = torch.where(hit[:, None], meas + (sev * self.acc)[:, None] * IMUSIG[None], out)
        hit = (kind == KINDS.index("imu_noise")) & started
        out = torch.where(hit[:, None], meas + s * torch.randn((N, NCH), device=dev, generator=gen), out)
        hit = (kind == KINDS.index("imu_freeze")) & started
        out = torch.where(hit[:, None], meas + (sev * w)[:, None] * (self.latch_imu - meas), out)
        return out

    def foot(self, meas, kind, foot_id, sev, w, gen):
        """(N,4,2) corrupted per-foot (|F|, contact); only the source foot is touched. The flag is re-derived
        from the corrupted force, so a force the sensor under-reports also reads as no contact."""
        if self.last_foot is None:
            self.last_foot = meas.clone()
        started = w > 0
        is_foot = kind >= len(ACT_KINDS) + len(ENC_KINDS) + len(IMU_KINDS)
        cur = meas[idx, foot_id]                                                        # (N,2)
        held = self.last_foot[idx, foot_id]
        drop = (torch.rand(N, device=dev, generator=gen) < sev) & started & is_foot
        v = cur.clone()
        v = torch.where(((kind == KINDS.index("contact_dropout")) & drop)[:, None],
                        torch.zeros_like(cur), v)                                      # missed contact
        v = torch.where(((kind == KINDS.index("contact_stuck")) & started)[:, None],
                        cur + (sev * w)[:, None] * (held - cur), v)                     # latched reading
        v = torch.where(((kind == KINDS.index("contact_bias")) & started)[:, None],
                        cur + (sev * w)[:, None] * FSIG[None], v)                       # force bias
        keep = ~((kind == KINDS.index("contact_stuck")) & started)
        self.last_foot[idx[keep], foot_id[keep]] = meas[idx[keep], foot_id[keep]]
        out = meas.clone()
        out[idx, foot_id] = torch.where((is_foot & started)[:, None], v, cur)
        out[..., 1] = (out[..., 0] > CONTACT_THR).float()
        return out


def raw_foot():
    """(N,4,2): contact-force magnitude and the binary contact flag. The timing channels are appended by the
    caller from the (possibly corrupted) flag, because that is what a contact estimator actually does."""
    f = cs.data.net_forces_w[:, foot_cols, :]                       # (N,4,3)
    mag = torch.linalg.norm(f, dim=-1)
    return torch.stack([mag, (mag > CONTACT_THR).float()], -1)


class FootTimers:
    """Stance and swing duration derived from the REPORTED contact flag, in seconds.

    These replace F_z and |F_xy|, which were both degenerate: the gate measured the foot channels' healthy
    spread as [73.35, 0.254, 73.35, 0.000] -- on flat ground the contact force is vertical, so F_z is a copy of
    |F| and the tangential component is identically zero. Two of the four foot channels carried nothing, and a
    channel that never varies cannot express a fault.

    Timing is the right replacement because it is exactly what a contact fault destroys: a dropout chops one
    stance into two short ones and injects a phantom swing, a stuck-high sensor reports a stance that never
    ends. Deriving the timers from the corrupted flag rather than from the true force is the honest choice --
    the estimator downstream of a lying sensor is lied to as well."""

    def __init__(self):
        self.contact_t = torch.zeros((N, 4), device=dev)
        self.air_t = torch.zeros((N, 4), device=dev)

    def step(self, contact):
        self.contact_t = torch.where(contact > 0.5, self.contact_t + dt, torch.zeros_like(self.contact_t))
        self.air_t = torch.where(contact > 0.5, torch.zeros_like(self.air_t), self.air_t + dt)
        return torch.stack([self.contact_t, self.air_t], -1)


def rollout(seed, faults=None, state=None):
    ACT._fsi_scale = ACT._fsi_coulomb = ACT._fsi_vlim = ACT._fsi_qerr = None
    env.reset(seed=seed)
    if state is not None:
        root, jp, jv = state
        r.write_root_state_to_sim(root.clone()); r.write_joint_state_to_sim(jp.clone(), jv.clone())
    captured = (r.data.root_state_w.clone(), r.data.joint_pos.clone(), r.data.joint_vel.clone())
    g = torch.Generator(device=dev).manual_seed(seed + 7)
    base_fr = 0.15 * torch.rand((N, NJ), generator=g, device=dev)
    r.write_joint_friction_to_sim(base_fr, joint_ids=JOINTS)
    if PRISTINE_DMP is not None:
        r.write_joint_damping_to_sim(PRISTINE_DMP.clone(), joint_ids=JOINTS)
    base_dmp = None if PRISTINE_DMP is None else PRISTINE_DMP.clone()
    phase = torch.rand((N, adim), generator=g, device=dev) * 6.283
    freq = 0.02 + 0.05 * torch.rand((N, adim), generator=g, device=dev)
    amp = 0.20 + 0.35 * torch.rand((N, 1), generator=g, device=dev)
    cor = Corruption() if faults is not None else None
    has_jam = False
    if faults is not None:
        kind, node, sev, prog = faults
        joint = torch.clamp(node - J0, 0, NJ - 1)                  # joint index when the source is a leg node
        foot_id = torch.clamp(node - F0, 0, 3)                     # foot index when the source is a foot node
        has_jam = bool((kind == KINDS.index("jammed")).any())
    timers = FootTimers()
    sig, cmd = [], []
    prev_dq = torch.zeros((N, NJ), device=dev)
    prev_qd = r.data.joint_vel[:, JOINTS].clone()
    for t in range(T):
        dq = torch.zeros((N, NJ), device=dev)
        if faults is not None:
            w = sev * prog[t]
            scale = torch.ones((N, NJ), device=dev)
            coul = torch.zeros((N, NJ), device=dev)
            tl = (kind == KINDS.index("torque_loss")) | (kind == KINDS.index("jammed"))
            scale[idx, joint] = torch.where(tl, 1.0 - 0.9 * w, scale[idx, joint])
            coul[idx, joint] = torch.where(kind == KINDS.index("friction"), 0.9 * w * demand_tau[joint],
                                           coul[idx, joint])
            vlim = torch.full((N, NJ), 1e6, device=dev)
            vlim[idx, joint] = torch.where(kind == KINDS.index("speed_limit"),
                                           torch.clamp(demand_vel[joint] * (1.0 - 0.9 * w), min=0.02),
                                           vlim[idx, joint])
            dq = cor.encoder(r.data.joint_pos[:, JOINTS], kind, joint, sev, prog[t], g)
            ACT._fsi_scale, ACT._fsi_coulomb, ACT._fsi_vlim, ACT._fsi_qerr = scale, coul, vlim, dq
            if t % 10 == 0 and has_jam and base_dmp is not None:
                d = base_dmp.clone()
                d[idx, joint] = torch.where(kind == KINDS.index("jammed"), base_dmp[idx, joint] + 60.0 * w,
                                            d[idx, joint])
                r.write_joint_damping_to_sim(d, joint_ids=JOINTS)
        env.step(amp * torch.sin(freq * t + phase))
        q_meas = r.data.joint_pos[:, JOINTS] + dq
        qd_raw = r.data.joint_vel[:, JOINTS] + (dq - prev_dq) / dt
        qd_meas = ALPHA * qd_raw + (1 - ALPHA) * prev_qd
        prev_dq, prev_qd = dq, qd_meas
        pg, av, lv = r.data.projected_gravity_b, r.data.root_ang_vel_b, r.data.root_lin_vel_b
        imu_meas = torch.stack([pg[:, 0], pg[:, 1], av[:, 0], av[:, 1]], -1)
        foot_meas = raw_foot()
        if faults is not None:
            imu_meas = cor.imu(imu_meas, kind, sev, prog[t], g)
            foot_meas = cor.foot(foot_meas, kind, foot_id, sev, prog[t], g)
        foot_meas = torch.cat([foot_meas, timers.step(foot_meas[..., 1])], -1)   # (N,4,4)
        tgt = r.data.joint_pos_target[:, JOINTS]
        s = torch.zeros((N, NNODE, NCH), device=dev)
        s[:, 0] = torch.stack([lv[:, 0], lv[:, 1], lv[:, 2], r.data.root_pos_w[:, 2]], -1)
        s[:, 1] = imu_meas
        s[:, J0:J0 + NJ] = torch.stack([q_meas, qd_meas, r.data.applied_torque[:, JOINTS], tgt - q_meas], -1)
        s[:, F0:] = foot_meas
        sig.append(s.detach().cpu().numpy())
        cmd.append(tgt.detach().cpu().numpy())
    ACT._fsi_scale = ACT._fsi_coulomb = ACT._fsi_vlim = ACT._fsi_qerr = None
    return np.asarray(sig, np.float32), np.asarray(cmd, np.float32), captured


_scal, _, _ = rollout(900)
demand_tau = torch.tensor(np.percentile(np.abs(_scal[:, :, J0:J0 + NJ, 2]), 90, axis=(0, 1)),
                          dtype=torch.float32, device=dev)
demand_vel = torch.tensor(np.percentile(np.abs(_scal[:, :, J0:J0 + NJ, 1]), 50, axis=(0, 1)),
                          dtype=torch.float32, device=dev)
QSIG = torch.tensor(_scal[:, :, J0:J0 + NJ, 0].std(axis=(0, 1)) + 1e-4, dtype=torch.float32, device=dev)
IMUSIG = torch.tensor(_scal[:, :, 1, :].std(axis=(0, 1)) + 1e-4, dtype=torch.float32, device=dev)
FSIG = torch.tensor(_scal[:, :, F0:, :2].std(axis=(0, 1)).mean(0) + 1e-4, dtype=torch.float32, device=dev)
print("demand p90 torque", np.round(demand_tau.cpu().numpy(), 2).tolist(), flush=True)
print("q sigma", np.round(QSIG.cpu().numpy(), 4).tolist(), "| imu sigma", np.round(IMUSIG.cpu().numpy(), 4).tolist(),
      "| foot sigma", np.round(FSIG.cpu().numpy(), 3).tolist(), flush=True)

# ---- GATE: every kind must leave an attributable signature on its OWN node -----------------------------------
# The criterion is the realized local effect ratio against a matched reference, scale-free, per the lesson from
# ANYmal v1 (where every kind's ratio was ~1.00 -- the source deviated no more than a random joint -- and a fully
# supervised model with ground-truth labels topped out at top1 0.190). Two things about HOW it is measured:
#
#   MAX over channels, not mean. A sensor fault concentrates in the one channel it corrupts, so averaging over
#   the node's 4 channels divides its signature by 4 while the victims' broad-band response survives. Measured
#   with the mean, `enc_bias` scored 1.20x and `enc_freeze` 1.26x and both were rejected -- yet the standalone
#   probe shows the ANYmal encoder bias realizes 93% of the commanded offset at the source joint. The mean was
#   mis-measuring a fault that binds perfectly well.
#
#   The gate does NOT require the source to be the loudest node. `loudest!=source` is printed because it is the
#   cause-victim gap, and a HIGH value is the point of the whole project, not a defect: enc_bias sits at 0.90,
#   the most strongly propagating kind in the catalogue. A gate that demanded local loudness would reject
#   precisely the faults inversion exists to solve.
print("\n=== BIND CHECK (mid severity) ===", flush=True)
on = T // 3
_on = torch.full((N,), on, device=dev)
_ref, _, _st = rollout(777)
_sd = _ref[on:].std(axis=(0, 1)) + 1e-6
SRC_FOR_CHECK = {"actuator": J0 + 4, "encoder": J0 + 4, "imu": 1, "foot": F0 + 1}
bad, stats = [], {}
for ki, kname in enumerate(KINDS):
    lo, hi = SEV[kname]
    src = SRC_FOR_CHECK[FAMILY[kname]]
    kind = torch.full((N,), ki, dtype=torch.long, device=dev)
    node = torch.full((N,), src, dtype=torch.long, device=dev)
    sev = torch.full((N,), 0.5 * (lo + hi), device=dev)
    s_f, _, _ = rollout(777, faults=(kind, node, sev, schedule(_on, torch.zeros(N, device=dev), dev)), state=_st)
    z = np.abs(s_f[on:] - _ref[on:]) / _sd                       # (T',N,NNODE,NCH) scale-free deviation
    per_node = z.max(axis=3).mean(axis=0)                        # (N,NNODE) best-channel effect at each node
    flat = z.mean(axis=(0, 3))                                   # the old mean-over-channels view, for the record
    ratio = per_node[:, src] / (np.median(per_node, axis=1) + 1e-9)
    med = float(np.median(ratio))
    frac = float((ratio > 1.0).mean())
    absz = float(np.median(per_node[:, src]))                    # does the source move at all, in sigma?
    hardfrac = float((per_node.argmax(1) != src).mean())
    stats[kname] = dict(ratio=med, above_median=frac, source_sigma=absz, cause_victim_gap=hardfrac,
                        ratio_meanchan=float(np.median(flat[:, src] / (np.median(flat, axis=1) + 1e-9))))
    print(f"  {kname:16s} [{FAMILY[kname]:8s}] effect ratio {med:8.2f}x  above-median {frac:.3f}  "
          f"source {absz:6.2f} sigma  loudest!=source {hardfrac:.3f}", flush=True)
    if med < 1.5 or frac < 0.9 or absz < ABS_SIGMA_FLOOR:
        bad.append(kname)
if bad:
    print(f"\nFAIL: these kinds leave no attributable signature: {bad}. Refusing to write a corpus whose labels "
          f"are unrecoverable -- v1's every-kind ratio was ~1.00 and a fully supervised model topped out at "
          f"0.190.", flush=True)
    env.close(); app.close(); sys.exit(2)
print("BIND CHECK PASSED\n", flush=True)
json.dump(stats, open(f"{args.out}/bind_check.json", "w"), indent=2)
if args.check_only:
    env.close(); app.close(); sys.exit(0)


def save(tag, b, s, c, labels=None):
    fn = f"{args.out}/anymal_{tag}_{b:03d}.npz"
    kw = {"labels": json.dumps(labels)} if labels else {}
    np.savez_compressed(fn, signals=s.transpose(1, 0, 2, 3), actions=c.transpose(1, 0, 2), **kw)
    manifest.append({"file": os.path.basename(fn), "kind": tag, "n": N})
    print(f"anymal {tag} {b}", flush=True)


CANDIDATES = [1] + list(range(J0, J0 + NJ)) + list(range(F0, F0 + 4))     # imu, 12 joints, 4 feet
NODE_FAMILY = {1: "imu", **{v: None for v in range(J0, J0 + NJ)}, **{v: "foot" for v in range(F0, F0 + 4)}}
KINDS_BY_FAMILY = {"imu": IMU_KINDS, "foot": FOOT_KINDS, "joint": ACT_KINDS + ENC_KINDS}


def sample_faults(rng):
    """Draw the NODE uniformly over the 17 candidates, then the kind from that node's family.

    The obvious alternative -- one kind per batch, node drawn within the kind's family -- is what a round robin
    over KINDS produces, and it is badly unbalanced: the four IMU kinds all land on the single imu node, so with
    13 kinds that ONE node collects 4/13 of every fault episode while each of the 12 leg joints collects 1/13
    spread over 12 nodes. Node 1 would hold ~31% of the corpus against a 1/17 = 5.9% chance level, and node 1 is
    also the lowest-indexed candidate -- exactly the node `zz_constant` names every time. The reference floor
    that makes a margin readable as a margin would have read ~0.31, above most real methods.

    Drawing the node first makes every candidate equally likely by construction. Mixing kinds within a batch is
    safe here because every mechanism is already masked per environment (scale, coulomb, vlim and the encoder
    error are all (N, NJ) tensors gated by `kind ==`); only the jam damping write is batch-level, and it is
    itself masked."""
    node = np.array(rng.choice(CANDIDATES, N))
    fam = np.array([NODE_FAMILY[v] or "joint" for v in node])
    kname = np.array([KINDS_BY_FAMILY[fm][rng.integers(len(KINDS_BY_FAMILY[fm]))] for fm in fam])
    kind = np.array([KINDS.index(k) for k in kname])
    lo = np.array([SEV[k][0] for k in kname]); hi = np.array([SEV[k][1] for k in kname])
    sev = (lo + (hi - lo) * rng.random(N)).astype(np.float32)
    return node, kname, kind, sev


rng = np.random.default_rng(0)
for b in range(args.healthy_batches):
    s, c, _ = rollout(b); save("healthy", b, s, c)

for b in range(args.fault_batches):
    node_np, kname_np, kind_np, sev_np = sample_faults(rng)
    kind = torch.tensor(kind_np, dtype=torch.long, device=dev)
    node = torch.tensor(node_np, device=dev)
    sev = torch.tensor(sev_np, device=dev)
    onset = torch.tensor(rng.integers(int(0.25 * T), int(0.65 * T), N), device=dev)
    ramp = torch.tensor(np.where(rng.random(N) < 0.5, 0, rng.integers(10, 40, N)), device=dev)
    s, c, _ = rollout(100 + b, faults=(kind, node, sev, schedule(onset, ramp, dev)))
    save("fault", b, s, c, [{"kind": str(kname_np[e]), "family": FAMILY[str(kname_np[e])],
                             "joint": int(node_np[e]), "severity": float(sev_np[e]), "onset": int(onset[e]),
                             "ramp": int(ramp[e]), "task": args.task} for e in range(N)])

# Unit-fault sweep -> one operator column per FAILABLE node (imu, 12 joints, 4 feet), each with the canonical
# kind for its family at a fixed severity, against a matched healthy twin from the same post-reset state.
FAILABLE = CANDIDATES
SWEEP_KIND = {"imu": "imu_bias", "actuator": "torque_loss", "foot": "contact_bias"}
SWEEP_SEV = {"imu_bias": 3.0, "torque_loss": 0.70, "contact_bias": 3.0}
for b in range(args.sweep_batches):
    S = 500 + b
    node_np = np.array([FAILABLE[i % len(FAILABLE)] for i in range(N)])
    kname_np = np.array(["imu_bias" if v == 1 else ("contact_bias" if v >= F0 else "torque_loss")
                         for v in node_np])
    kind = torch.tensor([KINDS.index(k) for k in kname_np], device=dev)
    node = torch.tensor(node_np, device=dev)
    sev = torch.tensor(np.array([SWEEP_SEV[k] for k in kname_np], np.float32), device=dev)
    onset = torch.full((N,), T // 3, device=dev)
    s_ref, c_ref, st = rollout(S)
    s, c, _ = rollout(S, faults=(kind, node, sev, schedule(onset, torch.zeros(N, device=dev), dev)), state=st)
    pre = np.abs(s[:on] - s_ref[:on]).mean(); post = np.abs(s[on:] - s_ref[on:]).mean()
    print(f"  sweep {b}: pre-onset divergence {pre:.4f} vs post {post:.4f} (ratio {post/max(pre,1e-9):.2f})",
          flush=True)
    lab = [{"kind": str(kname_np[e]), "family": FAMILY[str(kname_np[e])], "joint": int(node_np[e]),
            "severity": float(sev[e]), "onset": T // 3, "ramp": 0, "task": args.task, "sweep": True}
           for e in range(N)]
    save("sweep", b, s, c, lab); save("sweepref", b, s_ref, c_ref, lab)

json.dump(manifest, open(f"{args.out}/manifest.json", "w"))
print(f"TOTAL episodes: {sum(m['n'] for m in manifest if m['kind'] != 'sweepref')} (+{sum(m['n'] for m in manifest if m['kind'] == 'sweepref')} matched healthy references)",
      flush=True)
env.close(); app.close()
print("GEN_ANYMAL_SENSOR DONE", flush=True)
