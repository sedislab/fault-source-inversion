"""Probe the Isaac Lab task registry and dump a robot's articulation structure (joints, bodies, actuators) so we
can finalize the node->sim bindings. Run in-container: /isaac-sim/python.sh scripts/isaac/probes/inspect_env.py --headless"""
import argparse
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", default=None)
p.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
print("APP STARTING", flush=True)
app = AppLauncher(args).app
print("APP UP", flush=True)

import gymnasium as gym
import isaaclab_tasks  # noqa: F401  registers envs
from isaaclab_tasks.utils import parse_env_cfg

ids = sorted(str(k) for k in gym.registry.keys())
print("REGISTRY LOADED", flush=True)
print("TOTAL ENVS", len(ids))
for pat in ("Franka", "Anymal", "Quadcopter", "Crazyflie", "Go2", "Leatherback", "Carter", "Cartpole", "Reach", "Velocity"):
    hits = [i for i in ids if pat.lower() in i.lower()]
    if hits:
        print(f"PAT {pat} -> {hits[:10]}")

for task in ([args.task] if args.task else ["Isaac-Reach-Franka-v0", "Isaac-Velocity-Flat-Anymal-C-v0", "Isaac-Quadcopter-Direct-v0"]):
    try:
        cfg = parse_env_cfg(task, num_envs=args.num_envs)
        env = gym.make(task, cfg=cfg)
        u = env.unwrapped
        r = u.scene["robot"]
        print(f"\n=== {task} ===")
        print("num_envs", u.num_envs, "device", u.device)
        print("JOINTS", list(r.joint_names))
        print("BODIES", list(r.body_names))
        print("ACTUATORS", list(r.actuators.keys()))
        try:
            print("SENSORS", list(u.scene.sensors.keys()))
        except Exception as e:
            print("SENSORS n/a", e)
        env.close()
    except Exception as e:
        print(f"\n=== {task} FAILED: {type(e).__name__}: {str(e)[:200]}")

app.close()
