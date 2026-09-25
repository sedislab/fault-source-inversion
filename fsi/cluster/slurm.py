"""SLURM submission helpers. GPU jobs pick, among the accounts you list, the one with the best fairshare that
still has hours; CPU jobs go to a single CPU account. Every job writes its sbatch script under logs/slurm/ so runs
are reproducible and inspectable, and accounts with no hours left are skipped.

Nothing site-specific is hard-coded. Accounts, partitions, the container image and scratch space come from
environment variables (see docs/CONFIGURATION.md and cluster.env.example); keep your real values in an untracked
cluster.env and `source` it before submitting.
"""
from __future__ import annotations
import os, subprocess, shlex, re, sys
from dataclasses import dataclass
from pathlib import Path
from ..paths import REPO, ROOT

def _env_list(name):
    return [x.strip() for x in os.environ.get(name, "").split(",") if x.strip()]

GPU_ACCOUNTS = _env_list("FSI_GPU_ACCOUNTS")             # e.g. "acct1-gpu,acct2-gpu"
CPU_ACCOUNT = os.environ.get("FSI_CPU_ACCOUNT")          # e.g. "acct-cpu"
GPU_PARTITION = os.environ.get("FSI_GPU_PARTITION", "gpu")
CPU_PARTITION = os.environ.get("FSI_CPU_PARTITION", "cpu")
SIF = os.environ.get("FSI_ISAAC_SIF", str(ROOT / "isaac-lab.sif"))
ENV_SH = os.environ.get("FSI_ISAAC_ENV_SH")              # optional site script sourced before each job
BINDS = os.environ.get("FSI_APPTAINER_BINDS", "")        # extra apptainer -B mounts, comma separated
ACCOUNTS_CMD = os.environ.get("FSI_ACCOUNTS_CMD", "accounts")   # site tool that prints hours per account
SCRATCH = Path(os.environ.get("FSI_SCRATCH", ROOT / "scratch"))
LOGDIR = ROOT / "logs" / "slurm"

def _run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)

def account_hours() -> dict:
    """Remaining hours per account from the site accounting tool (FSI_ACCOUNTS_CMD); empty if unreachable.

    Parse the BALANCE column (the first number after the account name), not the end of the line: the tool's
    layout is `account  balance  deposited  project`, and the project title is truncated with an ellipsis
    ("a long project na..."), so an end-anchored `([\\d.]+)\\s*$` matches the dots and raises. A balance can
    be negative (an exhausted account), which the old pattern also could not express -- and reading a negative
    balance as unavailable is the whole point of the check."""
    out = _run(f"{ACCOUNTS_CMD} 2>/dev/null").stdout
    hours = {}
    for line in out.splitlines():
        m = re.match(r"\s*(\S+-(?:gpu|cpu))\s+(-?[\d.]+)\b", line)
        if m:
            hours[m.group(1)] = float(m.group(2))
    return hours

def fairshare() -> dict:
    """FairShare factor per account (higher = more entitled to run now). Empty if SLURM is down."""
    out = _run("sshare -n -P -o Account,FairShare -U 2>/dev/null").stdout
    fs = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) >= 2 and parts[0].strip():
            try:
                fs[parts[0].strip()] = float(parts[1])
            except ValueError:
                pass
    return fs

def pick_gpu_account(candidates=None, min_hours=1.0) -> str:
    """Highest-fairshare candidate account that still has hours. Falls back to the first candidate if SLURM
    is unreachable (verify once it is back)."""
    cands = candidates or GPU_ACCOUNTS
    if not cands:
        raise ValueError("no GPU account configured: set FSI_GPU_ACCOUNTS or pass account=...")
    fs, hrs = fairshare(), account_hours()
    viable = [a for a in cands if hrs.get(a, min_hours + 1) >= min_hours]
    if not viable:
        viable = cands
    return max(viable, key=lambda a: fs.get(a, 0.0)) if fs else viable[0]

@dataclass
class SlurmJob:
    name: str
    job_id: str | None
    script: Path
    account: str
    qos: str

def submit(name, cmd, kind="gpu", gpus=1, cpus=16, mem="64g", time="04:00:00", account=None,
           qos=None, partition=None, preempt=True, container=False, depends=None, dry_run=False,
           extra_sbatch=None) -> SlurmJob:
    """Write and (unless dry_run) sbatch-submit a job. GPU picks an account by fairshare; CPU uses FSI_CPU_ACCOUNT."""
    LOGDIR.mkdir(parents=True, exist_ok=True)
    if kind == "cpu":
        account = account or CPU_ACCOUNT
        if not account:
            raise ValueError("no CPU account configured: set FSI_CPU_ACCOUNT or pass account=...")
        partition = partition or CPU_PARTITION
        gpu_line = ""
    else:
        account = account or pick_gpu_account()
        partition = partition or GPU_PARTITION
        gpu_line = f"#SBATCH --gpus-per-node={gpus}\n"
    qos_line = f"#SBATCH --qos={qos}\n" if qos else ""  # account default QOS otherwise
    if container:
        binds = f"-B {BINDS} " if BINDS else ""
        cmd = f"apptainer exec --nv {binds}{SIF} {cmd}"
    dep = f"#SBATCH --dependency=afterok:{depends}\n" if depends else ""
    extra = "".join(f"#SBATCH {x}\n" for x in (extra_sbatch or []))
    script = LOGDIR / f"{name}.sbatch"
    script.write_text(
        f"#!/bin/bash\n#SBATCH --job-name=fsi_{name}\n#SBATCH --account={account}\n"
        f"#SBATCH --partition={partition}\n{qos_line}{gpu_line}"
        f"#SBATCH --cpus-per-task={cpus}\n#SBATCH --mem={mem}\n#SBATCH --time={time}\n"
        f"#SBATCH --output={LOGDIR}/{name}_%j.out\n#SBATCH --error={LOGDIR}/{name}_%j.out\n{dep}{extra}\n"
        # PY is the interpreter that SUBMITTED the job, so a job runs against the same environment it was written
        # for. A site env script (FSI_ISAAC_ENV_SH) may activate a different interpreter that lacks pandas, so jobs
        # call $PY rather than a bare `python3`. Scratch dirs are pointed at FSI_SCRATCH after sourcing it, because
        # torch writes generated code under TMPDIR on import and a full TMPDIR kills a job from inside an import.
        # Isaac Lab scripts do not go through this path; they run inside the container via
        # scripts/isaac/run_in_container.sh and supply their own interpreter.
        "set -e\n"
        + (f"source {shlex.quote(ENV_SH)} 2>/dev/null || true\n" if ENV_SH else "")
        + f"export TMPDIR={SCRATCH}/tmp XDG_CACHE_HOME={SCRATCH}/xdg PYTORCH_KERNEL_CACHE_PATH={SCRATCH}/torch\n"
        f"mkdir -p $TMPDIR $XDG_CACHE_HOME $PYTORCH_KERNEL_CACHE_PATH\n"
        f"export PY={shlex.quote(sys.executable)} PYTHONPATH={shlex.quote(str(REPO))}\n"
        f"cd {shlex.quote(str(REPO))}\n{cmd}\n")
    if dry_run:
        return SlurmJob(name, None, script, account, qos)
    res = _run(f"sbatch {shlex.quote(str(script))}")
    jid = res.stdout.strip().split()[-1] if res.returncode == 0 else None
    return SlurmJob(name, jid, script, account, qos)
