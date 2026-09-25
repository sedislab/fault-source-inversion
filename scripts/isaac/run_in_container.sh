#!/bin/bash
# Run a python script inside the Isaac Lab Apptainer container, with Kit's writable cache/data/log dirs and HOME
# on a disk you control (Kit fails on read-only or full filesystems otherwise).
#
# Usage, from the repository root:
#   export FSI_ISAAC_SIF=/path/to/isaac-lab.sif          # required: your Isaac Lab 2.3.x container image
#   export FSI_ISAAC_RT=/path/to/writable/kit/dir         # optional, default $FSI_ROOT/scratch/isaac_rt
#   export FSI_ISAAC_HOME_TEMPLATE=/path/to/container_home # optional: pre-populated HOME copied on first run
#   export FSI_APPTAINER_BINDS=/data,/scratch              # optional: extra bind mounts
#   scripts/isaac/run_in_container.sh scripts/isaac/gen_franka_sensor.py --headless
set -e
REPO=$(cd "$(dirname "$0")/../.." && pwd)
ROOT=${FSI_ROOT:-$REPO}
SIF=${FSI_ISAAC_SIF:?set FSI_ISAAC_SIF to your Isaac Lab .sif image (see docs/CONFIGURATION.md)}
W=${FSI_ISAAC_RT:-$ROOT/scratch/isaac_rt}
mkdir -p "$W/kit_cache" "$W/kit_data" "$W/kit_logs" "$W/home" "$W/ov"
if [ -n "${FSI_ISAAC_HOME_TEMPLATE:-}" ] && [ ! -d "$W/home/.nvidia-omniverse" ]; then
  cp -r "$FSI_ISAAC_HOME_TEMPLATE/." "$W/home/" 2>/dev/null || true
fi
BINDS="-B $REPO"
[ "$ROOT" != "$REPO" ] && BINDS="$BINDS -B $ROOT"
[ -n "${FSI_APPTAINER_BINDS:-}" ] && BINDS="$BINDS -B $FSI_APPTAINER_BINDS"
exec apptainer exec --nv --cleanenv \
  $BINDS \
  -B "$W/kit_cache:/isaac-sim/kit/cache" \
  -B "$W/kit_data:/isaac-sim/kit/data" \
  -B "$W/kit_logs:/isaac-sim/kit/logs" \
  --home "$W/home:/root" \
  --env OMNI_KIT_ACCEPT_EULA=YES --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y \
  --env OMNI_KIT_ALLOW_ROOT=1 --env PYTHONUNBUFFERED=1 \
  --env FSI_ROOT="$ROOT" \
  "$SIF" /isaac-sim/python.sh "$@"
