#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/det_%x_%j.out
# Submit from the repository root. Account and partition come from your environment, never from this file:
#   export SBATCH_ACCOUNT=<your-gpu-account> SBATCH_PARTITION=<your-gpu-partition>   (see docs/CONFIGURATION.md)
#   sbatch scripts/paper/detection_run.sbatch [args for detection_seeds.py]
set -e
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs/slurm
export PYTHONPATH="$PWD"
PY="${FSI_PYTHON:-python3}"
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
echo "host $(hostname) | args: $@"
$PY scripts/paper/detection_seeds.py "$@"
