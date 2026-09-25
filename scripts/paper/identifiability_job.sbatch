#!/bin/bash
#SBATCH --job-name=ident
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/identifiability_%j.out
# Submit from the repository root. Account and partition come from your environment, never from this file:
#   export SBATCH_ACCOUNT=<your-gpu-account> SBATCH_PARTITION=<your-gpu-partition>   (see docs/CONFIGURATION.md)
#   sbatch scripts/paper/identifiability_job.sbatch
set -e
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs/slurm
export PYTHONPATH="$PWD"
PY="${FSI_PYTHON:-python3}"
export OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
$PY scripts/paper/identifiability_confusion.py --seeds 5 --perms 2000 --threads 16
