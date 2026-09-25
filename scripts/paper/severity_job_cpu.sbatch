#!/bin/bash
#SBATCH --job-name=severity_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/severity_cpu_%j.out
# Submit from the repository root. Account and partition come from your environment, never from this file:
#   export SBATCH_ACCOUNT=<your-cpu-account> SBATCH_PARTITION=<your-cpu-partition>   (see docs/CONFIGURATION.md)
#   sbatch scripts/paper/severity_job_cpu.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs/slurm
export PYTHONPATH="$PWD"
PY="${FSI_PYTHON:-python3}"
OUT="${FSI_ROOT:-$PWD}/results/paper/severity"
export OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
$PY scripts/paper/severity_run.py \
  --out "$OUT" \
  --split_seeds 0 1 2 3 4 --model_seeds 5 --threads 16
$PY scripts/paper/severity_agg.py "$OUT"
echo SEVERITY_JOB_DONE
