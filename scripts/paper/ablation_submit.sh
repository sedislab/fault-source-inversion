#!/bin/bash
# Submit one SLURM GPU job per (embodiment, arm) for the FSI component ablation.
# Usage, from the repository root:
#   export SBATCH_ACCOUNT=<your-gpu-account> SBATCH_PARTITION=<your-gpu-partition>   (see docs/CONFIGURATION.md)
#   bash scripts/paper/ablation_submit.sh <embodiment> [build_seed]    with embodiment in {franka,anymal}
set -u
EMB=${1:?franka|anymal}
BSEED=${2:-0}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
ROOT=${FSI_ROOT:-$REPO}
PY=${FSI_PYTHON:-python3}
SCRATCH=$ROOT/scratch/paper/ablation
CSVDIR=$ROOT/results/paper/ablation/raw
LOGDIR=$ROOT/logs/slurm
mkdir -p "$SCRATCH" "$CSVDIR" "$LOGDIR"

if [ "$EMB" = "franka" ]; then
  RAW=$ROOT/raw/franka_v4; GRAPH=franka_joint; LABEL=franka_v4
else
  RAW=$ROOT/raw/anymal_v4; GRAPH=anymal_sim;  LABEL=anymal_v4
fi

ARMS="full no_attention no_node_emb no_multi_horizon no_denorm no_onset_windows no_masking"
for ARM in $ARMS; do
  EXTRA=""
  [ "$ARM" = "full" ] && EXTRA="--also_eval_no_mag_w"
  JOB=abl_${EMB}_${ARM}_b${BSEED}
  sbatch --gpus-per-node=1 --cpus-per-task=16 \
         --mem=96g --time=02:00:00 --job-name="$JOB" \
         --output="$LOGDIR/${JOB}_%j.out" <<JOBEOF
#!/bin/bash
set -x
export PYTHONPATH=$REPO
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
$PY $REPO/scripts/paper/ablation_run.py \
  --raw $RAW --graph $GRAPH --embodiment $LABEL --arm $ARM \
  --out $SCRATCH/${LABEL}_${ARM}_b${BSEED} --csv_dir $CSVDIR \
  --build_seed $BSEED --eval_seeds 3 $EXTRA
echo "EXIT \$?"
JOBEOF
done
squeue -u "$USER" -o "%.10i %.28j %.8T %.10M" | head -40
