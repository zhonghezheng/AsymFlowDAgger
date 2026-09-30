#!/bin/bash
#SBATCH --job-name=smoke_mmd
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --partition=ailab
#SBATCH --time=00:50:00
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=zz8976@princeton.edu

#   sbatch sbatch_scripts/smoke_mmd.sh [subspace|raw] [weight]
# 2 GPUs, not 1: _gather_feats returns early at world_size==1, so a single-rank smoke
# would leave the all_gather and the *world_size loss scaling untested -- and those
# are among the paths that have executed exactly once.
set -eo pipefail
FEAT="${1:-subspace}"
W="${2:-100}"
PROJ=/scratch/gpfs/AM43/zz8976/LakonLab
cd "$PROJ"
mkdir -p "slurm_outputs/${SLURM_JOB_NAME}"

source .venv/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + RANDOM % 20000))
export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export LAKON_MMD_FEATURE="$FEAT"
export LAKON_MMD_WEIGHT="$W"

CONFIG=configs/asymflow/asymflow_h_16_r8_imagenet_regft_mmd_sweep_smoke.py
echo "host=$(hostname)  feature=${FEAT}  weight=${W}  config=${CONFIG}"

NPROC=${SLURM_GPUS_ON_NODE:-2}
echo "nproc_per_node=${NPROC}"
torchrun --nnodes=1 --nproc_per_node="${NPROC}" \
    --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
    tools/train.py "${CONFIG}" --launcher pytorch --no-validate --diff_seed

echo "done rc=$?"
