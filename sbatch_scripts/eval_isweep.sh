#!/bin/bash
#SBATCH --job-name=eval_isweep
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=ailab
#SBATCH --time=07:00:00
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=zz8976@princeton.edu

# --- Unconditional (w=0) / conditional (w=1) / CFG (w=2.3) FID eval ---
# Runs all three guidance settings for ONE checkpoint in a single job, using the
# canonical Heun step-50 sampler. See
# configs/asymflow/asymflow_h_16_r8_imagenet_isweep.py for the exact settings.
#
# usage: sbatch -J <name> sbatch_scripts/eval_wsweep.sh <checkpoint.pth> [num_images]
#   num_images defaults to 10000 (matches the training-time FID curves).
#   50000 gives the canonical FID (~6h: CFG costs 2 net passes/step, uncond and
#   cond one each) -- pass `--time=10:00:00` on the sbatch command line then.
set -eo pipefail

CKPT=$1
NUM_IMAGES=${2:-10000}
if [ -z "$CKPT" ]; then
    echo "usage: sbatch eval_wsweep.sh <checkpoint.pth> [num_images]"; exit 1
fi
if [ ! -f "$CKPT" ]; then echo "no such checkpoint: $CKPT"; exit 1; fi

PROJ=/scratch/gpfs/AM43/zz8976/LakonLab
cd "$PROJ"
mkdir -p "slurm_outputs/${SLURM_JOB_NAME}"

source .venv/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + RANDOM % 20000))
# FID model + reference pkl are local under models/; force offline so nothing
# hits the network (compute nodes have no internet).
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
source "$PROJ/sbatch_scripts/wandb_env.sh"

# read by the config to size both the val dataset and the InceptionMetrics
export LAKON_EVAL_NUM_IMAGES="${NUM_IMAGES}"

NGPU=$(nvidia-smi -L | wc -l)
CONFIG=configs/asymflow/asymflow_h_16_r8_imagenet_isweep.py

echo "host=$(hostname)  gpus=${NGPU}  ckpt=${CKPT}  num_images=${NUM_IMAGES}"
echo "config=${CONFIG}"

torchrun \
    --nnodes=1 \
    --nproc_per_node="${NGPU}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    tools/test.py "${CONFIG}" \
    --ckpt "${CKPT}" \
    --launcher pytorch \
    --diff_seed

echo "done rc=$?"
