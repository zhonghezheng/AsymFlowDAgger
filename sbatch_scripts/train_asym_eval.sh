#!/bin/bash
#SBATCH --job-name=train_asym_eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=ailab
#SBATCH --time=01:00:00
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=zz8976@princeton.edu

# --- Smoke test WITH one eval loop (generation + InceptionMetrics/FID + wandb) ---
set -eo pipefail

PROJ=/scratch/gpfs/AM43/zz8976/LakonLab
cd "$PROJ"
mkdir -p "slurm_outputs/${SLURM_JOB_NAME}"

source .venv/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + RANDOM % 20000))
# FID model + reference pkl are stored locally under models/, so no downloads;
# force HF offline so nothing tries the network. wandb offline too (compute nodes
# can't reach api.wandb.ai) -- env overrides config, so it can never hang online.
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
source "$PROJ/sbatch_scripts/wandb_env.sh"

NGPU=$(nvidia-smi -L | wc -l)
CONFIG=configs/asymflow/asymflow_h_16_r8_imagenet_dagger_smoke_eval.py

echo "host=$(hostname)  gpus=${NGPU}  master_port=${MASTER_PORT}"
echo "config=${CONFIG}"

# NOTE: no --no-validate here, so the eval hook runs.
torchrun \
    --nnodes=1 \
    --nproc_per_node="${NGPU}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    tools/train.py "${CONFIG}" \
    --launcher pytorch \
    --diff_seed

echo "done rc=$?"
