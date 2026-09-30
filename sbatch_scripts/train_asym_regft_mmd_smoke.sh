#!/bin/bash
#SBATCH --job-name=regft_mmd_smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=ailab
#SBATCH --time=01:30:00
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=zz8976@princeton.edu

# 1-GPU smoke: samples_per_gpu is unchanged (256), so this reproduces the real
# per-GPU memory profile of the FM step + the differentiable MMD band rollout.
# LAKON_MMD_BATCH ($1, default 64) is the memory knob.
set -eo pipefail
PROJ=/scratch/gpfs/AM43/zz8976/LakonLab
cd "$PROJ"
mkdir -p "slurm_outputs/${SLURM_JOB_NAME}"

source .venv/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + RANDOM % 20000))
export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export LAKON_MMD_BATCH=${1:-64}
export LAKON_MMD_FEATURE=${2:-both}

CONFIG=configs/asymflow/asymflow_h_16_r8_imagenet_regft_mmd_smoke.py
echo "host=$(hostname)  mmd_batch=${LAKON_MMD_BATCH}  feature=${LAKON_MMD_FEATURE}  config=${CONFIG}"

torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    tools/train.py "${CONFIG}" \
    --launcher pytorch \
    --no-validate \
    --diff_seed

echo "done rc=$?"
