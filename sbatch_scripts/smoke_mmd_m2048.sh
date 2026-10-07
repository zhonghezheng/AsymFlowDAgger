#!/bin/bash
#SBATCH --job-name=smoke_mmd_m2048
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --partition=ailab
#SBATCH --time=01:30:00
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out

# Gate for train_regft_mmd_m2048.sh (submit that with --dependency=afterok:<this job>):
#   1. tools/check_mmd_chunked.py -- sharded estimator == mmd2_rbf, chunked gradient ==
#      one-graph gradient on the real model; exits 1 on a mismatch.
#   2. 6 iterations of the m2048 config at full size with the MMD from iter 0 -- peak
#      memory, s/iter, mmd_replay_err, DDP coexistence.
set -eo pipefail
PROJ=/scratch/gpfs/AM43/zz8976/LakonLab
cd "$PROJ"
mkdir -p "slurm_outputs/${SLURM_JOB_NAME}"

source .venv/bin/activate
export OMP_NUM_THREADS=8
export MASTER_ADDR=127.0.0.1
export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export LAKON_U8_CACHE=data/cached_latents/train_u8_256   # della: no /dev/shm cache

echo "host=$(hostname)  $(date)"
torchrun --nnodes=1 --nproc_per_node=4 \
    --master_addr="${MASTER_ADDR}" --master_port=$((10000 + RANDOM % 20000)) \
    tools/check_mmd_chunked.py

torchrun --nnodes=1 --nproc_per_node=4 \
    --master_addr="${MASTER_ADDR}" --master_port=$((10000 + RANDOM % 20000)) \
    tools/train.py configs/asymflow/asymflow_h_16_r8_imagenet_regft_mmd_m2048_smoke.py \
    --launcher pytorch --diff_seed

echo "done rc=$?  $(date)"
