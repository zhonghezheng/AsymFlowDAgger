#!/bin/bash
#SBATCH --job-name=mmd_m2048
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --partition=ailab
#SBATCH --time=4-00:00:00
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=zz8976@princeton.edu

#   sbatch sbatch_scripts/train_regft_mmd_m2048.sh
#   sbatch --dependency=afterok:<smoke job> sbatch_scripts/train_regft_mmd_m2048.sh
# configs/asymflow/asymflow_h_16_r8_imagenet_regft_mmd_m2048_4gpus.py: raw RBF MMD over
# sigma 0.98..0.88, 2048 pooled rollouts vs 10240 pooled targets, weight 100.
# ~45 s/iter once the MMD starts at iter 500 (estimate) -> ~2.5 days for 5000 iters; resubmitting
# resumes from checkpoints/<name>/latest.pth (saved every 500 iters).
# --mem: 10240 target images/iter stream through the u8 cache's page cache on top of the
# ~103G dataloader; della h200 nodes have 1.5T.
set -eo pipefail
PROJ=/scratch/gpfs/AM43/zz8976/LakonLab
cd "$PROJ"
mkdir -p "slurm_outputs/${SLURM_JOB_NAME}"

source .venv/bin/activate
export OMP_NUM_THREADS=8
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + RANDOM % 20000))
export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export LAKON_U8_CACHE=data/cached_latents/train_u8_256   # della: no /dev/shm cache
export LAKON_MMD_WEIGHT="${LAKON_MMD_WEIGHT:-100}"

CONFIG=configs/asymflow/asymflow_h_16_r8_imagenet_regft_mmd_m2048_4gpus.py
echo "host=$(hostname)  mmd_weight=${LAKON_MMD_WEIGHT}  $(date)"

torchrun --nnodes=1 --nproc_per_node=4 \
    --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
    tools/train.py "${CONFIG}" --launcher pytorch --diff_seed

echo "done rc=$?  $(date)"
