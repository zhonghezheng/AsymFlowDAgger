#!/bin/bash
#SBATCH --job-name=eval_splice
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=450G
#SBATCH --partition=ailab
#SBATCH --time=12:00:00
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=zz8976@princeton.edu

# Inference-time expert-velocity splice on the BASE model (no --ckpt; base weights
# via denoising.pretrained). Single GPU: each rank caches full-class banks (~dataset
# size) in host RAM. See configs/asymflow/asymflow_h_16_r8_imagenet_splice.py
set -eo pipefail

NUM_IMAGES=${1:-10000}
PROJ=/scratch/gpfs/AM43/zz8976/LakonLab
cd "$PROJ"
mkdir -p "slurm_outputs/${SLURM_JOB_NAME}"

source .venv/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + RANDOM % 20000))
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True
source "$PROJ/sbatch_scripts/wandb_env.sh"
export LAKON_EVAL_NUM_IMAGES="${NUM_IMAGES}"

CONFIG=${2:-configs/asymflow/asymflow_h_16_r8_imagenet_splice.py}
echo "host=$(hostname)  num_images=${NUM_IMAGES}  config=${CONFIG}  (BASE model, no ckpt)"

torchrun --nnodes=1 --nproc_per_node=1 \
    --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
    tools/test.py "${CONFIG}" --launcher pytorch --diff_seed

echo "done rc=$?"
