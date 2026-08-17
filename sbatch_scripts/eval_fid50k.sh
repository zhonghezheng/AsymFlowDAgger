#!/bin/bash
#SBATCH --job-name=eval_fid50k
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=ailab
#SBATCH --time=03:00:00
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=zz8976@princeton.edu

# --- Canonical 50k-image FID eval (eval-only, tools/test.py) ---
# usage: sbatch -J <name> sbatch_scripts/eval_fid50k.sh <checkpoint.pth>
set -eo pipefail

CKPT=$1
if [ -z "$CKPT" ]; then echo "usage: sbatch eval_fid50k.sh <checkpoint.pth>"; exit 1; fi

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

NGPU=$(nvidia-smi -L | wc -l)
CONFIG=configs/asymflow/asymflow_h_16_r8_imagenet_eval50k.py

echo "host=$(hostname)  gpus=${NGPU}  ckpt=${CKPT}"
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
