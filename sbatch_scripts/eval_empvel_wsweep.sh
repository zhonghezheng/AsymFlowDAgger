#!/bin/bash
#SBATCH --job-name=eval_empvel
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --partition=ailab
#SBATCH --time=08:00:00
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=zz8976@princeton.edu

# --- Empirical v*_cond on high-sigma windows, CFG-w sweep on [0, 0.88] (eval only) ---
# See configs/asymflow/asymflow_h_16_r8_imagenet_empvel_wsweep_h200.py. To run the
# windows as parallel jobs (one each), use sbatch_scripts/submit_empvel_sweep.sh.
#
# usage: mkdir -p slurm_outputs/<name>    # Slurm opens the log before this script runs
#        sbatch -J <name> sbatch_scripts/eval_empvel_wsweep.sh [checkpoint.pth | base] [num_images]
#   no checkpoint (or 'base') -> the released base weights. num_images defaults to 10000.
#   Env knobs pass through to the config: LAKON_GUIDANCE, LAKON_EV_WINDOWS, LAKON_EV_ARMS,
#   LAKON_EV_KERNEL, LAKON_EV_READ_THREADS, LAKON_U8_CACHE ('' forces JPEGs).
#
# COST (4 GPUs, 10k images, 6 w points): sampling ~1h20 per window by the wsweep's
# timing (~7 min at w = 1, ~14 min per CFG point). A v* window adds bank IO -- every
# eval batch loads its 64 rows' ENTIRE classes (~16 GB uint8 per rank) from the GPFS
# u8 cache: ~14 s per batch on the 1-GPU smoke (32 threads), ~9 min per point at 4 GPUs
# if per-rank throughput holds -> ~2h15 per window. All four default windows in ONE job
# is ~8h, at the limit: fan them out. --gres=gpu:8 --cpus-per-task=64 --mem=512G
# halves the time.
set -eo pipefail

CKPT="${1:-base}"
NUM_IMAGES="${2:-10000}"

PROJ=/scratch/gpfs/AM43/zz8976/LakonLab
cd "$PROJ"
if [ "$CKPT" = base ]; then
    CKPT=""
elif [ ! -f "$CKPT" ]; then
    echo "no such checkpoint: $CKPT"; exit 1
fi
mkdir -p "slurm_outputs/${SLURM_JOB_NAME}"

source .venv/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + RANDOM % 20000))
# FID model + reference pkl are local under models/; force offline so nothing
# hits the network (compute nodes have no internet).
export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
source "$PROJ/sbatch_scripts/wandb_env.sh"

# read by the config to size both the val dataset and the InceptionMetrics
export LAKON_EVAL_NUM_IMAGES="${NUM_IMAGES}"
# della's copy of the u8 image cache (verified against train.txt + JPEGs 2026-10-06).
# '-' not ':-', so an explicit LAKON_U8_CACHE='' still forces the JPEG path.
export LAKON_U8_CACHE="${LAKON_U8_CACHE-$PROJ/data/cached_latents/train_u8_256}"

NGPU=$(nvidia-smi -L | wc -l)
CONFIG=configs/asymflow/asymflow_h_16_r8_imagenet_empvel_wsweep_h200.py

echo "host=$(hostname)  gpus=${NGPU}  ckpt=${CKPT:-<base weights>}  num_images=${NUM_IMAGES}"
echo "config=${CONFIG}  u8=${LAKON_U8_CACHE:-<JPEG>}"
echo "w=${LAKON_GUIDANCE:-<default>}  windows=${LAKON_EV_WINDOWS:-<default>}" \
     "arms=${LAKON_EV_ARMS:-<default>}  kernel=${LAKON_EV_KERNEL:-<default>}"

torchrun \
    --nnodes=1 \
    --nproc_per_node="${NGPU}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    tools/test.py "${CONFIG}" \
    ${CKPT:+--ckpt "$CKPT"} \
    --launcher pytorch \
    --diff_seed

echo "done rc=$?"
