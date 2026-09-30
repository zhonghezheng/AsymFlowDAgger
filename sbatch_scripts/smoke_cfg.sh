#!/bin/bash
#SBATCH --job-name=smoke_cfg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=384G
#SBATCH --partition=ailab
#SBATCH --time=00:50:00
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=zz8976@princeton.edu

#   sbatch -J smoke_cfg_pp sbatch_scripts/smoke_cfg.sh project project 1 0.5
# $1 = cfg complement mode, $2 = emp_fm complement mode, $3 = w_cfg, $4 = frac_on_path
# 1 GPU is enough: the CFG band losses are per-rank (the only gather in this class is
# on the MMD path, and mmd_weight=0 here). Keeps samples_per_gpu=256, so the per-GPU
# memory profile is the real one. Host mem stays at the arm's 384G: the expert banks
# are rebuilt every banded iteration.
set -eo pipefail
CCM="${1:-full}"
FCM="${2:-full}"
W="${3:-1}"
F="${4:-0.5}"
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
export LAKON_CFG_CM="$CCM"
export LAKON_FM_CM="$FCM"
export LAKON_CFG_GAP_WEIGHT="$W"
export LAKON_BAND_FRAC="$F"

CONFIG=configs/asymflow/asymflow_h_16_r8_imagenet_dagger_bankfull_cfg_smoke.py
NAME=$(python -c "from mmcv import Config; print(Config.fromfile('$CONFIG').name)")
echo "host=$(hostname)  cfg_cm=${CCM}  fm_cm=${FCM}  w=${W}  frac=${F}  name=${NAME}"

torchrun --nnodes=1 --nproc_per_node=1 \
    --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
    tools/train.py "${CONFIG}" --launcher pytorch --no-validate --diff_seed

echo "done rc=$?"
