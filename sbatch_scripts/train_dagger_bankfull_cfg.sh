#!/bin/bash
#SBATCH --job-name=dagger_cfg
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --partition=ailab
#SBATCH --time=16:00:00
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=zz8976@princeton.edu

#   sbatch -J dcfg10_f50 sbatch_scripts/train_dagger_bankfull_cfg.sh 10 0.5
# $1 = w_cfg (CFG-gap weight), $2 = band_frac_on_path
set -eo pipefail
W="${1:?usage: train_dagger_bankfull_cfg.sh <w_cfg> <frac_on_path>}"
F="${2:?usage: train_dagger_bankfull_cfg.sh <w_cfg> <frac_on_path>}"
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
export LAKON_CFG_GAP_WEIGHT="$W"
export LAKON_BAND_FRAC="$F"

CONFIG=configs/asymflow/asymflow_h_16_r8_imagenet_dagger_bankfull_cfg_4gpus.py
NAME=$(python -c "from mmcv import Config; print(Config.fromfile('$CONFIG').name)")
echo "host=$(hostname)  w_cfg=${W}  frac_on_path=${F}  name=${NAME}"

torchrun --nnodes=1 --nproc_per_node=4 \
    --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
    tools/train.py "${CONFIG}" --launcher pytorch --diff_seed
echo "done rc=$?"
