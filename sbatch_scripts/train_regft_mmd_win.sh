#!/bin/bash
#SBATCH --job-name=regft_mmd_win
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --partition=ailab
#SBATCH --time=20:00:00
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=zz8976@princeton.edu

#   sbatch -J mmd_sub10 sbatch_scripts/train_regft_mmd_sweep.sh subspace 10
# No expert banks in this arm (expert=None), so host RAM is just the dataloader
# (~103G measured for reg_ft); 192G leaves headroom. GPU peaked at 79G.
set -eo pipefail
FEAT="${1:?usage: train_regft_mmd_sweep.sh <subspace|raw> <weight>}"
W="${2:?usage: train_regft_mmd_sweep.sh <subspace|raw> <weight>}"
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
export LAKON_MMD_FEATURE="$FEAT"
export LAKON_MMD_WEIGHT="$W"

CONFIG=configs/asymflow/asymflow_h_16_r8_imagenet_regft_mmd_win_4gpus.py
NAME=$(python -c "from mmcv import Config; print(Config.fromfile('$CONFIG').name)")
echo "host=$(hostname)  feature=${FEAT}  mmd_weight=${W}  name=${NAME}"

torchrun --nnodes=1 --nproc_per_node=4 \
    --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
    tools/train.py "${CONFIG}" --launcher pytorch --diff_seed

echo "done rc=$?"
