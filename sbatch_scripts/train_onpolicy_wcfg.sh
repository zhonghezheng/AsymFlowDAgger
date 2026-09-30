#!/bin/bash
#SBATCH --job-name=onpolicy_wcfg
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

# Train ONE onpolicy CFG-gap weight. Slurm counterpart of run_onpolicy_wcfg.sh
# (which is the interactive-node runner and also chains the wsweep).
#
#   sbatch -J onpolicy_wcfg10 sbatch_scripts/train_onpolicy_wcfg.sh 10
#
# w_cfg scales the full CFG term
#   w_cfg * || (v_cond - v_uncond) - (v*_cond - v*_uncond) ||^2
# and also names the run, so each weight gets its own checkpoints/ dir.
#
# --mem=384G: a plain reg_ft run already peaks at ~103G RSS, and this arm adds the
# expert's bank IO on top (full class + 2048 null per trajectory) whose page cache
# counts against the cgroup -- the DAGGER bankfull run was OOM-killed at 256G.
set -eo pipefail

W="${1:?usage: sbatch train_onpolicy_wcfg.sh <w_cfg>}"
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

NGPU=4
CONFIG=configs/asymflow/asymflow_h_16_r8_imagenet_onpolicy_wcfg_4gpus.py
NAME=$(python -c "from mmcv import Config; print(Config.fromfile('$CONFIG').name)")
echo "host=$(hostname)  gpus=${NGPU}  w_cfg=${W}  name=${NAME}"
echo "config=${CONFIG}"

torchrun \
    --nnodes=1 \
    --nproc_per_node="${NGPU}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    tools/train.py "${CONFIG}" \
    --launcher pytorch \
    --diff_seed

echo "done rc=$?"
