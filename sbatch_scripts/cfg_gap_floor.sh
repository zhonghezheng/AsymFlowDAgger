#!/bin/bash
#SBATCH --job-name=cfg_gap_floor
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=ailab
#SBATCH --time=01:30:00
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out

# tools/cfg_gap_floor.py: how much of the logged cfg_gap is dropout noise (independent
# masks on the two branches) or bank noise in v*, i.e. a floor no w_cfg can remove.
# Pretrained weights, then the w=0 control and the w=100 arm at iter 5000 -- if w=100
# mainly shrank drop_c/drop_u rather than gap_eval, the gap was buying dropout
# robustness, not CFG alignment.
set -eo pipefail
PROJ=/scratch/gpfs/AM43/zz8976/LakonLab
cd "$PROJ"
mkdir -p "slurm_outputs/${SLURM_JOB_NAME}" direct_outputs

source .venv/bin/activate
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export LAKON_U8_CACHE=data/cached_latents/train_u8_256   # della: no /dev/shm cache

echo "host=$(hostname)  $(date)"
python tools/cfg_gap_floor.py
for w in 0 100; do
    python tools/cfg_gap_floor.py \
        --ckpt "checkpoints/asymflow_h_16_r8_imagenet_dagger_bankfull_cmff${w}_f50_4gpus/latest.pth"
done
echo "done  $(date)"
