#!/bin/bash
#SBATCH --job-name=cfg_ts
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=768G
#SBATCH --partition=ailab
#SBATCH --time=40:00:00
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=zz8976@princeton.edu

#   sbatch -J cfg_ts1b sbatch_scripts/cfg_tempspread.sh 1.0 both
#   sbatch -J cfg_ts1u sbatch_scripts/cfg_tempspread.sh 1.0 null
# $1 = temp_spread (target sd of d2 after rescaling), $2 = scope: both | null
#
# ADAPTIVE softmax temperature in x0_hat: d2 -> d2 / (sd(d2)/temp_spread), clamped so
# it only ever smooths. The raw spread runs 10.2 at sigma=0.98 to 287 at sigma=0.88,
# so a FIXED temperature cannot serve the whole band -- T=100 left spread 1.19 at
# sigma=0.92 but 0.10 at 0.98, i.e. the plain bank mean there, and that run degraded
# (+0.141 over control by iter 1500). Targeting the SPREAD instead holds the
# smoothing constant in the units that decide softmax saturation.
#
# scope=null smooths only v*_uncond; scope=both also smooths v*_cond, making it a
# class-restricted posterior MEAN rather than the nearest-neighbour lookup it is at
# T=1 (measured ESS ~1.1 of 2048 at sigma=0.92).
#
# Control: cmff10_f0 (identical, no temperature) -- 4.4401 @1500, 5.149 @5000.
set -eo pipefail
TS="${1:-1.0}"
SCOPE="${2:-both}"
KSPACE="${3:-feat}"   # posterior-weight space: feat | latent
PROJ=/scratch/gpfs/AM43/zz8976/LakonLab
cd "$PROJ"
mkdir -p "slurm_outputs/${SLURM_JOB_NAME}"

source .venv/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + RANDOM % 20000))
export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export LAKON_CFG_CM=full
export LAKON_CFG_GAP_WEIGHT=10
export LAKON_BAND_FRAC=0
# --- cost settings, set HERE rather than inherited from the submitting shell ---
# (an unexported var silently reverts to the config default, which is how an earlier
#  pair of these jobs ended up at mmd_interval=8 without anyone noticing).
#   interval 1 : the band fires every iteration, not every 8th
#   ncls 8     : 8 distinct classes per band -> the per-class bank dedup has
#                something to merge (21 trajectories over ~21 classes has nothing)
#   null 1024  : at T=1 the x0_hat softmax ESS is flat in bank size, but with
#                adaptive temperature on ESS/M is ~constant, i.e. ESS grows with M --
#                so a temperature arm wants the bigger bank. 1024 is the compromise
#                between that and the IO: 2048 would be ~48-50 h at interval 1.
# Per banded iteration: 8*1300 cond + 21*1024 null ~ 32k images (vs ~70k unrestricted)
# -> ~24-25 s/iter, ~30-32 h for 5000 iters (anchor: smoke_c8, 53k img @ 38 s).
export LAKON_BAND_INTERVAL=1
export LAKON_BAND_NCLS=8
export LAKON_NULL_BANK=1024
export LAKON_TEMP_SPREAD="$TS"
export LAKON_TEMP_SCOPE="$SCOPE"
export LAKON_KERNEL_SPACE="$KSPACE"

CONFIG=configs/asymflow/asymflow_h_16_r8_imagenet_dagger_bankfull_cfg_4gpus.py
NAME=$(python -c "from mmcv import Config; print(Config.fromfile('$CONFIG').name)")
echo "host=$(hostname)  temp_spread=${TS}  scope=${SCOPE}  name=${NAME}"

torchrun --nnodes=1 --nproc_per_node=4 \
    --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
    tools/train.py "${CONFIG}" --launcher pytorch --diff_seed

echo "done rc=$?"
