#!/bin/bash
# --- Submit the uncond / cond / CFG eval for BOTH comparison models ---
# Login-node driver (not an sbatch script itself): submits one eval_wsweep job per
# checkpoint, each of which runs all three guidance settings.
#
#   usage: bash sbatch_scripts/eval_wsweep_all.sh [num_images]
#
# num_images defaults to 10000. For the canonical 50k FID:
#   NUM_IMAGES=50000 SBATCH_EXTRA="--time=10:00:00" bash sbatch_scripts/eval_wsweep_all.sh
set -eo pipefail

PROJ=/scratch/gpfs/AM43/zz8976/LakonLab
cd "$PROJ"

NUM_IMAGES=${1:-${NUM_IMAGES:-10000}}

# job-name -> checkpoint. `latest.pth` so this keeps working as the runs advance.
declare -A CKPTS=(
    [wsweep_regft]=checkpoints/asymflow_h_16_r8_imagenet_regft_4gpus/latest.pth
    [wsweep_dagger]=checkpoints/asymflow_h_16_r8_imagenet_dagger_full_bank128_4gpus/latest.pth
)

for JOB in "${!CKPTS[@]}"; do
    CKPT=${CKPTS[$JOB]}
    if [ ! -f "$CKPT" ]; then
        echo "SKIP ${JOB}: missing ${CKPT}"
        continue
    fi
    echo "submitting ${JOB}  ckpt=${CKPT}  num_images=${NUM_IMAGES}"
    sbatch -J "${JOB}" ${SBATCH_EXTRA} sbatch_scripts/eval_wsweep.sh "${CKPT}" "${NUM_IMAGES}"
done
