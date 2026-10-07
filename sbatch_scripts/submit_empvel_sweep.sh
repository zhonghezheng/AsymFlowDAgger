#!/bin/bash
# Fan the empvel w-sweep out over ailab nodes: one eval_empvel_wsweep.sh job per v*
# window (each sweeps every w), so the default four windows run in parallel instead of
# ~8h back to back. Run it on the login node -- it only calls sbatch.
#
#   sbatch_scripts/submit_empvel_sweep.sh [checkpoint.pth | base] [num_images]
#
# LAKON_EV_WINDOWS picks the windows (default off,0.94-1,0.88-1,0.88-0.94); every other
# LAKON_* knob is exported through to each job (sbatch's default --export=ALL). Jobs are
# named ev_<run>_<window>, logs under slurm_outputs/<job name>/.
set -eo pipefail
cd "$(dirname "$0")/.."

CKPT="${1:-base}"
NUM_IMAGES="${2:-10000}"
if [ "$CKPT" = base ]; then
    TAG=base
elif [ -f "$CKPT" ]; then
    TAG="$(basename "$(dirname "$CKPT")")"
    TAG="${TAG#asymflow_h_16_r8_imagenet_}_$(basename "$CKPT" .pth)"
else
    echo "no such checkpoint: $CKPT"; exit 1
fi
IFS=',' read -ra WINDOWS <<< "${LAKON_EV_WINDOWS:-off,0.94-1,0.88-1,0.88-0.94}"

for W in "${WINDOWS[@]}"; do
    JOB="ev_${TAG}_${W}"
    mkdir -p "slurm_outputs/${JOB}"
    LAKON_EV_WINDOWS="$W" sbatch -J "$JOB" sbatch_scripts/eval_empvel_wsweep.sh "$CKPT" "$NUM_IMAGES"
done
