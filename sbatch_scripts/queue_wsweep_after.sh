#!/bin/bash
# Wait for a running training job to finish, then wsweep the checkpoint it left.
#
#   ./sbatch_scripts/queue_wsweep_after.sh <train_pid> <run_name> <gpu_ids>
#
# Waits on the PID of an already-launched run_direct.sh (so it attaches to a run
# that is ALREADY going -- unlike run_onpolicy_align.sh, which starts its own).
# The sweep is SKIPPED if training left no checkpoint, so a crashed or killed run
# cannot silently produce a sweep of stale weights.
set -eo pipefail
PROJ=/scratch/gpfs/AM43/zz8976/LakonLab
cd "$PROJ"

PID="${1:?usage: queue_wsweep_after.sh <train_pid> <run_name> <gpu_ids>}"
NAME="${2:?missing run_name}"
GPUS="${3:?missing gpu_ids}"
CKPT="checkpoints/${NAME}/latest.pth"

echo "[queue] waiting on training pid=${PID} (${NAME})"
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
echo "[queue] training pid=${PID} exited at $(date)"

if [ ! -f "$CKPT" ]; then
    echo "[queue] no checkpoint at ${CKPT} -- training did not finish; skipping sweep." >&2
    exit 1
fi
echo "[queue] sweeping ${CKPT} -> $(readlink -f "$CKPT" | xargs basename)"
exec ./sbatch_scripts/run_direct_eval.sh \
    configs/asymflow/asymflow_h_16_r8_imagenet_wsweep.py "$CKPT" "$GPUS"
