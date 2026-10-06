#!/bin/bash
# Direct (non-Slurm) counterpart of queue_wsweep_after.sh, for a node where the repo
# is not at the cluster PROJ path: wait for a training run to exit, then wsweep its
# FINAL checkpoint with run_direct_eval.sh's settings, logging into direct_outputs/.
#
#   nohup ./sbatch_scripts/queue_wsweep_after_direct.sh <train_pid> <run_name> [final_iter] \
#       > /dev/null 2>&1 &
#
# Requires checkpoints/<run_name>/iter_<final_iter>.pth (default 5000), not merely
# latest.pth: a run that crashed after its iter-2500 save still leaves latest.pth, and
# sweeping that would report a half-trained model as the run's result.
# Progress goes to direct_outputs/queue_wsweep_<run_name>.log; the sweep itself to
# direct_outputs/out_log_asymflow_h_16_r8_imagenet_wsweep_<run_name>_<timestamp>.log.
set -eo pipefail
cd "$(dirname "$0")/.."

PID="${1:?usage: queue_wsweep_after_direct.sh <train_pid> <run_name> [final_iter]}"
NAME="${2:?missing run_name}"
FINAL="${3:-5000}"
NGPU="${NGPU:-8}"
CKPT="checkpoints/${NAME}/iter_${FINAL}.pth"
CONFIG=configs/asymflow/asymflow_h_16_r8_imagenet_wsweep.py
mkdir -p direct_outputs
exec >> "direct_outputs/queue_wsweep_${NAME}.log" 2>&1

echo "[queue] $(date) waiting on training pid=${PID} (${NAME}), then ${CKPT}"
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
echo "[queue] $(date) training pid=${PID} exited"
if [ ! -f "$CKPT" ]; then
    echo "[queue] no ${CKPT} -- the run did not finish; skipping the sweep."
    exit 1
fi

# let the training processes release the GPUs (at most 10 min) before claiming them
GPUS=$(seq -s, 0 $((NGPU - 1)))
for _ in $(seq 60); do
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPUS" \
           | awk '$1 > 1024' | wc -l)
    [ "$busy" -eq 0 ] && break
    sleep 10
done

source .venv/bin/activate
export CUDA_VISIBLE_DEVICES="$GPUS"
export OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
# read by the wsweep config to size both the val dataset and the InceptionMetrics
export LAKON_EVAL_NUM_IMAGES="${LAKON_EVAL_NUM_IMAGES:-10000}"

LOG="direct_outputs/out_log_$(basename "$CONFIG" .py)_${NAME}_$(date +%Y%m%d_%H%M%S).log"
echo "[queue] $(date) sweeping ${CKPT} on GPUs ${GPUS} (${LAKON_EVAL_NUM_IMAGES} images) -> ${LOG}"
rc=0
torchrun --nnodes=1 --nproc_per_node="$NGPU" \
    --master_addr=127.0.0.1 --master_port=$((10000 + RANDOM % 20000)) \
    tools/test.py "$CONFIG" --ckpt "$CKPT" --launcher pytorch --diff_seed > "$LOG" 2>&1 || rc=$?
echo "[queue] $(date) sweep done rc=${rc}"
exit "$rc"
