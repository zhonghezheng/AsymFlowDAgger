#!/bin/bash
# Wait for a process to exit (a training run, or a queue_wsweep_after_direct.sh chained
# behind one), then run the inference-time MMD-guidance alpha sweep
# (configs/asymflow/asymflow_h_16_r8_imagenet_mmdguide.py) on all NGPU GPUs.
#
#   nohup ./sbatch_scripts/queue_mmdguide_after_direct.sh <pid> [ckpt] > /dev/null 2>&1 &
#
# No ckpt (or '') -> the released base weights. Env knobs pass through to the config
# (LAKON_MMDG_ALPHA, LAKON_MMDG_WINDOW, ..., LAKON_EVAL_NUM_IMAGES).
# Progress goes to direct_outputs/queue_mmdguide_<tag>.log; the sweep itself to
# direct_outputs/out_log_asymflow_h_16_r8_imagenet_mmdguide_<tag>_<timestamp>.log.
set -eo pipefail
cd "$(dirname "$0")/.."

PID="${1:?usage: queue_mmdguide_after_direct.sh <pid> [ckpt]}"
CKPT="${2:-}"
NGPU="${NGPU:-8}"
CONFIG=configs/asymflow/asymflow_h_16_r8_imagenet_mmdguide.py
TAG=$([ -n "$CKPT" ] && basename "$(dirname "$CKPT")" || echo base)
mkdir -p direct_outputs
exec >> "direct_outputs/queue_mmdguide_${TAG}.log" 2>&1

echo "[queue] $(date) waiting on pid=${PID}, then mmdguide sweep of ${CKPT:-base weights}"
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
echo "[queue] $(date) pid=${PID} exited"
if [ -n "$CKPT" ] && [ ! -f "$CKPT" ]; then echo "[queue] no such checkpoint: ${CKPT}"; exit 1; fi

# let the previous job release the GPUs (at most 10 min) before claiming them
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
export LAKON_EVAL_NUM_IMAGES="${LAKON_EVAL_NUM_IMAGES:-10000}"

LOG="direct_outputs/out_log_$(basename "$CONFIG" .py)_${TAG}_$(date +%Y%m%d_%H%M%S).log"
echo "[queue] $(date) sweeping alphas=${LAKON_MMDG_ALPHA:-<config default>} on GPUs ${GPUS}" \
     "(${LAKON_EVAL_NUM_IMAGES} images) -> ${LOG}"
rc=0
torchrun --nnodes=1 --nproc_per_node="$NGPU" \
    --master_addr=127.0.0.1 --master_port=$((10000 + RANDOM % 20000)) \
    tools/test.py "$CONFIG" ${CKPT:+--ckpt "$CKPT"} --launcher pytorch --diff_seed \
    > "$LOG" 2>&1 || rc=$?
echo "[queue] $(date) sweep done rc=${rc}"
grep -h "_fid = \|_mmdg_norm_ratio = \|_mmdg_mmd = " "$LOG" || true
exit "$rc"
