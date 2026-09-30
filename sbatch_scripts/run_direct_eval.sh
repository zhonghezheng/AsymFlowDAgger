#!/bin/bash
# Launch an EVAL run DIRECTLY on an interactive node (e.g. della-ani, 8x H200),
# no Slurm. The eval counterpart of run_direct.sh: drives tools/test.py against a
# checkpoint instead of tools/train.py.
#
#   ./sbatch_scripts/run_direct_eval.sh <config.py> <ckpt.pth>            # auto-pick idle GPUs
#   ./sbatch_scripts/run_direct_eval.sh <config.py> <ckpt.pth> 0,1,2,3    # pick explicitly
#   NGPU=2 ./sbatch_scripts/run_direct_eval.sh <config.py> <ckpt.pth>     # different count
#   LAKON_EVAL_NUM_IMAGES=50000 ./sbatch_scripts/run_direct_eval.sh ...   # sample count
#
# Runs in the foreground. To survive an ssh drop, wrap it:
#   nohup ./sbatch_scripts/run_direct_eval.sh <config.py> <ckpt.pth> > /dev/null 2>&1 &
# (the script tees its own logfile either way).
set -eo pipefail
PROJ=/scratch/gpfs/AM43/zz8976/LakonLab
cd "$PROJ"

CONFIG="${1:?usage: run_direct_eval.sh <config.py> <ckpt.pth> [gpu_ids]}"
CKPT="${2:?usage: run_direct_eval.sh <config.py> <ckpt.pth> [gpu_ids]}"
[ -f "$CONFIG" ] || { echo "no such config: $CONFIG" >&2; exit 1; }
[ -f "$CKPT" ]   || { echo "no such checkpoint: $CKPT" >&2; exit 1; }
NGPU="${NGPU:-4}"

# Pick GPUs: explicit arg, else the NGPU with the least memory in use.
EXPLICIT=0
if [ -n "$3" ]; then
    GPUS="$3"; EXPLICIT=1
else
    GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
           | sort -t, -k2 -n | head -n "$NGPU" | cut -d, -f1 | sort -n | paste -sd,)
fi
NSEL=$(tr ',' '\n' <<< "$GPUS" | grep -c .)
[ "$NSEL" -eq "$NGPU" ] || { echo "picked $NSEL GPUs, wanted $NGPU" >&2; exit 1; }

# Don't silently land on a GPU that already has a job on it -- sharing an H200
# with a 130GB training run would OOM both. Auto-picked ids refuse; explicitly
# passed ids only warn, on the assumption you meant it.
for g in ${GPUS//,/ }; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    if [ "$used" -gt 1024 ]; then
        if [ "$EXPLICIT" -eq 1 ]; then
            echo "WARNING: GPU $g already has ${used} MiB in use -- proceeding anyway." >&2
        else
            echo "GPU $g already has ${used} MiB in use -- refusing." >&2
            echo "Pass GPU ids explicitly to override: run_direct_eval.sh $CONFIG $CKPT <ids>" >&2
            exit 1
        fi
    fi
done

source .venv/bin/activate
export CUDA_VISIBLE_DEVICES="$GPUS"
# Node is shared: leave CPU headroom for whoever owns the other GPUs.
export OMP_NUM_THREADS=8
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + RANDOM % 20000))   # avoid colliding with the other run
export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
source "$PROJ/sbatch_scripts/wandb_env.sh"

# read by the wsweep config to size both the val dataset and the InceptionMetrics
export LAKON_EVAL_NUM_IMAGES="${LAKON_EVAL_NUM_IMAGES:-10000}"

NAME=$(basename "$CONFIG" .py)
CKPT_TAG=$(basename "$(dirname "$CKPT")")
mkdir -p direct_outputs
LOG="direct_outputs/out_log_${NAME}_${CKPT_TAG}_$(date +%Y%m%d_%H%M%S).log"

echo "host=$(hostname)  gpus=${GPUS} (${NGPU} of $(nvidia-smi -L | wc -l))  port=${MASTER_PORT}"
echo "config=${CONFIG}"
echo "ckpt=${CKPT}  num_images=${LAKON_EVAL_NUM_IMAGES}"
echo "log=${LOG}"

torchrun \
    --nnodes=1 \
    --nproc_per_node="${NGPU}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    tools/test.py "${CONFIG}" \
    --ckpt "${CKPT}" \
    --launcher pytorch \
    --diff_seed 2>&1 | tee "$LOG"

echo "done rc=${PIPESTATUS[0]}"
