#!/bin/bash
# Launch a training run DIRECTLY on an interactive node (e.g. della-ani, 8x H200),
# no Slurm. Uses 4 GPUs; by default the 4 most idle ones, so it coexists with a
# run already occupying the other half of the node.
#
#   ./sbatch_scripts/run_direct.sh <config.py>              # auto-pick 4 idle GPUs
#   ./sbatch_scripts/run_direct.sh <config.py> 4,5,6,7      # pick explicitly
#   NGPU=2 ./sbatch_scripts/run_direct.sh <config.py>       # use a different count
#
# Runs in the foreground. To survive an ssh drop, wrap it:
#   nohup ./sbatch_scripts/run_direct.sh <config.py> > /dev/null 2>&1 &
# (the script tees its own logfile either way).
set -eo pipefail
PROJ=/scratch/gpfs/AM43/zz8976/LakonLab
cd "$PROJ"

CONFIG="${1:?usage: run_direct.sh <config.py> [gpu_ids]}"
[ -f "$CONFIG" ] || { echo "no such config: $CONFIG" >&2; exit 1; }
NGPU="${NGPU:-4}"

# Pick GPUs: explicit arg, else the NGPU with the least memory in use.
EXPLICIT=0
if [ -n "$2" ]; then
    GPUS="$2"; EXPLICIT=1
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
            echo "Pass GPU ids explicitly to override: run_direct.sh $CONFIG <ids>" >&2
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

NAME=$(basename "$CONFIG" .py)
mkdir -p direct_outputs
LOG="direct_outputs/out_log_${NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "host=$(hostname)  gpus=${GPUS} (${NGPU} of $(nvidia-smi -L | wc -l))  port=${MASTER_PORT}"
echo "config=${CONFIG}"
echo "log=${LOG}"

torchrun \
    --nnodes=1 \
    --nproc_per_node="${NGPU}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    tools/train.py "${CONFIG}" \
    --launcher pytorch \
    --diff_seed 2>&1 | tee "$LOG"

echo "done rc=${PIPESTATUS[0]}"
