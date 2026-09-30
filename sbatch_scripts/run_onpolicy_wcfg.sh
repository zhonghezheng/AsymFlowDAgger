#!/bin/bash
# Train ONE onpolicy w_cfg weight, then immediately wsweep the checkpoint it produced.
# Login/interactive-node runner (no Slurm), the train+eval counterpart of
# run_direct.sh -> run_direct_eval.sh.
#
#   ./sbatch_scripts/run_onpolicy_wcfg.sh <w_cfg> [gpu_ids]
#   ./sbatch_scripts/run_onpolicy_wcfg.sh 10 0,1,2,3
#
# Runs in the foreground; to survive an ssh drop:
#   nohup ./sbatch_scripts/run_onpolicy_wcfg.sh 10 0,1,2,3 > /dev/null 2>&1 &
# Both stages tee their own logs under direct_outputs/.
#
# The eval stage is SKIPPED if training does not produce a checkpoint, so a failed
# or killed run does not silently report a stale sweep.
set -eo pipefail
PROJ=/scratch/gpfs/AM43/zz8976/LakonLab
cd "$PROJ"

W="${1:?usage: run_onpolicy_wcfg.sh <w_cfg> [gpu_ids]}"
GPUS="$2"

export LAKON_CFG_GAP_WEIGHT="$W"
CONFIG=configs/asymflow/asymflow_h_16_r8_imagenet_onpolicy_wcfg_4gpus.py

# Ask the CONFIG for the run name rather than reimplementing its tag rule here --
# the tag is the raw weight string and also folds in LAKON_CFG_ALIGN_MODE, so any
# local copy of that logic would drift the moment the config changes.
source .venv/bin/activate
NAME=$(python -c "
from mmcv import Config
print(Config.fromfile('$CONFIG').name)")
[ -n "$NAME" ] || { echo "could not resolve the run name from $CONFIG" >&2; exit 1; }
CKPT="checkpoints/${NAME}/latest.pth"

echo "=== [1/2] TRAIN  w_cfg=${W}  name=${NAME} ==="
./sbatch_scripts/run_direct.sh "$CONFIG" $GPUS

if [ ! -f "$CKPT" ]; then
    echo "!! training produced no checkpoint at ${CKPT} -- skipping the sweep." >&2
    exit 1
fi

echo "=== [2/2] WSWEEP ${CKPT} ==="
./sbatch_scripts/run_direct_eval.sh \
    configs/asymflow/asymflow_h_16_r8_imagenet_wsweep.py "$CKPT" $GPUS

echo "=== done: ${NAME} ==="
