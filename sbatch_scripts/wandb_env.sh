#!/bin/bash
# Shared wandb environment. Sourced by the training sbatch scripts AND by
# sync_wandb.sh on the login node -- both sides must agree on these paths, since
# artifact staging written during training is read back at sync time.
#
# Everything lives under $PROJ (scratch). The defaults put run data, the cache
# and the artifact staging dir under /home (~/.cache/wandb), which is quota-tight
# here: a full run died in WandbLoggerHook.after_run with
#   OSError: [Errno 122] Disk quota exceeded  (shutil.copyfile -> staging_path)
# after training had already completed. Keeping these on scratch avoids that
# while leaving `wandb sync` fully functional.
: "${PROJ:=/scratch/gpfs/AM43/zz8976/LakonLab}"
export WANDB_DIR="$PROJ"
export WANDB_CACHE_DIR="$PROJ/.wandb_cache"
export WANDB_DATA_DIR="$PROJ/.wandb_data"
export WANDB_ARTIFACT_DIR="$PROJ/.wandb_artifacts"
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$WANDB_ARTIFACT_DIR"
