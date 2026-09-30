#!/bin/bash
# Sync offline wandb runs to the cloud. Run this from a LOGIN NODE (compute nodes
# on della can't reach api.wandb.ai), after training finishes.
#
#   ./sbatch_scripts/sync_wandb.sh                  # sync every un-synced run
#   ./sbatch_scripts/sync_wandb.sh offline-run-20260816_184712-qp8cc7fn
#   ./sbatch_scripts/sync_wandb.sh wandb/offline-run-*bank8*
#
# Sourcing wandb_env.sh matters: training staged artifact files under
# $PROJ/.wandb_data, and the sender reads them back from there at sync time.
set -eo pipefail
PROJ=/scratch/gpfs/AM43/zz8976/LakonLab
cd "$PROJ"

source .venv/bin/activate
source sbatch_scripts/wandb_env.sh

# wandb sync must talk to the API, so clear the offline flag inherited from the
# training environment.
unset WANDB_MODE

if [ "$#" -gt 0 ]; then
    wandb sync "$@"
else
    wandb sync --sync-all
fi
