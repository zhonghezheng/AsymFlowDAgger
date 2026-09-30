#!/bin/bash
# Wait for the current lane B chain to finish, then run the class-restricted CFG arm.
set -eo pipefail
cd /scratch/gpfs/AM43/zz8976/LakonLab
echo "[queue] waiting on lane B (pid 1336762)"
while kill -0 1336762 2>/dev/null; do sleep 120; done
echo "[queue] lane B finished at $(date); starting c8nb variant"
exec ./sbatch_scripts/laneB_cfg_c8nb.sh
