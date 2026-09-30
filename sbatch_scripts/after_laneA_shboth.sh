#!/bin/bash
# Wait for the sub cls8/shboth lane to finish, then run the raw counterpart on its GPUs.
set -eo pipefail
cd /scratch/gpfs/AM43/zz8976/LakonLab
echo "[queue] waiting on laneA cls8_shboth (pid 1669059)"
while kill -0 1669059 2>/dev/null; do sleep 120; done
echo "[queue] sub lane finished at $(date); starting raw counterpart"
exec ./sbatch_scripts/laneA_mmd_cls8_shboth_raw.sh
