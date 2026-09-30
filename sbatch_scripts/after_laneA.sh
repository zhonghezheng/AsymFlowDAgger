#!/bin/bash
# Wait for lane A's chain to finish, then run the classes_per_batch variant on its GPUs.
set -eo pipefail
cd /scratch/gpfs/AM43/zz8976/LakonLab
echo "[queue] waiting on lane A (pid 1133630)"
while kill -0 1133630 2>/dev/null; do sleep 120; done
echo "[queue] lane A finished at $(date); starting classes_per_batch variant"
exec ./sbatch_scripts/laneA_mmd_cls.sh
