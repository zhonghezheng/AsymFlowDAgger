#!/bin/bash
# Same arm as laneA_mmd_cls8_shboth.sh (cls8, mmd_target_share='both', w=100) but in
# RAW pixel space instead of the projected subspace, so the pair isolates the feature
# space with everything else held.
#
# Expectation to check against, NOT to assume: raw measured INERT at this band --
# mmd_raw reads +-0.0000 in every training log, and tools/tsplit_snr_m.py put it at
# z~0.17 at t_split=0.875, m=256. If raw stays inert here this run reproduces
# raw100's mild FM drift (4.33 -> 4.54) rather than testing share='both' at all;
# the sub arm is the one carrying live signal. Run it to confirm that reading, not
# in expectation of a gain.
set -eo pipefail
cd /scratch/gpfs/AM43/zz8976/LakonLab
LAKON_MMD_FEATURE=raw LAKON_MMD_WEIGHT=100 LAKON_MMD_CLASSES=8 LAKON_MMD_SHARE=both \
  ./sbatch_scripts/run_direct.sh configs/asymflow/asymflow_h_16_r8_imagenet_regft_mmd_sweep_4gpus.py 0,1,2,3 && \
./sbatch_scripts/run_direct_eval.sh configs/asymflow/asymflow_h_16_r8_imagenet_wsweep.py \
  checkpoints/asymflow_h_16_r8_imagenet_regft_mmd_raw100_cls8_shboth_4gpus/latest.pth 0,1,2,3
