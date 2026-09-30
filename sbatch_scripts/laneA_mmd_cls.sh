#!/bin/bash
# classes_per_batch variant: sub w=100 with the step's rollout confined to 8 DISTINCT
# classes (8 trajectories each) instead of the batch's ~62 classes at ~1 each.
# Everything else matches the sub100 arm, so the pair isolates class density.
set -eo pipefail
cd /scratch/gpfs/AM43/zz8976/LakonLab
LAKON_MMD_FEATURE=subspace LAKON_MMD_WEIGHT=100 LAKON_MMD_CLASSES=8 \
  ./sbatch_scripts/run_direct.sh configs/asymflow/asymflow_h_16_r8_imagenet_regft_mmd_sweep_4gpus.py 0,1,2,3 && \
./sbatch_scripts/run_direct_eval.sh configs/asymflow/asymflow_h_16_r8_imagenet_wsweep.py \
  checkpoints/asymflow_h_16_r8_imagenet_regft_mmd_sub100_cls8_4gpus/latest.pth 0,1,2,3
