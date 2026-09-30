#!/bin/bash
# cls8 re-run with mmd_target_share='both': the target images AND the interpolation
# noise are drawn ONCE per iteration and reused across the band's steps, so the
# target set is a set of straight-line trajectories rather than an independent draw
# per step (d-flow's share='both').
#
# Motivation: the cls8 run under share='none' degraded exactly like sub100
# (4.32 -> 4.84 by iter 2000), and sub100 itself was BETTER (4.29) under the older
# cached-target code than under fresh-per-step draws (6.24). Sharing restores the
# between-step correlation that the cache used to provide, without going back to a
# stale FIFO.
set -eo pipefail
cd /scratch/gpfs/AM43/zz8976/LakonLab
LAKON_MMD_FEATURE=subspace LAKON_MMD_WEIGHT=100 LAKON_MMD_CLASSES=8 LAKON_MMD_SHARE=both \
  ./sbatch_scripts/run_direct.sh configs/asymflow/asymflow_h_16_r8_imagenet_regft_mmd_sweep_4gpus.py 0,1,2,3 && \
./sbatch_scripts/run_direct_eval.sh configs/asymflow/asymflow_h_16_r8_imagenet_wsweep.py \
  checkpoints/asymflow_h_16_r8_imagenet_regft_mmd_sub100_cls8_shboth_4gpus/latest.pth 0,1,2,3
