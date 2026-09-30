#!/bin/bash
# Same arm as laneA_mmd_cls8_shboth_raw.sh (raw, cls8, mmd_target_share='both') at
# mmd_weight=1000 instead of 100, so the pair isolates the weight.
#
# What this is testing: at w=100 the raw term contributed loss_mmd ~0.0006 against
# loss_diffusion ~0.060 -- about 1%, with mmd_raw reading +-0.0000 in every logged
# line. At w=1000 it becomes ~10% of the objective, so the term finally has enough
# weight to move training.
#
# The risk, stated plainly: tools/tsplit_snr_m.py put raw at z~0.17 at t_split=0.875,
# m=256, i.e. AT its noise floor. Multiplying a noise-floor statistic by 10 amplifies
# gradient noise, not signal. If this degrades faster than raw100 did (+0.21 over
# 5000 iters), that is the expected outcome, not a surprise -- and it would confirm
# the band location rather than the weight is what limits raw.
set -eo pipefail
cd /scratch/gpfs/AM43/zz8976/LakonLab
LAKON_MMD_FEATURE=raw LAKON_MMD_WEIGHT=1000 LAKON_MMD_CLASSES=8 LAKON_MMD_SHARE=both \
  ./sbatch_scripts/run_direct.sh configs/asymflow/asymflow_h_16_r8_imagenet_regft_mmd_sweep_4gpus.py 0,1,2,3 && \
./sbatch_scripts/run_direct_eval.sh configs/asymflow/asymflow_h_16_r8_imagenet_wsweep.py \
  checkpoints/asymflow_h_16_r8_imagenet_regft_mmd_raw1000_cls8_shboth_4gpus/latest.pth 0,1,2,3
