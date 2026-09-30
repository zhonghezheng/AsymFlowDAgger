#!/bin/bash
# Lane B: CFG-gap w=10 frac=0 complement=full, band confined to 8 DISTINCT classes,
# but the NULL bank drawn from the ENTIRE datapool (band_null_from_batch=0).
#
# This splits the two knobs that cmff10_f0_c8nb changed together. That run restricted
# both the band's classes AND the null bank to the same 8 classes, and came out worse
# than the unrestricted control (5.268 vs 5.103 at iter 4500, still widening).
#
# The reading that motivates this variant: the model's own v_uncond is trained
# against the FULL 1000-class marginal, so an empirical v*_uncond drawn from only 8
# classes changes what the TARGET gap means without changing what the MODEL's gap
# means -- the two sides then disagree by construction. Keeping the null bank
# dataset-wide restores that agreement while still giving each class ~2-3
# trajectories instead of ~1.
#
# Controls: cmff10_f0 (no restriction, 5.236 @5000) and cmff10_f0_c8nb (both
# restricted, 5.268 @4500 when stopped).
set -eo pipefail
cd /scratch/gpfs/AM43/zz8976/LakonLab
LAKON_CFG_GAP_WEIGHT=10 LAKON_BAND_FRAC=0 LAKON_CFG_CM=full \
LAKON_BAND_NCLS=8 LAKON_BAND_NULLBATCH=0 \
  ./sbatch_scripts/run_direct.sh configs/asymflow/asymflow_h_16_r8_imagenet_dagger_bankfull_cfg_4gpus.py 4,5,6,7 && \
./sbatch_scripts/run_direct_eval.sh configs/asymflow/asymflow_h_16_r8_imagenet_wsweep.py \
  checkpoints/asymflow_h_16_r8_imagenet_dagger_bankfull_cmff10_f0_c8_4gpus/latest.pth 4,5,6,7
