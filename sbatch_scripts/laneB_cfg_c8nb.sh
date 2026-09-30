#!/bin/bash
# Lane B: CFG-gap w=10 frac=0 complement=full, with the band confined to 8 DISTINCT
# classes AND the empirical v*_uncond drawn from those same 8 classes.
#
# Both halves matter together. band_classes_per_batch=8 gives ~2-3 trajectories per
# class instead of ~1 over 1000. band_null_from_batch=1 then makes the null bank a
# draw from that same restricted set, so the TARGET gap v*_cond - v*_uncond compares
# a class against the mixture it actually sits in. With the null left unrestricted
# the two halves of the target gap refer to different universes -- the conditional to
# 8 classes, the unconditional to all 1000 -- and the model is asked to match a
# difference taken across that mismatch.
#
# Control: cmff10_f0 (same weight, frac and complement mode, unrestricted classes).
set -eo pipefail
cd /scratch/gpfs/AM43/zz8976/LakonLab
LAKON_CFG_GAP_WEIGHT=10 LAKON_BAND_FRAC=0 LAKON_CFG_CM=full \
LAKON_BAND_NCLS=8 LAKON_BAND_NULLBATCH=1 \
  ./sbatch_scripts/run_direct.sh configs/asymflow/asymflow_h_16_r8_imagenet_dagger_bankfull_cfg_4gpus.py 4,5,6,7 && \
./sbatch_scripts/run_direct_eval.sh configs/asymflow/asymflow_h_16_r8_imagenet_wsweep.py \
  checkpoints/asymflow_h_16_r8_imagenet_dagger_bankfull_cmff10_f0_c8nb_4gpus/latest.pth 4,5,6,7
