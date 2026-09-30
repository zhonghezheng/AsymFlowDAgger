#!/bin/bash
# Lane B: CFG-gap w=10 frac=0 complement=full, UNRESTRICTED classes (the cmff10_f0
# baseline), with a softmax temperature T=100 on the NULL bank only.
#
# Why: x0_hat is the exact Bayes posterior over the empirical prior, but in 2048-d
# feature space it saturates -- measured ESS ~1.1 of 2048 at sigma=0.92, i.e.
# v*_uncond is a NEAREST-NEIGHBOUR lookup, not a posterior mean, so the CFG target
# gap has been a difference of two single-image lookups all along. d2 -> d2/100
# restores a genuine average: measured ESS ~1022, with ||v*_c - v*_u|| at 0.94x its
# T=1 value and v*_uncond still ~1.0 bank-mean-norms from the plain bank mean (T>=1000
# degenerates to the bank mean itself).
#
# The uncond_bank SAMPLING is deliberately unchanged: still an independent
# null_bank_size=2048 draw from the whole dataset per trajectory. Only the weighting
# changes. The conditional bank keeps T=1 -- smoothing it would blur the class
# identity the gap is meant to isolate.
#
# Control: cmff10_f0, identical but T=1 (5.236 @5000, best sweep 5.13 @w=2.4).
set -eo pipefail
cd /scratch/gpfs/AM43/zz8976/LakonLab
LAKON_CFG_GAP_WEIGHT=10 LAKON_BAND_FRAC=0 LAKON_CFG_CM=full LAKON_NULL_TEMP=100 \
  ./sbatch_scripts/run_direct.sh configs/asymflow/asymflow_h_16_r8_imagenet_dagger_bankfull_cfg_4gpus.py 4,5,6,7 && \
./sbatch_scripts/run_direct_eval.sh configs/asymflow/asymflow_h_16_r8_imagenet_wsweep.py \
  checkpoints/asymflow_h_16_r8_imagenet_dagger_bankfull_cmff10_f0_T100_4gpus/latest.pth 4,5,6,7
