#!/bin/bash
# Lane A (GPUs 0-3): MMD raw w=100, then MMD sub w=100 -- each train immediately
# followed by its wsweep. Chained with && so a failed stage stops the lane rather
# than sweeping a checkpoint that was never written.
set -eo pipefail
cd /scratch/gpfs/AM43/zz8976/LakonLab
LAKON_MMD_FEATURE=raw      LAKON_MMD_WEIGHT=100 ./sbatch_scripts/run_direct.sh configs/asymflow/asymflow_h_16_r8_imagenet_regft_mmd_sweep_4gpus.py 0,1,2,3 && \
./sbatch_scripts/run_direct_eval.sh configs/asymflow/asymflow_h_16_r8_imagenet_wsweep.py checkpoints/asymflow_h_16_r8_imagenet_regft_mmd_raw100_4gpus/latest.pth 0,1,2,3 && \
LAKON_MMD_FEATURE=subspace LAKON_MMD_WEIGHT=100 ./sbatch_scripts/run_direct.sh configs/asymflow/asymflow_h_16_r8_imagenet_regft_mmd_sweep_4gpus.py 0,1,2,3 && \
./sbatch_scripts/run_direct_eval.sh configs/asymflow/asymflow_h_16_r8_imagenet_wsweep.py checkpoints/asymflow_h_16_r8_imagenet_regft_mmd_sub100_4gpus/latest.pth 0,1,2,3
