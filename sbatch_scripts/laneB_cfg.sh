#!/bin/bash
# Lane B (GPUs 4-7): CFG-gap w=10 frac=0, complement full then project -- each
# train immediately followed by its wsweep, chained with &&.
set -eo pipefail
cd /scratch/gpfs/AM43/zz8976/LakonLab
LAKON_CFG_GAP_WEIGHT=10 LAKON_BAND_FRAC=0 LAKON_CFG_CM=full    ./sbatch_scripts/run_direct.sh configs/asymflow/asymflow_h_16_r8_imagenet_dagger_bankfull_cfg_4gpus.py 4,5,6,7 && \
./sbatch_scripts/run_direct_eval.sh configs/asymflow/asymflow_h_16_r8_imagenet_wsweep.py checkpoints/asymflow_h_16_r8_imagenet_dagger_bankfull_cmff10_f0_4gpus/latest.pth 4,5,6,7 && \
LAKON_CFG_GAP_WEIGHT=10 LAKON_BAND_FRAC=0 LAKON_CFG_CM=project ./sbatch_scripts/run_direct.sh configs/asymflow/asymflow_h_16_r8_imagenet_dagger_bankfull_cfg_4gpus.py 4,5,6,7 && \
./sbatch_scripts/run_direct_eval.sh configs/asymflow/asymflow_h_16_r8_imagenet_wsweep.py checkpoints/asymflow_h_16_r8_imagenet_dagger_bankfull_cmpf10_f0_4gpus/latest.pth 4,5,6,7
