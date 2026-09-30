#!/bin/bash
#SBATCH --job-name=rollout_grad
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=ailab
#SBATCH --time=00:30:00
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out
set -eo pipefail
cd /scratch/gpfs/AM43/zz8976/LakonLab
mkdir -p "slurm_outputs/${SLURM_JOB_NAME}"
source .venv/bin/activate
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8
python tools/check_rollout_grad.py; python tools/kernel_compare.py
echo "done rc=$?"
