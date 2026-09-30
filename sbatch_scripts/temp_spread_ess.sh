#!/bin/bash
#SBATCH --job-name=ts_ess
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --partition=ailab
#SBATCH --time=01:00:00
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=zz8976@princeton.edu
set -eo pipefail
cd /scratch/gpfs/AM43/zz8976/LakonLab
mkdir -p "slurm_outputs/${SLURM_JOB_NAME}"
source .venv/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export HF_HUB_OFFLINE=1
python tools/temp_spread_ess.py
echo "done rc=$?"
