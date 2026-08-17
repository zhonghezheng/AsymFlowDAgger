#!/bin/bash
#SBATCH --job-name=train_asym_test
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=ailab
#SBATCH --time=01:00:00
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=zz8976@princeton.edu

# --- Smoke test: DAGGER AsymFlow finetuning end-to-end on a single node ---
# Verifies the pipeline loads the released checkpoint and runs on-path +
# rollout/expert + buffer steps without crashing. NOT a real training run.
set -eo pipefail

PROJ=/scratch/gpfs/AM43/zz8976/LakonLab
cd "$PROJ"
mkdir -p "slurm_outputs/${SLURM_JOB_NAME}"

# environment
source .venv/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + RANDOM % 20000))   # avoid rendezvous port clashes

# use every GPU the scheduler gave us
NGPU=$(nvidia-smi -L | wc -l)
CONFIG=configs/asymflow/asymflow_h_16_r8_imagenet_dagger_smoke.py

echo "host=$(hostname)  gpus=${NGPU}  master_port=${MASTER_PORT}"
echo "config=${CONFIG}"
nvidia-smi -L

# --launcher pytorch = DDP via torchrun; --diff_seed = per-rank seeds;
# --no-validate = skip the (heavy, Inception-FID) eval for the smoke test.
torchrun \
    --nnodes=1 \
    --nproc_per_node="${NGPU}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    tools/train.py "${CONFIG}" \
    --launcher pytorch \
    --diff_seed \
    --no-validate

echo "done rc=$?"
