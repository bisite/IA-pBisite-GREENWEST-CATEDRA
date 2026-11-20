#!/bin/bash
#SBATCH --job-name=subsample_c4
#SBATCH --partition=gpu-small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/subsample_c4_%j.out
#SBATCH --error=slurm/subsample_c4_%j.err

# Initialize Lmod module system for SLURM batch jobs
# /etc/profile.d/lmod.sh has a SLURM guard (returns early if SLURM_NODELIST is set)
# so we must source the OpenHPC Lmod init directly.
export LMOD_CMD=/opt/ohpc/admin/lmod/lmod/libexec/lmod
export LMOD_SETTARG_CMD=":"
export LMOD_FULL_SETTARG_SUPPORT=no
export LMOD_COLORIZE=no
export LMOD_PREPEND_BLOCK=normal
export MODULEPATH=/opt/ohpc/pub/modulefiles
. /opt/ohpc/admin/lmod/lmod/init/bash

# Verify module system is available
if type module &>/dev/null; then
  module purge
  module load cuda/12.9.1 gnu13/13.2.0
  echo "[modules] loaded cuda/12.9.1 gnu13/13.2.0"
else
  echo "[modules] WARNING: module command not found — skipping module load"
  echo "[modules] The conda environment (rapids_env) must provide its own CUDA toolkit."
fi

echo "=== Job started: $(date) ==="
echo "=== Node: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'nvidia-smi failed') ==="

# Activate conda env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rapids_env

# Submit from the repository root (sbatch scripts/revision/<script>.sh)
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Run subsampling experiment for c4
python scripts/revision/subsampling_experiment.py --target c4 --n-reps 5

echo "=== Job finished: $(date) ==="