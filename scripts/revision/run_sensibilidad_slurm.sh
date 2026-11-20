#!/bin/bash
#SBATCH --job-name=sensibilidad
#SBATCH --partition=gpu-small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/sensibilidad_%j.out
#SBATCH --error=slurm/sensibilidad_%j.err

export LMOD_CMD=/opt/ohpc/admin/lmod/lmod/libexec/lmod
export LMOD_SETTARG_CMD=":"
export LMOD_FULL_SETTARG_SUPPORT=no
export LMOD_COLORIZE=no
export LMOD_PREPEND_BLOCK=normal
export MODULEPATH=/opt/ohpc/pub/modulefiles
. /opt/ohpc/admin/lmod/lmod/init/bash

if type module &>/dev/null; then
  module purge
  module load cuda/12.9.1 gnu13/13.2.0
  echo "[modules] loaded cuda/12.9.1 gnu13/13.2.0"
fi

echo "=== Job started: $(date) ==="
echo "=== Node: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'nvidia-smi failed') ==="

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env_cuml

# Submit from the repository root (sbatch scripts/revision/<script>.sh)
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

python scripts/revision/generar_sensibilidad_escenarios.py

echo "=== Job finished: $(date) ==="
