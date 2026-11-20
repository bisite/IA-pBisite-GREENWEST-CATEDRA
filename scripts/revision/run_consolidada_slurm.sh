#!/bin/bash
#SBATCH --job-name=consolidada
#SBATCH --partition=gpu-small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/consolidada_%j.out
#SBATCH --error=slurm/consolidada_%j.err

# Initialize Lmod module system for SLURM batch jobs
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
else
  echo "[modules] WARNING: module command not found — skipping module load"
fi

echo "=== Job started: $(date) ==="
echo "=== Node: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'nvidia-smi failed') ==="

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env_cuml

# Submit from the repository root (sbatch scripts/revision/<script>.sh)
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Re-genera las 4 figuras Consolidado_IFN{2,3}_{c4,carbono_bruto4}_clean.pdf
# (+ las _paper.pdf con histograma, no usadas en el artículo) con los paneles
# Base/Stack/Best completos -- los modelos de stacking necesitan GPU/cuML.
python scripts/revision/generar_figura_consolidada.py

echo "=== Job finished: $(date) ==="
