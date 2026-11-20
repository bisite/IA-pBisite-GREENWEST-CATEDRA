#!/bin/bash
#SBATCH --job-name=add_baseline
#SBATCH --partition=gpu-small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:40:00
#SBATCH --output=slurm/add_baseline_%j.out
#SBATCH --error=slurm/add_baseline_%j.err

# Añade la fila de baseline (IFN3-puro, IFN2+3-completo) a las tablas
# comparativas ya generadas. Necesita GPU real (cuml) + LD_PRELOAD para
# cargar la libcudart.so.12 del entorno conda (ver run_fig62_stacking_slurm.sh).

echo "=== Job started: $(date) ==="
echo "=== Node: $(hostname) ==="

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rapids_env

export LD_PRELOAD="${CONDA_PREFIX}/lib/libcudart.so.12"

# Submit from the repository root (sbatch scripts/revision/<script>.sh)
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

python scripts/revision/add_baseline_to_comparison.py --target all

echo "=== Job finished: $(date) ==="
