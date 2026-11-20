#!/bin/bash
#SBATCH --job-name=fig62_stacking
#SBATCH --partition=gpu-small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=slurm/fig62_stacking_%j.out
#SBATCH --error=slurm/fig62_stacking_%j.err

# Genera las figuras Stack5_MLP para Fig 6.2. Random_Forest (uno
# de los 6 modelos base del stack) se entrenó con cuml real (GPU) -> necesita
# GPU real para cargar, y ademas necesita LD_PRELOAD (cuda-pathfinder
# encuentra la libcudart.so.13 de torch antes
# que la libcudart.so.12 correcta del propio entorno conda; LD_PRELOAD fuerza
# cual se carga primero sin tocar ningun paquete instalado).

echo "=== Job started: $(date) ==="
echo "=== Node: $(hostname) ==="

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rapids_env

export LD_PRELOAD="${CONDA_PREFIX}/lib/libcudart.so.12"

# Submit from the repository root (sbatch scripts/revision/<script>.sh)
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

python scripts/revision/generar_figuras_publicacion.py --models stacking

echo "=== Job finished: $(date) ==="
