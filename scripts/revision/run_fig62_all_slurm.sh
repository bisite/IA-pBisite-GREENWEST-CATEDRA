#!/bin/bash
#SBATCH --job-name=fig62_all
#SBATCH --partition=gpu-small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=slurm/fig62_all_%j.out
#SBATCH --error=slurm/fig62_all_%j.err

# Regenera los 4 paneles de Fig 6.2 (base + stacking) con el test de sesgo
# adicional sobre el test set COMPLETO (sin recortar al rango [0,max_escala]
# mostrado en la figura), ya que ~0.5% de outliers extremos cambia la
# conclusion del test de sesgo. patch_cuml_setstate_verbose()
# importa cuml incondicionalmente al cargar el modulo,
# asi que hace falta GPU real + LD_PRELOAD incluso para los modelos base
# (LightGBM/CatBoost), que en si mismos no usan cuml.

echo "=== Job started: $(date) ==="
echo "=== Node: $(hostname) ==="

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rapids_env

export LD_PRELOAD="${CONDA_PREFIX}/lib/libcudart.so.12"

# Submit from the repository root (sbatch scripts/revision/<script>.sh)
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

python scripts/revision/generar_figuras_publicacion.py --models all

echo "=== Job finished: $(date) ==="
