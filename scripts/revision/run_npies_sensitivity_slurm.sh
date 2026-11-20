#!/bin/bash
#SBATCH --job-name=npies_sensitivity
#SBATCH --partition=gpu-small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:30:00
#SBATCH --output=slurm/npies_sensitivity_%j.out
#SBATCH --error=slurm/npies_sensitivity_%j.err

# Reentrena los 9 modelos globales (IFN2+IFN3) sin npies*/NPies* para
# cuantificar cuanto R2 se retiene sin esa variable. Necesita GPU real
# (cuml RF/SVR, XGBoost gpu_hist, CatBoost GPU, MLP cuda) + LD_PRELOAD para
# que el resultado sea comparable al modelo global original (LD_PRELOAD
# fuerza la libcudart.so.12 del entorno; ver run_fig62_stacking_slurm.sh).

echo "=== Job started: $(date) ==="
echo "=== Node: $(hostname) ==="

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rapids_env

export LD_PRELOAD="${CONDA_PREFIX}/lib/libcudart.so.12"

# Submit from the repository root (sbatch scripts/revision/<script>.sh)
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

python scripts/revision/npies_sensitivity_experiment.py --target all --with-control

echo "=== Job finished: $(date) ==="
