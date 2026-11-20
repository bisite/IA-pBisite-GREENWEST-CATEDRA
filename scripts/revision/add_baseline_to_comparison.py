# -*- coding: utf-8 -*-
"""
Añade la fila de baseline (IFN3-puro y IFN2+3-completo) a las tablas
comparativas del experimento de subsampling cuando estas se generaron sin
ella (la evaluación del baseline requiere GPU real para cargar los modelos
cuml).

No repite el experimento de subsampling (5 repeticiones x 9 modelos, varias
horas) — reutiliza los summary CSV ya calculados y solo recalcula la parte
de baseline, que es rápida (joblib.load + predict de los modelos ya
entrenados).

Uso (requiere GPU real para cuml -> vía SLURM, ver run_add_baseline_slurm.sh):
    python add_baseline_to_comparison.py --target c4
    python add_baseline_to_comparison.py --target carbono_bruto4
    python add_baseline_to_comparison.py --target all
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from subsampling_experiment import (  # noqa: E402
    OUTPUT_ROOT, build_comparison_table,
    PyTorchMLPDropout, MLPRegressorTorch, cast_to_str, make_ohe,  # noqa: F401
    # Re-expuestas aquí porque los training_context.joblib se pickle-aron
    # referenciando __main__.<nombre> (el script de entrenamiento original
    # se ejecutó como programa principal) — al correr ESTE script como
    # __main__, estos nombres tienen que vivir en su propio espacio de
    # nombres, no solo en el de subsampling_experiment.
)


def add_baseline(target):
    summary_path = OUTPUT_ROOT / f"subsampling_{target}_summary.csv"
    if not summary_path.exists():
        print(f"❌ No existe {summary_path} — hay que correr subsampling_experiment.py primero.")
        return
    summary = pd.read_csv(summary_path)
    print(f"\n📌 {target}: recalculando tabla comparativa con baseline incluido...")
    build_comparison_table(target, summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', choices=['c4', 'carbono_bruto4', 'all'], default='all')
    args = parser.parse_args()

    targets = ['c4', 'carbono_bruto4'] if args.target == 'all' else [args.target]
    for t in targets:
        add_baseline(t)
