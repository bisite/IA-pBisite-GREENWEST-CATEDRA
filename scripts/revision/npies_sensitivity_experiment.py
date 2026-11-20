# -*- coding: utf-8 -*-
"""
Experimento de sensibilidad: cuanto cae el R2 sin `npies`.

`npies` (num. de pies/arboles por clase diametrica) es multiplicador
aritmetico directo del target (carbono_bruto = biomasa(especie,dbh,altura)
x (1+raices) x npies, ver catedra/pipeline/3.1.Carbono_bruto.ipynb), lo que
plantea un posible problema de endogeneidad. Este script cuantifica cuanto
R2 retiene el modelo sin npies/NPies*, es decir, que parte del rendimiento
se debe a ese vinculo aritmetico y que parte a informacion real de
estructura de rodal.

Reentrena los mismos 9 modelos (mismos hiperparametros fijos que el modelo
global, sin GridSearch -- igual que subsampling_experiment.py) quitando
`npies_1..npies_70` de la lista de features, sobre el mismo pool
IFN2+IFN3 y el mismo split (GroupShuffleSplit seed=42, groups=parcela_id)
que se usa para el modelo global -- asi el R2_reducido es directamente
comparable a R2_global_full en comparison_table_<target>.csv
(generado por subsampling_experiment.py), sin
necesidad de reentrenar tambien el modelo completo.

Uso:
    python npies_sensitivity_experiment.py --target c4
    python npies_sensitivity_experiment.py --target carbono_bruto4
    python npies_sensitivity_experiment.py --target all
"""
import argparse
import gc
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import pandas as pd

from subsampling_experiment import (  # noqa: E402
    BASE_CAT_FEATURES, NPIES_ALL, TARGET_CONFIG,
    load_and_filter, split_by_group,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_ROOT = REPO_ROOT / 'models' / 'metrics' / 'npies_sensitivity'
COMPARISON_ROOT = REPO_ROOT / 'models' / 'metrics' / 'subsampling'
HELPER = Path(__file__).resolve().parent / '_train_one_reduced_model.py'


def fit_eval_one_no_npies(name, ctx_path):
    """Entrena UN modelo en un subproceso aislado (ver _train_one_reduced_model.py
    -- un segfault de cuml/GPU en un modelo no debe tumbar los demas)."""
    proc = subprocess.run([sys.executable, str(HELPER), str(ctx_path), name],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        reason = "segfault u otro crash del proceso" if proc.returncode < 0 or proc.returncode == 139 \
            else (proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else f"exit code {proc.returncode}")
        logging.warning(f"    No se pudo entrenar '{name}' (sin npies): {reason}")
        return None
    line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
    if line is None:
        logging.warning(f"    '{name}' (sin npies): subproceso sin resultado parseable.")
        return None
    parts = dict(p.split('=') for p in line.removeprefix("RESULT ").split())
    return {'Modelo': parts['Modelo'], 'Test_R2': float(parts['Test_R2']),
            'Test_RMSE': float(parts['Test_RMSE']), 'Tiempo_s': float(parts['Tiempo_s'])}


def run_target(target, use_npies):
    """use_npies=False (modo normal): quita npies*, resultado comparable a
    R2_global_full. use_npies=True (modo CONTROL): reentrena con TODAS las
    features (igual que el modelo global original) para verificar que
    reentrenar-desde-cero-ahora reproduce R2_global_full -- sin este
    control no se puede distinguir "efecto de quitar npies" de "ruido de
    reentrenar en este entorno"."""
    config = TARGET_CONFIG[target]
    full_features = config['features']
    features = full_features if use_npies else [f for f in full_features if f not in NPIES_ALL]
    tag = 'CON npies (control)' if use_npies else 'sin npies'
    logging.info(f"=== Sensibilidad npies — target={target} [{tag}] ===")
    logging.info(f"Features completas: {len(full_features)}, usadas: {len(features)}")

    df = load_and_filter(target, restrict_to_ifn3=False)
    present_cols = [c for c in features if c in df.columns]
    present_cat = [c for c in BASE_CAT_FEATURES if c in present_cols]
    present_npies = [c for c in NPIES_ALL if c in present_cols]
    present_num = [c for c in present_cols if c not in present_cat and c not in present_npies]

    X = df[present_cols].copy().astype('float32')
    y = df[target]
    groups = df['parcela_id'] if 'parcela_id' in df.columns else None

    X_train, X_test, y_train, y_test, groups_train, cv = split_by_group(X, y, groups)
    logging.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    suffix = 'control_con_npies' if use_npies else 'sin_npies'
    ctx_path = OUTPUT_ROOT / f"_ctx_{target}_{suffix}.joblib"
    joblib.dump({
        'target': target, 'X_train': X_train, 'y_train': y_train,
        'X_test': X_test, 'y_test': y_test,
        'present_num': present_num, 'present_npies': present_npies, 'present_cat': present_cat,
    }, ctx_path)
    del X_train, X_test, y_train, y_test, X, df
    gc.collect()

    rows = []
    for name in config['param_grids']:
        logging.info(f"  Entrenando {name} [{tag}]...")
        result = fit_eval_one_no_npies(name, ctx_path)
        if result is None:
            continue
        logging.info(f"    Test_R2={result['Test_R2']:.4f} Test_RMSE={result['Test_RMSE']:.3f} "
                      f"({result['Tiempo_s']:.1f}s)")
        rows.append(result)

    ctx_path.unlink(missing_ok=True)
    df_result = pd.DataFrame(rows)
    out_path = OUTPUT_ROOT / f"sensitivity_{suffix}_{target}.csv"
    df_result.to_csv(out_path, index=False)
    logging.info(f"  💾 {out_path}")
    return df_result


def compare_against_baseline(target, df_reduced, df_control):
    comp_path = COMPARISON_ROOT / f"comparison_table_{target}.csv"
    if not comp_path.exists():
        logging.warning(f"  No se encontró {comp_path} — no se puede comparar contra el modelo completo.")
        return
    comp = pd.read_csv(comp_path)[['Modelo', 'R2_global_full', 'RMSE_global_full']]

    if df_control is not None and not df_control.empty:
        ctrl = comp.merge(df_control.rename(columns={'Test_R2': 'Test_R2_control_reentrenado'}),
                           on='Modelo', how='inner')
        ctrl['diff_R2'] = ctrl['Test_R2_control_reentrenado'] - ctrl['R2_global_full']
        print(f"\n=== {target}: CONTROL — reentrenar con npies (ahora) vs. R2_global_full (cargado) ===")
        print(ctrl[['Modelo', 'R2_global_full', 'Test_R2_control_reentrenado', 'diff_R2']]
              .to_string(index=False))
        max_diff = ctrl['diff_R2'].abs().max()
        print(f"Diferencia máxima absoluta: {max_diff:.5f} "
              f"({'✅ reentrenar reproduce el original' if max_diff < 0.01 else '⚠️ diferencia no trivial'})")

    merged = df_reduced.merge(comp, on='Modelo', how='left')
    merged['R2_drop'] = merged['R2_global_full'] - merged['Test_R2']
    merged['R2_retenido_pct'] = 100 * merged['Test_R2'] / merged['R2_global_full']
    print(f"\n=== {target}: con npies (global_full) vs. sin npies (sensibilidad) ===")
    print(merged[['Modelo', 'R2_global_full', 'Test_R2', 'R2_drop', 'R2_retenido_pct']]
          .to_string(index=False))
    merged.to_csv(OUTPUT_ROOT / f"comparison_no_npies_{target}.csv", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', choices=['c4', 'carbono_bruto4', 'all'], default='all')
    parser.add_argument('--with-control', action='store_true',
                        help='También reentrena CON npies para verificar que reproduce R2_global_full')
    args = parser.parse_args()
    targets = ['c4', 'carbono_bruto4'] if args.target == 'all' else [args.target]
    for t in targets:
        df_control = run_target(t, use_npies=True) if args.with_control else None
        df_reduced = run_target(t, use_npies=False)
        compare_against_baseline(t, df_reduced, df_control)
