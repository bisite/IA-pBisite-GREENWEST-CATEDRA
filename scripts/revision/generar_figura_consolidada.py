# -*- coding: utf-8 -*-
"""
GENERAR FIGURAS CONSOLIDADAS
============================
- Rangos temporales: IFN2 (15+) e IFN3 (<=20).
- Leyendas movidas para no tapar datos.
- Versiones con y sin histograma.
- Optimización de memoria (usecols, gc.collect).

Adaptado del script de figuras del pipeline de entrenamiento (round4). Genera:
    Consolidado_IFN{2,3}_{c4,carbono_bruto4}_clean.pdf   (usadas en el artículo)
    Consolidado_IFN{2,3}_{c4,carbono_bruto4}_paper.pdf   (versión con histograma, no usada en el artículo)
La versión "_clean" no renderiza el nombre del inventario en la figura,
solo en el nombre de fichero, que debe coincidir con
\includegraphics{Consolidado_IFN2_c4_clean.pdf} etc. del fuente LaTeX.
"""

import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import gc
import warnings
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# ==================== CONFIGURACIÓN ====================
REPO_BASE = Path(__file__).resolve().parents[2]  # .../article-ecof

# Trained models (global = NFI2+NFI3, local = NFI2 or NFI3 only).
# Layout: models/artifacts/<IFN2|IFN3|IFN2-3>/<target>/<base|stackings>/
ARTIFACTS_DIR = REPO_BASE / "models" / "artifacts"

CSV_FILE = str(REPO_BASE / "data/processed/ifn_entrenamiento_year_agrupado_periodo_completado_2.csv")
OUTPUT_DIR = str(REPO_BASE / "figures/consolidado")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Features necesarias
FEATURES_C4 = [
    'especie_id', 'tipo_especie', 'periodo_agrupado', 'radio', 'ocupa', 'npies_1', 'npies_2', 'npies_5',
    'npies_10', 'npies_15', 'npies_20', 'npies_25', 'npies_30', 'npies_35', 'npies_40', 'npies_45',
    'npies_50', 'npies_55', 'npies_60', 'npies_65', 'npies_70', 'estado_id', 'fccarb', 'disesp_id',
    'modcomb_id', 'nivel2_id', 'tratmasa_id', 'rocosidad_id', 'orientacion_cat', 'elevacion', 'pendiente_cat',
    'martonneidx_id', 'skt_mean_primavera', 'pr_sum_primavera', 'skt_std_verano', 'skt_std_primavera',
    'pr_sum_otoño', 'skt_mean_verano', 'pr_sum_verano', 'pr_sum_invierno', 'gndvi_mean_verano',
    'ndii_mean_primavera', 'gndvi_std_primavera', 'evi_mean_primavera'
]
FEATURES_CB = [
    'especie_id', 'tipo_especie', 'periodo_agrupado', 'radio', 'ocupa', 'npies_1', 'npies_2', 'npies_5',
    'npies_10', 'npies_15', 'npies_20', 'npies_25', 'npies_30', 'npies_35', 'npies_40', 'npies_45',
    'npies_50', 'npies_55', 'npies_60', 'npies_65', 'npies_70', 'estado_id', 'fccarb', 'disesp_id',
    'modcomb_id', 'nivel2_id', 'tratmasa_id', 'rocosidad_id', 'orientacion_cat', 'elevacion', 'pendiente_cat',
    'martonneidx_id', 'gndvi_mean_verano', 'ndii_mean_primavera', 'gndvi_std_primavera', 'evi_mean_primavera'
]

MAPA_PENDIENTE = {'muy suave (<5°)': 0, 'suave (5-10°)': 1, 'moderada (10-15°)': 2, 'fuerte (15-20°)': 3, 'muy fuerte (20-30°)': 4, 'escarpada (30-50°)': 5, 'extrema (>50°)': 6}
MAPA_ORIENTACION = {'N': 0, 'NE': 1, 'E': 2, 'SE': 3, 'S': 4, 'SO': 5, 'O': 6, 'NO': 7}

PATHS = {
    'c4': {
        'global_base': str(ARTIFACTS_DIR / 'IFN2-3' / 'c4' / 'base'),
        'ifn2_base': str(ARTIFACTS_DIR / 'IFN2' / 'c4' / 'base'),
        'ifn3_base': str(ARTIFACTS_DIR / 'IFN3' / 'c4' / 'base'),
        'global_stack': str(ARTIFACTS_DIR / 'IFN2-3' / 'c4' / 'stackings'),
        'ifn2_stack': str(ARTIFACTS_DIR / 'IFN2' / 'c4' / 'stackings'),
        'ifn3_stack': str(ARTIFACTS_DIR / 'IFN3' / 'c4' / 'stackings'),
        'color': '#3498db', 'unit': 't C/ha'
    },
    'carbono_bruto4': {
        'global_base': str(ARTIFACTS_DIR / 'IFN2-3' / 'carbono_bruto4' / 'base'),
        'ifn2_base': str(ARTIFACTS_DIR / 'IFN2' / 'carbono_bruto4' / 'base'),
        'ifn3_base': str(ARTIFACTS_DIR / 'IFN3' / 'carbono_bruto4' / 'base'),
        'global_stack': str(ARTIFACTS_DIR / 'IFN2-3' / 'carbono_bruto4' / 'stackings'),
        'ifn2_stack': str(ARTIFACTS_DIR / 'IFN2' / 'carbono_bruto4' / 'stackings'),
        'ifn3_stack': str(ARTIFACTS_DIR / 'IFN3' / 'carbono_bruto4' / 'stackings'),
        'color': '#27ae60', 'unit': 't C'
    }
}

STACK_CONFIGS = {
    1: ['LightGBM', 'Random_Forest'],
    2: ['LightGBM', 'XGBoost', 'GBDT'],
    3: ['CatBoost', 'Random_Forest', 'GBDT'],
    4: ['CatBoost', 'LightGBM', 'Random_Forest', 'GBDT'],
    5: ['CatBoost', 'LightGBM', 'XGBoost', 'Random_Forest', 'GBDT', 'BaggedDT'],
}

# Clases Dummy (necesarias para deserializar algunos .joblib con clases custom)
def cast_to_str(X): return pd.DataFrame(X).astype(str).values
def make_ohe(): return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
class PyTorchMLPDropout(nn.Module):
    def __init__(self, input_dim=1, hidden_layer_sizes=(100,), activation='relu', dropout_rate=0.2, use_batchnorm=True):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(input_dim, 1))
    def forward(self, x): return self.model(x)
class MLPRegressorTorch(BaseEstimator, RegressorMixin):
    def predict(self, X): return np.zeros(len(X))

def preparar_datos(df_raw, target, features, inventario_filter=None):
    if inventario_filter == 'IFN2':
        df_base = df_raw[df_raw['ifn_id'].astype(str).str.contains('2')].copy()
    elif inventario_filter == 'IFN3':
        df_base = df_raw[df_raw['ifn_id'].astype(str).str.contains('3')].copy()
    else:
        df_base = df_raw.copy()

    df_base['pendiente_cat'] = df_base['pendiente_cat'].map(MAPA_PENDIENTE)
    df_base['orientacion_cat'] = df_base['orientacion_cat'].map(MAPA_ORIENTACION)
    mask = (df_base[target].notna()) & (df_base[target] != 0) & ((df_base['c'].isna()) | (df_base['c'] < df_base[target]))
    df = df_base[mask].copy()
    df = df[~(df['fccarb'].notna() & (df['fccarb'] < 20))].copy()
    df['ifn_id_str'] = df['ifn_id'].astype(str)

    del df_base, mask

    present_features = [f for f in features if f in df.columns]
    X = df[present_features].astype('float32')
    y = df[target].values.astype('float32')

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_idx = next(gss.split(X, y, groups=df['parcela_id']))

    res = {
        'df_full': df[['periodo_agrupado', 'ifn_id']],
        'df_test': df.iloc[test_idx].copy(),
        'X_test': X.iloc[test_idx].copy(),
        'y_test': y[test_idx].copy()
    }
    del df, X, y; gc.collect()
    return res

def get_metrics_by_period(df_test, y_test, y_pred, inventory):
    """Calcula métricas aplicando los filtros de rango solicitados."""
    periodos = sorted(df_test['periodo_agrupado'].dropna().unique())
    res = {'Periodo': [], 'RMSE': [], 'RMSE_q25': [], 'RMSE_q75': [], 'RMSE_median': []}

    # Filtros de rango: IFN2 empieza alrededor de 15, IFN3 termina alrededor de 20
    MIN_P = 14 if 'IFN2' in inventory else 0
    MAX_P = 31 if 'IFN2' in inventory else 21

    for p in periodos:
        if p < MIN_P or p > MAX_P: continue

        mask = df_test['periodo_agrupado'] == p
        if mask.sum() < 5: continue

        y_r = y_test[mask]; y_p = y_pred[mask]
        res['Periodo'].append(int(p))
        res['RMSE'].append(np.sqrt(mean_squared_error(y_r, y_p)))
        err_abs = np.abs(y_r - y_p)
        res['RMSE_q25'].append(np.percentile(err_abs, 25))
        res['RMSE_q75'].append(np.percentile(err_abs, 75))
        res['RMSE_median'].append(np.percentile(err_abs, 50))
    return pd.DataFrame(res)


def _patch_model_compat(model):
    """Completa atributos que cambiaron de nombre/dejaron de existir entre
    versiones de XGBoost/cuML y los que se usaron para entrenar estos modelos
    (round4, ~enero 2026). Sin esto, joblib.load() funciona pero .predict()
    falla con AttributeError al acceder a get_params() internamente."""
    steps = model.steps if hasattr(model, 'steps') else [(None, model)]
    for _, step in steps:
        for attr, default in [('gpu_id', None), ('predictor', None),
                               ('_verbose', 0), ('n_gpus', 1),
                               ('output_type', 'numpy')]:
            if not hasattr(step, attr):
                try:
                    setattr(step, attr, default)
                except Exception:
                    pass
    return model

def get_prediction(model_name, folder_path, X, stack_folder=None, base_models_cache=None):
    base_path = f"{folder_path}/{model_name}.joblib"
    if os.path.exists(base_path) and "Stack" not in model_name:
        if base_path not in base_models_cache:
            base_models_cache[base_path] = _patch_model_compat(joblib.load(base_path))
        return base_models_cache[base_path].predict(X)

    if stack_folder:
        stack_path = f"{stack_folder}/{model_name}.joblib"
        if os.path.exists(stack_path):
            if stack_path not in base_models_cache:
                base_models_cache[stack_path] = _patch_model_compat(joblib.load(stack_path))
            meta_model = base_models_cache[stack_path]
            try:
                stack_id = int(model_name.split('_')[0].replace('Stack', ''))
                needed = STACK_CONFIGS[stack_id]
                preds_base = []
                for m_name in needed:
                    child_path = f"{folder_path}/{m_name}.joblib"
                    if child_path not in base_models_cache:
                         if os.path.exists(child_path): base_models_cache[child_path] = _patch_model_compat(joblib.load(child_path))
                    preds_base.append(base_models_cache[child_path].predict(X))

                X_meta = np.column_stack(preds_base)
                return meta_model.predict(X_meta)
            except Exception as e:
                print(f"Err {model_name}: {e}")
                return None
    return None

def resolve_model_with_data(comp_def, key_model, key_path_type, inventory, config, data_global, data_local, cache):
    m_name = comp_def[key_model]
    p_type = comp_def[key_path_type]

    if p_type == 'global':
        folder_base = config['global_base']
        folder_stack = config['global_stack']
        df_t = data_global['df_test']
        mask = df_t['ifn_id'].astype(str).str.contains('2' if 'IFN2' in inventory else '3')
        df_test = df_t[mask]
        X_test = data_global['X_test'].iloc[mask.values]
        y_test = data_global['y_test'][mask.values]
    elif p_type == 'local':
        folder_base = config[f'{inventory.lower()}_base']
        folder_stack = config[f'{inventory.lower()}_stack']
        df_test = data_local['df_test']
        X_test = data_local['X_test']
        y_test = data_local['y_test']
    else: return None, None, None

    preds = get_prediction(m_name, folder_base, X_test, folder_stack, cache)
    return preds, df_test, y_test

def plot_single_version(results, data_global, data_local, inventory, unit, save_path, with_hist=True):
    n_plots = len(results)

    if with_hist:
        fig = plt.figure(figsize=(6, 1.8 + 1.8*n_plots))
        ratios = [0.8] + [1.8] * n_plots
        gs = gridspec.GridSpec(n_plots + 1, 1, figure=fig, height_ratios=ratios, hspace=0.35)
        start_plot_idx = 1
    else:
        fig = plt.figure(figsize=(6, 1.8*n_plots))
        ratios = [1.0] * n_plots
        gs = gridspec.GridSpec(n_plots, 1, figure=fig, height_ratios=ratios, hspace=0.35)
        start_plot_idx = 0

    FS_LABEL, FS_TICK, FS_LEGEND, LW = 11, 9, 7, 2.0

    # Rango X fijo por inventario
    XMIN = 14.5 if 'IFN2' in inventory else 4.5
    XMAX = 30.5 if 'IFN2' in inventory else 20.5
    XTICKS = [15, 20, 25, 30] if 'IFN2' in inventory else [5, 10, 15, 20]

    # 1. Histograma (Opcional, solo en la versión "_paper", no usada en el artículo)
    ax_hist = None
    if with_hist:
        ax_hist = fig.add_subplot(gs[0])
        eje_x = np.arange(5, 31, 1)
        n_full_global = data_global['df_full']['periodo_agrupado'].value_counts().reindex(eje_x, fill_value=0)
        n_full_local = data_local['df_full']['periodo_agrupado'].value_counts().reindex(eje_x, fill_value=0)

        ax_hist.bar(n_full_global.index, n_full_global.values, color='gray', alpha=0.3, label='Global Samples')
        color_hist = '#3498db' if 'IFN2' in inventory else '#27ae60'
        # NFI en vez de IFN en el texto visible (nomenclatura del artículo) --
        # solo afecta a la versión "_paper".
        inv_label = inventory.replace('IFN', 'NFI')
        ax_hist.bar(n_full_local.index, n_full_local.values, color=color_hist, alpha=0.6, label=f'{inv_label} Samples')

        ax_hist.set_ylabel('N Samples', fontsize=FS_LABEL)
        ax_hist.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), fontsize=FS_LEGEND, ncol=2, frameon=False)
        ax_hist.tick_params(labelsize=FS_TICK, labelbottom=False)
        ax_hist.set_xlim(XMIN, XMAX)
        ax_hist.grid(axis='y', alpha=0.3)

    # 2. Plots
    for i, (ma, mb, title, label_a, label_b) in enumerate(results):
        ax = fig.add_subplot(gs[i + start_plot_idx], sharex=ax_hist if with_hist else None)

        def format_label(txt):
            parts = txt.split(' (')
            model_part = parts[0].replace('Stack', 'S').replace('GradientBoosting', 'GB').replace('_', ' ')
            scope_part = '(' + parts[1] if len(parts) > 1 else ''
            return f"{model_part} {scope_part}"

        color_a = '#3498db' if 'IFN2' in inventory else '#27ae60'
        ax.plot(ma['Periodo'], ma['RMSE_median'], 'o-', color=color_a, linewidth=LW, markersize=6, label=format_label(label_a))
        ax.fill_between(ma['Periodo'], ma['RMSE_q25'], ma['RMSE_q75'], color=color_a, alpha=0.15)

        ax.plot(mb['Periodo'], mb['RMSE_median'], '^-', color='#e74c3c', linewidth=LW, markersize=6, linestyle='--', label=format_label(label_b))
        ax.fill_between(mb['Periodo'], mb['RMSE_q25'], mb['RMSE_q75'], color='#e74c3c', alpha=0.1)

        from matplotlib.patches import Patch
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Patch(facecolor='gray', alpha=0.2, linewidth=0))
        labels.append("IQR (Q25-Q75)")

        ax.legend(handles=handles, labels=labels, loc='upper right', fontsize=FS_LEGEND, framealpha=0.8)

        ax.set_ylabel(f'Abs. Error ({unit})', fontsize=FS_LABEL)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=FS_TICK)
        ax.set_xlim(XMIN, XMAX)
        ax.set_xticks(XTICKS)

        if i == n_plots - 1:
            ax.set_xlabel('Years since intervention', fontsize=FS_LABEL)
        else:
            ax.tick_params(labelbottom=False)

    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    gc.collect()

def generate_figure_pair(inventory, target, comparisons_list, df_raw):
    print(f"\nProcesando {inventory} - {target}")
    feats = FEATURES_C4 if target == 'c4' else FEATURES_CB
    config = PATHS[target]

    data_global = preparar_datos(df_raw, target, feats, None)
    data_local = preparar_datos(df_raw, target, feats, inventory)

    cache = {}
    results = []

    for comp in comparisons_list:
        preds_a, df_test_a, y_test_a = resolve_model_with_data(comp, 'mA', 'pathA', inventory, config, data_global, data_local, cache)
        preds_b, df_test_b, y_test_b = resolve_model_with_data(comp, 'mB', 'pathB', inventory, config, data_global, data_local, cache)

        if preds_a is not None and preds_b is not None:
            ma = get_metrics_by_period(df_test_a, y_test_a, preds_a, inventory)
            mb = get_metrics_by_period(df_test_b, y_test_b, preds_b, inventory)
            if not ma.empty and not mb.empty:
                results.append((ma, mb, comp['title'], f"{comp['mA']} ({comp['pathA'].title()})", f"{comp['mB']} ({comp['pathB'].title()})"))
            del preds_a, preds_b
            gc.collect()

    del cache; gc.collect()
    if not results:
        print(f"   Sin datos en los rangos seleccionados para {inventory}")
        return

    # Nombres de fichero: deben coincidir con \includegraphics{Consolidado_IFN2_c4_clean.pdf}
    # etc. del tex -- IFN se mantiene en el nombre de fichero (no en el contenido visible).
    path_hist = f"{OUTPUT_DIR}/Consolidado_{inventory}_{target}_paper.pdf"
    plot_single_version(results, data_global, data_local, inventory, config['unit'], path_hist, with_hist=True)

    path_no_hist = f"{OUTPUT_DIR}/Consolidado_{inventory}_{target}_clean.pdf"
    plot_single_version(results, data_global, data_local, inventory, config['unit'], path_no_hist, with_hist=False)

    del data_global, data_local, results; gc.collect()

def main():
    print("Generando figuras consolidadas")

    needed_cols = set(['parcela_id', 'ifn_id', 'periodo_agrupado', 'c', 'c4', 'carbono_bruto4', 'fccarb', 'pendiente_cat', 'orientacion_cat'])
    needed_cols.update(FEATURES_C4); needed_cols.update(FEATURES_CB)

    df_raw = pd.read_csv(CSV_FILE, usecols=list(needed_cols))

    reqs = [
        ('IFN2', 'c4', 'LightGBM', 'global', 'LightGBM', 'local', 'Base'),
        ('IFN2', 'c4', 'Stack1_MLP', 'global', 'Stack5_GradientBoosting', 'local', 'Stack'),
        ('IFN2', 'c4', 'LightGBM', 'global', 'Stack5_GradientBoosting', 'local', 'Best'),
        ('IFN2', 'carbono_bruto4', 'LightGBM', 'global', 'LightGBM', 'local', 'Base'),
        ('IFN2', 'carbono_bruto4', 'Stack1_MLP', 'global', 'Stack5_GradientBoosting', 'local', 'Stack'),
        ('IFN2', 'carbono_bruto4', 'LightGBM', 'global', 'Stack5_GradientBoosting', 'local', 'Best'),
        ('IFN3', 'c4', 'LightGBM', 'global', 'LightGBM', 'local', 'Base'),
        ('IFN3', 'c4', 'Stack4_MLP', 'global', 'Stack4_GradientBoosting', 'local', 'Stack'),
        ('IFN3', 'c4', 'LightGBM', 'global', 'Stack4_GradientBoosting', 'local', 'Best'),
        ('IFN3', 'carbono_bruto4', 'CatBoost', 'global', 'LightGBM', 'local', 'Base'),
        ('IFN3', 'carbono_bruto4', 'Stack5_MLP', 'global', 'Stack4_MLP', 'local', 'Stack'),
        ('IFN3', 'carbono_bruto4', 'CatBoost', 'global', 'Stack4_MLP', 'local', 'Best'),
    ]

    tasks = {}
    for r in reqs:
        k = (r[0], r[1]); tasks.setdefault(k, []).append({'mA': r[2], 'pathA': r[3], 'mB': r[4], 'pathB': r[5], 'title': r[6]})

    for (inv, tgt), comps in tasks.items():
        generate_figure_pair(inv, tgt, comps, df_raw)

if __name__ == "__main__":
    main()
