# -*- coding: utf-8 -*-
"""
SHAP dependence de `periodo_agrupado`: explica el salto no monotono en el
"anio 17" visible en Fig 6.4/6.5.

Hipotesis (confirmada por codigo, ver catedra/pipeline/4.Extraer_BBDD.ipynb,
funcion `agrupar_periodo`): el salto no es un efecto ecologico real, es un
artefacto de la codificacion. `periodo` (anios entre campanias IFN, crudo)
se agrupa asi:

    p<6        -> 5
    7<=p<=10   -> 10
    11<=p<=17  -> sin agrupar (se queda igual: 11,12,...,17)
    18<=p<=21  -> 20
    22<=p<=28  -> sin agrupar
    p>=29      -> 30

periodo=17 conserva su propio valor, pero periodo=18 salta directamente a
20 -- un salto de +3 en la variable que el modelo realmente ve como input,
sin que exista un cambio real en el periodo temporal entre 17 y 18 anios.

Genera, por target, dos paneles: SHAP(periodo) vs periodo_agrupado (lo que
el modelo ve) y SHAP(periodo) vs periodo crudo (0-34, disponible en el CSV
sin agrupar) -- si el salto desaparece en el segundo panel, confirma que es
artefacto de la codificacion y no una no-linealidad real del modelo.

Uso:
    python shap_periodo_analysis.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import types
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import GroupShuffleSplit

# pandas <2.0 pickle shim -- necesario para
# deserializar los training_context/pipelines guardados con pandas viejo.
shim = types.ModuleType('pandas.core.indexes.numeric')
for _name in ('Int64Index', 'UInt64Index', 'Float64Index', 'NumericIndex'):
    setattr(shim, _name, pd.Index)
sys.modules['pandas.core.indexes.numeric'] = shim


def cast_to_str(X):
    """Necesaria en __main__: los pipelines pickleados referencian
    __main__.cast_to_str (el script de entrenamiento se ejecuto como
    programa principal)."""
    return pd.DataFrame(X).astype(str).values


import __main__  # noqa: E402
__main__.cast_to_str = cast_to_str

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CSV_FILE = REPO_ROOT / 'data' / 'processed' / 'ifn_entrenamiento_year_agrupado_periodo_completado_2.csv'
ARTIFACTS_ROOT = REPO_ROOT / 'models' / 'artifacts' / 'IFN2-3'
OUTPUT_DIR = REPO_ROOT / 'figures' / 'shap_periodo'

MAPA_PENDIENTE = {'muy suave (<5°)': 0, 'suave (5-10°)': 1, 'moderada (10-15°)': 2,
                  'fuerte (15-20°)': 3, 'muy fuerte (20-30°)': 4, 'escarpada (30-50°)': 5,
                  'extrema (>50°)': 6}
MAPA_ORIENTACION = {'N': 0, 'NE': 1, 'E': 2, 'SE': 3, 'S': 4, 'SO': 5, 'O': 6, 'NO': 7}

NPIES_ALL = [f'npies_{n}' for n in (1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70)]

FEATURES_C4 = [
    'especie_id', 'tipo_especie', 'periodo_agrupado', 'radio', 'ocupa',
    *NPIES_ALL, 'estado_id', 'fccarb', 'disesp_id', 'modcomb_id',
    'nivel2_id', 'tratmasa_id', 'rocosidad_id', 'orientacion_cat',
    'elevacion', 'pendiente_cat', 'martonneidx_id',
    'skt_mean_primavera', 'pr_sum_primavera', 'skt_std_verano', 'skt_std_primavera',
    'pr_sum_otoño', 'skt_mean_verano', 'pr_sum_verano', 'pr_sum_invierno',
    'gndvi_mean_verano', 'ndii_mean_primavera', 'gndvi_std_primavera', 'evi_mean_primavera',
]
FEATURES_CB = [
    'especie_id', 'tipo_especie', 'periodo_agrupado', 'radio', 'ocupa',
    *NPIES_ALL, 'estado_id', 'fccarb', 'disesp_id', 'modcomb_id',
    'nivel2_id', 'tratmasa_id', 'rocosidad_id', 'orientacion_cat',
    'elevacion', 'pendiente_cat', 'martonneidx_id',
    'gndvi_mean_verano', 'ndii_mean_primavera', 'gndvi_std_primavera', 'evi_mean_primavera',
]

CONFIGS = [
    {'target': 'c4', 'model_name': 'LightGBM', 'features': FEATURES_C4},
    {'target': 'carbono_bruto4', 'model_name': 'CatBoost', 'features': FEATURES_CB},
]

# Fronteras de `agrupar_periodo` (ver docstring del modulo) -- para marcar
# visualmente donde el esquema de agrupacion cambia de comportamiento.
BIN_EDGES = [6, 10.5, 17.5, 21.5, 28.5]


def load_and_split(target, features):
    df_base = pd.read_csv(CSV_FILE)
    df_base['pendiente_cat'] = df_base['pendiente_cat'].map(MAPA_PENDIENTE)
    df_base['orientacion_cat'] = df_base['orientacion_cat'].map(MAPA_ORIENTACION)

    df = df_base[(df_base[target].notna()) & (df_base[target] != 0)
                 & ((df_base['c'].isna()) | (df_base['c'] < df_base[target]))].copy()
    df = df[~(df['fccarb'].notna() & (df['fccarb'] < 20))].copy()
    df = df[df['periodo_agrupado'].notna()].copy()

    present_features = [c for c in features if c in df.columns]
    X_full = df[present_features].copy().astype('float32')
    y_full = df[target].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_idx = next(gss.split(X_full, y_full, groups=df['parcela_id']))
    return df.iloc[test_idx], X_full.iloc[test_idx]


def periodo_shap_index(preprocessor):
    """periodo_agrupado es la primera columna del bloque 'num', y 'num' es
    el primer transformer del ColumnTransformer -> indice 0 en la salida
    transformada. Se calcula en vez de asumirlo, por si el orden cambiara."""
    num_cols = list(preprocessor.transformers_[0][2])
    offset = 0  # 'num' es siempre el primer bloque en estos pipelines
    return offset + num_cols.index('periodo_agrupado')


def summarize_jump(periodo_agrupado, shap_vals):
    """Cuantifica el salto: SHAP medio en periodo_agrupado=17 (ultimo valor
    sin agrupar) vs periodo_agrupado=20 (agrupa 18-21)."""
    m17 = shap_vals[periodo_agrupado == 17]
    m20 = shap_vals[periodo_agrupado == 20]
    if len(m17) == 0 or len(m20) == 0:
        return None
    return {
        'mean_shap_17': float(np.mean(m17)), 'n_17': len(m17),
        'mean_shap_20': float(np.mean(m20)), 'n_20': len(m20),
        'jump': float(np.mean(m20) - np.mean(m17)),
    }


def run_config(config):
    target = config['target']
    model_name = config['model_name']
    print(f"\n📌 {target} — {model_name}")

    df_test, X_test = load_and_split(target, config['features'])
    model = joblib.load(ARTIFACTS_ROOT / target / 'base' / f'{model_name}.joblib')
    preprocessor = model.named_steps['preprocessor']
    regressor = model.named_steps['regressor']

    X_test_t = preprocessor.transform(X_test)
    idx = periodo_shap_index(preprocessor)

    explainer = shap.TreeExplainer(regressor)
    shap_values = explainer.shap_values(X_test_t)
    periodo_shap = shap_values[:, idx]

    periodo_agrupado = df_test['periodo_agrupado'].values
    has_raw = 'periodo' in df_test.columns
    periodo_raw = df_test['periodo'].values if has_raw else None

    jump = summarize_jump(periodo_agrupado, periodo_shap)
    if jump:
        print(f"   SHAP medio en periodo_agrupado=17 (n={jump['n_17']}): {jump['mean_shap_17']:.3f}")
        print(f"   SHAP medio en periodo_agrupado=20 (n={jump['n_20']}): {jump['mean_shap_20']:.3f}")
        print(f"   Salto (20 - 17): {jump['jump']:+.3f}")

    fig, axes = plt.subplots(1, 2 if has_raw else 1, figsize=(12 if has_raw else 6, 5))
    if not has_raw:
        axes = [axes]

    ax = axes[0]
    ax.scatter(periodo_agrupado, periodo_shap, s=6, alpha=0.3, color='#3498db')
    for edge in BIN_EDGES:
        ax.axvline(edge, color='red', linestyle='--', linewidth=1, alpha=0.6)
    ax.axvspan(17.5, 21.5, color='red', alpha=0.06)
    ax.set_xlabel('periodo_agrupado (lo que el modelo ve)')
    ax.set_ylabel('SHAP(periodo)')
    ax.set_title(f'{target} / {model_name}\nvs. periodo_agrupado (codificado)')
    ax.grid(True, alpha=0.3)

    if has_raw:
        ax2 = axes[1]
        ax2.scatter(periodo_raw, periodo_shap, s=6, alpha=0.3, color='#27ae60')
        ax2.set_xlabel('periodo (años, sin agrupar)')
        ax2.set_ylabel('SHAP(periodo)')
        ax2.set_title(f'{target} / {model_name}\nvs. periodo crudo (sin codificar)')
        ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f'{target}_{model_name}_periodo_dependence'
    fig.savefig(f'{out}.png', dpi=200, bbox_inches='tight')
    fig.savefig(f'{out}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"   💾 {out}.png/.pdf")


if __name__ == '__main__':
    for config in CONFIGS:
        run_config(config)
