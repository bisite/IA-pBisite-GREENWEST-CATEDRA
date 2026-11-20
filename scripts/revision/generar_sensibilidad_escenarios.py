# -*- coding: utf-8 -*-
"""
Análisis de Sensibilidad: Evolución del Carbono bajo distintos Escenarios Iniciales
===================================================================================

Adaptado del script de figuras del pipeline de entrenamiento (round4). Genera:
    sensibilidad_escenarios_c4_P10_P35_P65_P90.pdf
    sensibilidad_escenarios_carbono_bruto4_P10_P35_P65_P90.pdf
(usadas en el artículo) además de la versión combinada y el .txt de
configuración de escenarios.

Este script selecciona 4 escenarios reales de densidad inicial (basados en datos de inventario)
y proyecta su crecimiento en el tiempo usando los modelos entrenados.
"""

import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import os
import warnings
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.base import BaseEstimator, RegressorMixin

warnings.filterwarnings("ignore")

# ==================== CONFIGURACIÓN ESTÉTICA (Publicación) ====================
FONT_SIZES = {'title': 14, 'label': 12, 'tick': 10, 'legend': 9}

plt.rcParams.update({
    'font.size': FONT_SIZES['tick'],
    'axes.labelsize': FONT_SIZES['label'],
    'axes.titlesize': FONT_SIZES['title'],
    'xtick.labelsize': FONT_SIZES['tick'],
    'ytick.labelsize': FONT_SIZES['tick'],
    'legend.fontsize': FONT_SIZES['legend'],
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'lines.linewidth': 1.5,
})

from sklearn.preprocessing import OneHotEncoder

# ==================== CLASES PYTORCH Y HELPER FUNCTIONS ====================
def cast_to_str(X):
    return pd.DataFrame(X).astype(str).values

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)

class PyTorchMLPDropout(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes=(100,), activation='relu',
                 dropout_rate=0.2, use_batchnorm=True):
        super(PyTorchMLPDropout, self).__init__()
        layers = []
        in_dim = input_dim
        act_fn = nn.ReLU if activation == 'relu' else nn.Tanh
        for h_dim in hidden_layer_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_batchnorm: layers.append(nn.BatchNorm1d(h_dim))
            layers.append(act_fn())
            if dropout_rate > 0: layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

PyTorchMLP = PyTorchMLPDropout

class MLPRegressorTorch(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        for k, v in kwargs.items(): setattr(self, k, v)
        self.model_ = None
    def predict(self, X):
        if hasattr(X, 'values'): X = X.values
        if not hasattr(self, 'device') or self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_.to(self.device)
        self.model_.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.model_(X_tensor).cpu().numpy().flatten()

# ==================== CONFIGURACIÓN DE RUTAS Y MODELOS ====================
REPO_BASE = Path(__file__).resolve().parents[2]  # .../article-ecof
# Global (NFI2+NFI3) base models: models/artifacts/IFN2-3/<target>/base/
ARTIFACTS_DIR = REPO_BASE / "models" / "artifacts"
# The Stack5_MLP meta-models used for this figure come from an earlier training
# run of the global stacking (14 Jan 2026) than the one in models/artifacts
# (22 Jan 2026): same configuration, different fitted weights. They are kept in
# models/scenarios/ so that the published figure can be reproduced exactly.
SCENARIO_MODELS_DIR = REPO_BASE / "models" / "scenarios"

CSV_FILE = str(REPO_BASE / "data/processed/ifn_entrenamiento_year_agrupado_periodo_completado_2.csv")
OUTPUT_DIR = str(REPO_BASE / "figures/sensibilidad_escenarios")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NPIES_COLS = [f'npies_{i}' for i in [1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]]

FEATURES_C4 = [
    'especie_id', 'tipo_especie', 'periodo_agrupado', 'radio', 'ocupa',
    *NPIES_COLS,
    'estado_id', 'fccarb', 'disesp_id', 'modcomb_id', 'nivel2_id', 'tratmasa_id',
    'rocosidad_id', 'orientacion_cat', 'elevacion', 'pendiente_cat', 'martonneidx_id',
    'skt_mean_primavera', 'pr_sum_primavera', 'skt_std_verano', 'skt_std_primavera',
    'pr_sum_otoño', 'skt_mean_verano', 'pr_sum_verano', 'pr_sum_invierno',
    'gndvi_mean_verano', 'ndii_mean_primavera', 'gndvi_std_primavera', 'evi_mean_primavera'
]

FEATURES_CB = [
    'especie_id', 'tipo_especie', 'periodo_agrupado', 'radio', 'ocupa',
    *NPIES_COLS,
    'estado_id', 'fccarb', 'disesp_id', 'modcomb_id', 'nivel2_id', 'tratmasa_id',
    'rocosidad_id', 'orientacion_cat', 'elevacion', 'pendiente_cat', 'martonneidx_id',
    'gndvi_mean_verano', 'ndii_mean_primavera', 'gndvi_std_primavera', 'evi_mean_primavera'
]

CONFIGS = [
    {
        'target': 'c4', 'model_name': 'LightGBM', 'model_type': 'base',
        'path': str(ARTIFACTS_DIR / 'IFN2-3' / 'c4' / 'base' / 'LightGBM.joblib'),
        'features': FEATURES_C4
    },
    {
        'target': 'c4', 'model_name': 'Stack5_MLP', 'model_type': 'stacking',
        'path': str(SCENARIO_MODELS_DIR / 'c4' / 'Stack5_MLP.joblib'),
        'base_dir': str(ARTIFACTS_DIR / 'IFN2-3' / 'c4' / 'base'),
        'base_models': ['CatBoost', 'LightGBM', 'XGBoost', 'Random_Forest', 'GBDT', 'BaggedDT'],
        'features': FEATURES_C4
    },
    {
        'target': 'carbono_bruto4', 'model_name': 'CatBoost', 'model_type': 'base',
        'path': str(ARTIFACTS_DIR / 'IFN2-3' / 'carbono_bruto4' / 'base' / 'CatBoost.joblib'),
        'features': FEATURES_CB
    },
    {
        'target': 'carbono_bruto4', 'model_name': 'Stack5_MLP', 'model_type': 'stacking',
        'path': str(SCENARIO_MODELS_DIR / 'carbono_bruto4' / 'Stack5_MLP.joblib'),
        'base_dir': str(ARTIFACTS_DIR / 'IFN2-3' / 'carbono_bruto4' / 'base'),
        'base_models': ['CatBoost', 'LightGBM', 'XGBoost', 'Random_Forest', 'GBDT', 'BaggedDT'],
        'features': FEATURES_CB
    },
]

MAPA_PENDIENTE = {'muy suave (<5°)': 0, 'suave (5-10°)': 1, 'moderada (10-15°)': 2,
                  'fuerte (15-20°)': 3, 'muy fuerte (20-30°)': 4, 'escarpada (30-50°)': 5, 'extrema (>50°)': 6}
MAPA_ORIENTACION = {'N': 0, 'NE': 1, 'E': 2, 'SE': 3, 'S': 4, 'SO': 5, 'O': 6, 'NO': 7}

# ==================== LÓGICA DE DIBUJADO ====================

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

def get_model_prediction(config, X_df):
    try:
        if config['model_type'] == 'base':
            model = _patch_model_compat(joblib.load(config['path']))
            if hasattr(model, 'verbose'): model.verbose = 0
            return model.predict(X_df)

        elif config['model_type'] == 'stacking':
            base_preds = []
            for base_name in config['base_models']:
                b_path = f"{config['base_dir']}/{base_name}.joblib"
                b_model = _patch_model_compat(joblib.load(b_path))
                if hasattr(b_model, 'verbose'): b_model.verbose = 0
                base_preds.append(b_model.predict(X_df))

            X_stack = np.column_stack(base_preds)
            meta = _patch_model_compat(joblib.load(config['path']))
            return meta.predict(X_stack)

    except Exception as e:
        print(f"Error en predicción ({config['model_name']}): {e}")
        return None  # no rellenar con ceros -- eso falsearia la figura

# ==================== MAIN ====================
print("="*70)
print("ANALISIS ESCENARIOS DE DENSIDAD")
print("="*70)

print("Cargando dataset...")
df = pd.read_csv(CSV_FILE)
df['pendiente_cat'] = df['pendiente_cat'].map(MAPA_PENDIENTE)
df['orientacion_cat'] = df['orientacion_cat'].map(MAPA_ORIENTACION)

PERIODOS_GLOBALES = sorted(df['periodo_agrupado'].dropna().unique().astype(int).tolist())
print(f"Periodos globales (IFN2+IFN3): {PERIODOS_GLOBALES}")

print("Filtrando solo datos de IFN3...")
if 'ifn_id' in df.columns:
    df = df[df['ifn_id'].astype(str).str.contains('3')].copy()
elif 'dataset' in df.columns:
    df = df[df['dataset'].astype(str).str.contains('3')].copy()

if len(df) == 0:
    raise ValueError("No se encontraron datos del IFN3 tras el filtrado.")

especie_top = df['especie_id'].mode()[0]
df_esp = df[df['especie_id'] == especie_top].copy()
print(f"Especie mas comun: {int(especie_top)} (N={len(df_esp)})")

df_esp['densidad_total'] = df_esp[NPIES_COLS].sum(axis=1)

percentiles = [10, 35, 65, 90]
labels = ['Very Low (P10)', 'Low (P35)', 'High (P65)', 'Very High (P90)']
escenarios = []

suffix = "_" + "_".join([f"P{p}" for p in percentiles])
print(f"Sufijo de archivo: {suffix}")

print("\nSeleccionando escenarios reales de densidad inicial:")
for pct, label in zip(percentiles, labels):
    target_val = np.percentile(df_esp['densidad_total'], pct)
    idx_closest = (df_esp['densidad_total'] - target_val).abs().idxmin()
    row = df_esp.loc[idx_closest]

    escenarios.append({
        'label': label, 'pct': pct, 'densidad': row['densidad_total'], 'data': row
    })
    print(f"   {label}: {int(row['densidad_total'])} pies totales (Idx: {idx_closest})")

    npies_str = []
    for col in NPIES_COLS:
        val = row.get(col, 0)
        if pd.notna(val) and val != 0:
            npies_str.append(f"{col.replace('npies_', '')}cm:{val:.0f}")

    if not npies_str: npies_str = ["Ninguno"]
    desc_str = f"      [Input Npies > 0] {', '.join(npies_str)}"

    full_desc = []
    full_desc.append(f"FCC: {row.get('fccarb', 'N/A')}%, Elev: {row.get('elevacion', 'N/A')}m")
    full_desc.append(desc_str.strip())
    full_desc.append("\n      [Resto de Features del Modelo]:")

    all_features_used = set(FEATURES_C4 + FEATURES_CB)
    already_shown = set(NPIES_COLS + ['fccarb', 'elevacion', 'periodo_agrupado'])
    features_to_show = sorted(list(all_features_used - already_shown))

    for col in features_to_show:
        if col in row and pd.notna(row[col]):
            val = row[col]
            if isinstance(val, float):
                full_desc.append(f"      {col}: {val:.4f}")
            else:
                full_desc.append(f"      {col}: {val}")

    escenarios[-1]['descripcion'] = "\n".join(full_desc)

config_file = f"{OUTPUT_DIR}/escenarios_config{suffix}.txt"
with open(config_file, "w") as f:
    f.write("CONFIGURACION DE ESCENARIOS DE SENSIBILIDAD\n")
    f.write("===========================================\n\n")
    for esc in escenarios:
        f.write(f"--- Escenario: {esc['label']} ---\n")
        f.write(f"Densidad Total: {int(esc['densidad'])} pies\n")
        f.write(f"{esc['descripcion']}\n\n")
print(f"\nConfiguracion guardada en: {config_file}")

periodos_sim = PERIODOS_GLOBALES
print(f"Periodos para simulacion: {periodos_sim}")
resultados = {}

print("\nGenerando proyecciones...")

for config in CONFIGS:
    key = f"{config['target']}_{config['model_name']}"
    print(f"   Procesando {key}...")

    resultados[key] = []

    for esc in escenarios:
        df_sim = pd.DataFrame([esc['data']] * len(periodos_sim))
        df_sim['periodo_agrupado'] = periodos_sim

        cols_necesarias = [c for c in config['features'] if c in df_sim.columns]
        X_sim = df_sim[cols_necesarias].astype('float32')

        preds = get_model_prediction(config, X_sim)
        if preds is None:
            print(f"   [SKIP] {key} / {esc['label']}: prediccion no disponible en este entorno")

        resultados[key].append({
            'escenario': esc['label'], 'periodos': periodos_sim, 'predicciones': preds
        })

colors = ['#2c3e50', '#3498db', '#e67e22', '#c0392b']

def plot_scenario(ax, results, target_name, colors, ylabel, add_legend=False):
    any_stack = False
    for i, esc_res in enumerate(results):
        if esc_res['predicciones'] is None:
            continue
        ax.plot(esc_res['periodos'], esc_res['predicciones'], color=colors[i], linestyle='-', linewidth=2, alpha=0.9, label=f"{esc_res['escenario']} (Base)")

        target_stack = target_name.replace('LightGBM', 'Stack5_MLP').replace('CatBoost', 'Stack5_MLP')
        st_res = resultados[target_stack][i]
        if st_res['predicciones'] is None:
            # Modelo de stacking no disponible en este entorno (p.ej. por
            # incompatibilidad binaria de cuML) -- se omite la serie en vez
            # de dibujar una linea falsa en 0.
            continue
        any_stack = True
        ax.plot(st_res['periodos'], st_res['predicciones'], color=colors[i], linestyle='--', linewidth=2, alpha=0.9, label=f"{esc_res['escenario']} (Stack)")

    ax.set_xlabel('Period (years)')
    ax.set_ylabel(ylabel)
    ax.set_xlim(5, 30)
    ax.grid(True, alpha=0.3)

    if add_legend:
        custom_lines = [Line2D([0], [0], color='gray', lw=2, linestyle='-', label='Base Model')]
        if any_stack:
            custom_lines.append(Line2D([0], [0], color='gray', lw=2, linestyle='--', label='Stacking'))
        for i, esc in enumerate(results):
            custom_lines.append(Line2D([0], [0], color=colors[i], lw=2, marker='o', label=esc['escenario']))

        ax.legend(handles=custom_lines, loc='upper left', fontsize=8, ncol=2, frameon=True, edgecolor='gray')

MARGINS = {'left': 0.15, 'right': 0.95, 'top': 0.95, 'bottom': 0.15}

fig1 = plt.figure(figsize=(6, 6))
fig1.subplots_adjust(**MARGINS)
ax1 = fig1.add_subplot(111)
plot_scenario(ax1, resultados['c4_LightGBM'], 'c4_LightGBM', colors, 'Predicted C4 (t C/ha)', add_legend=True)
fig1.savefig(f"{OUTPUT_DIR}/sensibilidad_escenarios_c4{suffix}.png", dpi=300)
fig1.savefig(f"{OUTPUT_DIR}/sensibilidad_escenarios_c4{suffix}.pdf")
plt.close(fig1)

fig2 = plt.figure(figsize=(6, 6))
fig2.subplots_adjust(**MARGINS)
ax2 = fig2.add_subplot(111)
plot_scenario(ax2, resultados['carbono_bruto4_CatBoost'], 'carbono_bruto4_CatBoost', colors, 'Predicted C. Bruto (t C)', add_legend=True)
fig2.savefig(f"{OUTPUT_DIR}/sensibilidad_escenarios_carbono_bruto4{suffix}.png", dpi=300)
fig2.savefig(f"{OUTPUT_DIR}/sensibilidad_escenarios_carbono_bruto4{suffix}.pdf")
plt.close(fig2)

fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.2, top=0.95, bottom=0.15, left=0.08, right=0.95)

ax_comb1 = fig.add_subplot(gs[0])
plot_scenario(ax_comb1, resultados['c4_LightGBM'], 'c4_LightGBM', colors, 'Predicted C4 (t C/ha)', add_legend=False)

ax_comb2 = fig.add_subplot(gs[1])
plot_scenario(ax_comb2, resultados['carbono_bruto4_CatBoost'], 'carbono_bruto4_CatBoost', colors, 'Predicted C. Bruto (t C)', add_legend=False)

custom_lines = [Line2D([0], [0], color='gray', lw=2, linestyle='-'),
                Line2D([0], [0], color='gray', lw=2, linestyle='--')]
for c in colors:
    custom_lines.append(Line2D([0], [0], color=c, lw=2, marker='o'))

legend_labels = ['LightGBM / CatBoost (Base)', 'Stack5_MLP (Stacking)'] + [e['label'] for e in escenarios]

fig.legend(custom_lines, legend_labels, loc='lower center', ncol=6, bbox_to_anchor=(0.5, 0.02), frameon=False)

output_name = f"{OUTPUT_DIR}/sensibilidad_escenarios_densidad{suffix}"
fig.savefig(f"{output_name}.pdf", bbox_inches='tight')
fig.savefig(f"{output_name}.png", dpi=300, bbox_inches='tight')

print(f"\nGraficos generados:")
print(f"   {output_name}.png")
print(f"   {output_name}.pdf")
