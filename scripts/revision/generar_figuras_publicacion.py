# -*- coding: utf-8 -*-
"""
SCRIPT UNIFICADO PARA GENERAR FIGURAS DE PUBLICACIÓN (V2: VECTORIAL & CONSISTENTE)
==================================================================================
Genera gráficos usando GridSpec para garantizar consistencia exacta en tamaños de fuente
y elementos entre las figuras individuales y las combinadas.

Modelos incluidos:
  - c4: LightGBM (base), Stack5_MLP (stacking)
  - carbono_bruto4: CatBoost (base), Stack5_MLP (stacking)

--------------------------------------------------------------------------------
Fig 6.2 (panel de densidad, draw_density()):
  - Eje Y = observado, eje X = predicho (Piñeiro et al. 2008, Ecological
    Modelling 216, 316-322), con histogramas marginales de cada eje.
  - Regresión observado = a + b * predicho con dos t-tests (H0: a=0,
    H0: b=1) anotados en el panel, vía `test_unbiased_fit()`
    (scripts/revision/regression_scatter.py).
Las rutas son relativas al repositorio (data/processed/,
models/artifacts/IFN2-3/), vía Path(__file__).
--------------------------------------------------------------------------------
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gc
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import gaussian_kde, percentileofscore
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from regression_scatter import test_unbiased_fit  # noqa: E402


def patch_xgboost_legacy_attrs(pipeline_or_estimator):
    """
    Los .joblib de XGBoost pickleados con una versión distinta a la instalada
    (aquí: 1.7.6) pueden carecer de atributos legacy (`gpu_id`, `predictor`)
    que XGBModel.get_params() espera encontrar via getattr() al construir el
    __init__ signature de la clase padre. Sin ellos, .predict() falla con
    AttributeError la primera vez que internamente llama a get_xgb_params()
    (para decidir si usar predicción in-place). Ponerlos a None es seguro:
    no cambia los pesos ni el comportamiento del modelo, solo evita el
    fallo al consultar un parámetro de aceleración GPU que no se usa aquí.
    No-op para modelos que no son XGBoost.
    """
    steps = pipeline_or_estimator.steps if hasattr(pipeline_or_estimator, 'steps') else [(None, pipeline_or_estimator)]
    for _, step in steps:
        if type(step).__module__.startswith('xgboost'):
            for attr in ('gpu_id', 'predictor'):
                if not hasattr(step, attr):
                    setattr(step, attr, None)
    return pipeline_or_estimator


def patch_cuml_setstate_verbose():
    """
    Los .joblib de cuml (Random Forest, SVR, KNN — entrenados con GPU real)
    guardados con una versión anterior de cuml no tienen la clave `_verbose`
    en su estado pickleado, que el `__setstate__` de la version instalada
    (cuml 25.02) espera encontrar. Sin esto, joblib.load() falla con
    `KeyError: '_verbose'` para CUALQUIER estimador de cuml, no solo Random
    Forest. Se aplica a las 3 clases usadas en este proyecto,
    una vez al importar el script. No-op si cuml no está instalado
    (entorno sin GPU/RAPIDS) o si el pickle ya trae `_verbose` (versión
    coincidente).
    """
    try:
        from cuml.ensemble.randomforestregressor import RandomForestRegressor
        from cuml.svm.svm_base import SVMBase
        from cuml.neighbors.kneighbors_regressor import KNeighborsRegressor
    except ImportError:
        return

    for cls in (RandomForestRegressor, SVMBase, KNeighborsRegressor):
        orig_setstate = cls.__setstate__

        def patched_setstate(self, state, _orig=orig_setstate):
            if isinstance(state, dict) and '_verbose' not in state:
                state['_verbose'] = state.get('verbose', False)
            return _orig(self, state)

        cls.__setstate__ = patched_setstate


patch_cuml_setstate_verbose()

warnings.filterwarnings("ignore")

# ==================== CONFIGURACIÓN DE GRÁFICOS ====================
FONT_SIZES = {
    'title': 14,
    'label': 12,
    'tick': 10,
    'legend': 8,
    'text': 9
}

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
    'lines.markersize': 5
})


# ==================== CLASES PARA CARGAR MODELOS PYTORCH ====================
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
        for k, v in kwargs.items():
            setattr(self, k, v)
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


# ==================== RUTAS Y CONFIGURACIÓN ====================
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CSV_FILE = REPO_ROOT / 'data' / 'processed' / 'ifn_entrenamiento_year_agrupado_periodo_completado_2.csv'
ARTIFACTS_ROOT = REPO_ROOT / 'models' / 'artifacts' / 'IFN2-3'  # modelo global (IFN2+IFN3),
                                                                  # ver el marcador
                                                                  # modelos_grid_search_todos_c4_round4.txt
OUTPUT_DIR = REPO_ROOT / 'figures' / 'fig62_publicacion'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_C4 = [
    'especie_id', 'tipo_especie', 'periodo_agrupado', 'radio', 'ocupa',
    'npies_1', 'npies_2', 'npies_5', 'npies_10', 'npies_15', 'npies_20',
    'npies_25', 'npies_30', 'npies_35', 'npies_40', 'npies_45', 'npies_50',
    'npies_55', 'npies_60', 'npies_65', 'npies_70',
    'estado_id', 'fccarb', 'disesp_id', 'modcomb_id', 'nivel2_id', 'tratmasa_id',
    'rocosidad_id', 'orientacion_cat', 'elevacion', 'pendiente_cat', 'martonneidx_id',
    'skt_mean_primavera', 'pr_sum_primavera', 'skt_std_verano', 'skt_std_primavera',
    'pr_sum_otoño', 'skt_mean_verano', 'pr_sum_verano', 'pr_sum_invierno',
    'gndvi_mean_verano', 'ndii_mean_primavera', 'gndvi_std_primavera', 'evi_mean_primavera'
]
FEATURES_CB = [
    'especie_id', 'tipo_especie', 'periodo_agrupado', 'radio', 'ocupa',
    'npies_1', 'npies_2', 'npies_5', 'npies_10', 'npies_15', 'npies_20',
    'npies_25', 'npies_30', 'npies_35', 'npies_40', 'npies_45', 'npies_50',
    'npies_55', 'npies_60', 'npies_65', 'npies_70',
    'estado_id', 'fccarb', 'disesp_id', 'modcomb_id', 'nivel2_id', 'tratmasa_id',
    'rocosidad_id', 'orientacion_cat', 'elevacion', 'pendiente_cat', 'martonneidx_id',
    'gndvi_mean_verano', 'ndii_mean_primavera', 'gndvi_std_primavera', 'evi_mean_primavera'
]
STACK5_BASE_MODELS = ['CatBoost', 'LightGBM', 'XGBoost', 'Random_Forest', 'GBDT', 'BaggedDT']

CONFIGS = [
    {
        'target': 'c4',
        'model_name': 'LightGBM',
        'model_type': 'base',
        'model_path': ARTIFACTS_ROOT / 'c4' / 'base' / 'LightGBM.joblib',
        'features': FEATURES_C4,
        'unit': 't C/ha',
        'density_scale': 300,
        'color': '#3498db'
    },
    {
        'target': 'c4',
        'model_name': 'Stack5_MLP',
        'model_type': 'stacking',
        'model_path': ARTIFACTS_ROOT / 'c4' / 'stackings' / 'Stack5_MLP.joblib',
        'base_models_dir': ARTIFACTS_ROOT / 'c4' / 'base',
        'base_models': STACK5_BASE_MODELS,
        'features': FEATURES_C4,
        'unit': 't C/ha',
        'density_scale': 300,
        'color': '#9b59b6'
    },
    {
        'target': 'carbono_bruto4',
        'model_name': 'CatBoost',
        'model_type': 'base',
        'model_path': ARTIFACTS_ROOT / 'carbono_bruto4' / 'base' / 'CatBoost.joblib',
        'features': FEATURES_CB,
        'unit': 't C',
        'density_scale': 200,
        'color': '#27ae60'
    },
    {
        'target': 'carbono_bruto4',
        'model_name': 'Stack5_MLP',
        'model_type': 'stacking',
        'model_path': ARTIFACTS_ROOT / 'carbono_bruto4' / 'stackings' / 'Stack5_MLP.joblib',
        'base_models_dir': ARTIFACTS_ROOT / 'carbono_bruto4' / 'base',
        'base_models': STACK5_BASE_MODELS,
        'features': FEATURES_CB,
        'unit': 't C',
        'density_scale': 200,
        'color': '#9b59b6'
    },
]

MAPA_PENDIENTE = {'muy suave (<5°)': 0, 'suave (5-10°)': 1, 'moderada (10-15°)': 2,
                  'fuerte (15-20°)': 3, 'muy fuerte (20-30°)': 4, 'escarpada (30-50°)': 5, 'extrema (>50°)': 6}
MAPA_ORIENTACION = {'N': 0, 'NE': 1, 'E': 2, 'SE': 3, 'S': 4, 'SO': 5, 'O': 6, 'NO': 7}
COLOR_IFN2 = '#3498db'
COLOR_IFN3 = '#27ae60'


# ==================== LÓGICA DE DIBUJADO (DRAWING) ====================
def draw_combined_metrics(fig, cell_spec, df_test, y_test, y_pred, df_train, periodos, config):
    """
    Dibuja el gráfico combinado (Hist/RMSE/%RMSE) en un GridSpec específico.
    """
    resultados = []
    cuantiles = []

    for periodo in periodos:
        mask = df_test['periodo_agrupado'] == periodo
        if mask.sum() < 10: continue
        y_real_p = y_test[mask]; y_pred_p = y_pred[mask]

        rmse = np.sqrt(mean_squared_error(y_real_p, y_pred_p))
        mae = mean_absolute_error(y_real_p, y_pred_p)
        media_real = np.mean(y_real_p)

        resultados.append({'Periodo': int(periodo), 'RMSE': rmse, 'MAE': mae, 'Media_Real': media_real})
        errores_abs = np.abs(y_real_p - y_pred_p)
        pct_rmse_ind = (errores_abs / media_real * 100) if media_real > 0 else np.zeros_like(errores_abs)

        for q in [10, 25, 50, 75, 90]:
            cuantiles.append({
                'Periodo': int(periodo), 'Cuantil': q,
                'RMSE_q': np.percentile(errores_abs, q),
                'PCT_RMSE_q': np.percentile(pct_rmse_ind, q)
            })

    df_res = pd.DataFrame(resultados); df_q = pd.DataFrame(cuantiles)

    n_train_por_periodo = df_train['periodo_agrupado'].value_counts().reindex(periodos, fill_value=0)
    df_train = df_train.copy()
    df_train['ifn_id_str'] = df_train['ifn_id'].astype(str)
    n_train_ifn2 = df_train[df_train['ifn_id_str'].str.contains('2')].groupby('periodo_agrupado').size().reindex(periodos, fill_value=0)
    n_train_ifn3 = df_train[df_train['ifn_id_str'].str.contains('3')].groupby('periodo_agrupado').size().reindex(periodos, fill_value=0)
    max_n = n_train_por_periodo.max()
    color = config['color']; unit = config['unit']

    gs_inner = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=cell_spec,
                                                height_ratios=[0.85, 2.4, 2.4], hspace=0.1)

    ax_hist = fig.add_subplot(gs_inner[0])
    ax_rmse = fig.add_subplot(gs_inner[1], sharex=ax_hist)
    ax_pct = fig.add_subplot(gs_inner[2], sharex=ax_hist)

    ax_hist.bar(n_train_ifn2.index, n_train_ifn2.values, color=COLOR_IFN2, edgecolor='black', alpha=0.8, label='NFI2')
    ax_hist.bar(n_train_ifn3.index, n_train_ifn3.values, bottom=n_train_ifn2.values, color=COLOR_IFN3, edgecolor='black', alpha=0.8, label='NFI3')
    ax_hist.set_ylabel('N (Train)', fontsize=FONT_SIZES['label'])
    ax_hist.set_ylim(0, max_n * 1.3)
    ax_hist.legend(loc='upper left', fontsize=FONT_SIZES['legend'], ncol=2, frameon=True, edgecolor='gray')
    ax_hist.grid(True, alpha=0.3, axis='y')
    ax_hist.tick_params(labelbottom=False)

    df_pivot_rmse = df_q.pivot(index='Periodo', columns='Cuantil', values='RMSE_q')
    if len(df_pivot_rmse) > 0:
        periodos_p = df_pivot_rmse.index.values
        ax_rmse.fill_between(periodos_p, df_pivot_rmse[10], df_pivot_rmse[90], color=color, alpha=0.15, label='Q10-Q90')
        ax_rmse.fill_between(periodos_p, df_pivot_rmse[25], df_pivot_rmse[75], color=color, alpha=0.35, label='Q25-Q75')
        ax_rmse.plot(periodos_p, df_pivot_rmse[50], 'o-', color=color, linewidth=2, markersize=5, label='Median')
        if len(df_res) > 0:
            ax_rmse.plot(df_res['Periodo'], df_res['MAE'], 'x--', color='black', linewidth=1.5, markersize=5, label='MAE')
            ax_rmse.plot(df_res['Periodo'], df_res['RMSE'], '^-', color='purple', linewidth=1.5, markersize=5, label='RMSE')
    ax_rmse.set_ylabel(f'Absolute Error ({unit})', fontsize=FONT_SIZES['label'])
    ax_rmse.set_ylim(bottom=0)
    ax_rmse.legend(loc='upper left', fontsize=FONT_SIZES['legend'], frameon=True, edgecolor='gray', ncol=2)
    ax_rmse.grid(True, alpha=0.3)
    ax_rmse.tick_params(labelbottom=False)

    df_pivot_pct = df_q.pivot(index='Periodo', columns='Cuantil', values='PCT_RMSE_q')
    if len(df_pivot_pct) > 0:
        periodos_p = df_pivot_pct.index.values
        ax_pct.fill_between(periodos_p, df_pivot_pct[10], df_pivot_pct[90], color=color, alpha=0.15, label='Q10-Q90')
        ax_pct.fill_between(periodos_p, df_pivot_pct[25], df_pivot_pct[75], color=color, alpha=0.35, label='Q25-Q75')
        ax_pct.plot(periodos_p, df_pivot_pct[50], 'o-', color=color, linewidth=2, markersize=5, label='Median')
        if len(df_res) > 0:
            mean_pct = (df_res['MAE'] / df_res['Media_Real']) * 100
            ax_pct.plot(df_res['Periodo'], mean_pct, 'x--', color='black', linewidth=1.5, markersize=5, label='%MAE')
            rmse_pct = (df_res['RMSE'] / df_res['Media_Real']) * 100
            ax_pct.plot(df_res['Periodo'], rmse_pct, '^-', color='purple', linewidth=1.5, markersize=5, label='%RMSE')
    ax_pct.set_ylabel('%Absolute Error', fontsize=FONT_SIZES['label'])
    ax_pct.set_ylim(bottom=0)
    ax_pct.set_xlabel('Period (years)', fontsize=FONT_SIZES['label'])

    periodos_mostrar = [5, 10, 15, 20, 25, 30]
    ax_pct.set_xticks(periodos_mostrar)
    ax_pct.set_xlim(min(periodos) - 0.5, max(periodos) + 0.5)

    ax_pct.legend(loc='upper left', fontsize=FONT_SIZES['legend'], frameon=True, edgecolor='gray')
    ax_pct.grid(True, alpha=0.3)


def draw_density(fig, cell_spec, y_real, y_pred, unit, max_escala):
    """
    Dibuja gráfico de densidad en un GridSpec específico.

    Eje X=Predicted, eje Y=Observed + test de sesgo H0:a=0 / H0:b=1,
    con KDE de densidad, histogramas marginales de Count y colorbar
    Density %.
    """
    # 1. Preparar datos
    mask_vis = (y_real <= max_escala) & (y_pred <= max_escala) & (y_real >= 0) & (y_pred >= 0)
    y_r_vis = y_real[mask_vis]; y_p_vis = y_pred[mask_vis]
    if len(y_r_vis) > 5000:
        # Semilla fija para que la sub-muestra de 5000 puntos dibujada sea
        # reproducible entre ejecuciones (la sub-muestra solo afecta al
        # aspecto de la nube de puntos, no al test de sesgo, que usa todos
        # los puntos). Versiones anteriores de la figura usaron una
        # sub-muestra sin semilla, por lo que los puntos dibujados pueden
        # diferir ligeramente.
        idx = np.random.default_rng(42).choice(len(y_r_vis), 5000, replace=False)
        y_r_sample = y_r_vis[idx]; y_p_sample = y_p_vis[idx]
    else:
        y_r_sample = y_r_vis; y_p_sample = y_p_vis

    # Calcular densidad
    try:
        xy = np.vstack([y_r_sample, y_p_sample])
        z = gaussian_kde(xy)(xy)
        z_pct = np.array([percentileofscore(z, d) for d in z])
        idx_sorted = z_pct.argsort()
        y_r_s = y_r_sample[idx_sorted]; y_p_s = y_p_sample[idx_sorted]; c_s = z_pct[idx_sorted]
    except Exception:
        y_r_s = y_r_sample; y_p_s = y_p_sample; c_s = 'blue'

    # 2. Configurar GridSpec anidado
    gs_inner = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=cell_spec,
                                                width_ratios=[4, 1.2, 0.2], height_ratios=[1, 4],
                                                wspace=0.15, hspace=0.1)

    ax_main = fig.add_subplot(gs_inner[1, 0])
    ax_histx = fig.add_subplot(gs_inner[0, 0], sharex=ax_main)
    ax_histy = fig.add_subplot(gs_inner[1, 1], sharey=ax_main)
    ax_cbar  = fig.add_subplot(gs_inner[1, 2])

    # 3. Main Plot: X=Predicted (y_p_s), Y=Observed (y_r_s).
    scatter = ax_main.scatter(y_p_s, y_r_s, c=c_s, s=8, cmap='viridis', alpha=0.7, vmin=0, vmax=100)
    ax_main.plot([0, max_escala], [0, max_escala], 'k--', lw=1.5, label='Perfect', alpha=0.8)

    # Test de sesgo (H0: a=0, H0: b=1) sobre todos los puntos visibles, ver
    # scripts/revision/regression_scatter.py. El fit se hace sobre y_r_vis/
    # y_p_vis (todos los puntos filtrados, no solo la sub-muestra de 5000
    # usada para la densidad) para máxima potencia estadística.
    #
    # Este fit restringido al rango [0, max_escala] es el que se DIBUJA
    # (para que la línea sea coherente con la nube de puntos visible). El
    # resultado sobre el test set completo sin recortar se calcula aparte en
    # `result['full_range']` (más abajo) y ambos se guardan en bias_test.csv. Un
    # puñado de outliers extremos (fuera de [0, max_escala]) puede tener
    # mucho leverage en la OLS y cambiar la conclusión de sesgo.
    result = test_unbiased_fit(y_r_vis, y_p_vis)
    ax_main.plot([0, max_escala],
                 [result['intercept'], result['slope'] * max_escala + result['intercept']],
                 'b-', lw=2, alpha=0.9,
                 label=f"Trend: a={result['intercept']:.2f}, b={result['slope']:.2f}")

    n_total = len(y_real)
    n_excluded = int(n_total - mask_vis.sum())
    result['n_total'] = n_total
    result['n_excluded'] = n_excluded
    result['pct_excluded'] = 100.0 * n_excluded / n_total
    result['full_range'] = test_unbiased_fit(y_real, y_pred)

    ax_main.set_xlabel(f'Predicted ({unit})', fontsize=FONT_SIZES['label'])
    ax_main.set_ylabel(f'Observed ({unit})', fontsize=FONT_SIZES['label'])
    ax_main.set_xlim(0, max_escala)
    ax_main.set_ylim(0, max_escala)
    ax_main.legend(loc='upper left', fontsize=FONT_SIZES['legend'], frameon=True, edgecolor='gray')
    ax_main.grid(True, alpha=0.3)

    def fmt_p(p):
        # p<0.001 en vez de "0.000": un p-valor nunca es exactamente cero
        # (aqui, por ejemplo, ~1e-6), y mostrarlo redondeado a 0.000 puede
        # leerse como si lo fuera.
        return "p<0.001" if p < 0.001 else f"p={p:.3f}"

    verdict = 'unbiased' if result['unbiased'] else 'biased'
    ax_main.text(
        0.98, 0.02,
        f"a=0: {fmt_p(result['p_intercept_vs_0'])}\n"
        f"b=1: {fmt_p(result['p_slope_vs_1'])}\n({verdict})",
        transform=ax_main.transAxes, fontsize=FONT_SIZES['legend'], va='bottom', ha='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    )

    # 4. Histograma X (Arriba): eje Predicted -> y_p_sample
    bins_hist = np.linspace(0, max_escala, 30)
    ax_histx.hist(y_p_sample, bins=bins_hist, color='#3498db', alpha=0.7, edgecolor='white')
    ax_histx.tick_params(labelbottom=False, labelsize=8)
    ax_histx.set_ylabel('Count', fontsize=9)
    ax_histx.grid(True, alpha=0.3, axis='x')

    # 5. Histograma Y (Derecha): eje Observed -> y_r_sample
    ax_histy.hist(y_r_sample, bins=bins_hist, orientation='horizontal', color='#e74c3c', alpha=0.7, edgecolor='white')
    ax_histy.tick_params(labelleft=False, labelsize=8)
    ax_histy.set_xlabel('Count', fontsize=9)
    ax_histy.grid(True, alpha=0.3, axis='y')

    # 6. Colorbar
    cbar = fig.colorbar(scatter, cax=ax_cbar, orientation='vertical')
    cbar.set_label('Density %', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    return result


# ==================== CARGAR DATOS GENERALES ====================
if __name__ == '__main__':
    print("📂 Cargando datos globales...")
    df_base = pd.read_csv(CSV_FILE)
    df_base['pendiente_cat'] = df_base['pendiente_cat'].map(MAPA_PENDIENTE)
    df_base['orientacion_cat'] = df_base['orientacion_cat'].map(MAPA_ORIENTACION)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', choices=['base', 'stacking', 'all'], default='all')
    args = parser.parse_args()
    wanted_types = {'base': ['base'], 'stacking': ['stacking'], 'all': ['base', 'stacking']}[args.models]

    bias_results = {}

    # ==================== PROCESO PRINCIPAL ====================
    for config in CONFIGS:
        if config['model_type'] not in wanted_types:
            continue
        target = config['target']
        model_name = config['model_name']
        model_type = config['model_type']
        print(f"\n📌 Procesando {target} - {model_name}...")

        # Preparar datos (mismos filtros y split que en entrenamiento)
        df = df_base[(df_base[target].notna()) & (df_base[target] != 0) & ((df_base['c'].isna()) | (df_base['c'] < df_base[target]))].copy()
        df = df[~(df['fccarb'].notna() & (df['fccarb'] < 20))].copy()
        df = df[df['periodo_agrupado'].notna()].copy()
        features = [c for c in config['features'] if c in df.columns]
        X_full = df[features].copy().astype('float32'); y_full = df[target].values

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X_full, y_full, groups=df['parcela_id']))
        df_train = df.iloc[train_idx]; df_test = df.iloc[test_idx]
        X_test = X_full.iloc[test_idx]; y_test = y_full[test_idx]
        periodos = sorted([p for p in df_test['periodo_agrupado'].unique() if p > 0])

        # Predecir (con compatibilidad de atributos legacy de XGBoost)
        try:
            if model_type == 'base':
                modelo = joblib.load(config['model_path'])
                patch_xgboost_legacy_attrs(modelo)
                y_pred = modelo.predict(X_test)
            else:
                base_preds = []
                for b_name in config['base_models']:
                    b_model = joblib.load(config['base_models_dir'] / f"{b_name}.joblib")
                    patch_xgboost_legacy_attrs(b_model)
                    if hasattr(b_model, 'verbose'):
                        b_model.verbose = 0
                    base_preds.append(b_model.predict(X_test))
                    del b_model
                    gc.collect()
                X_stack = np.column_stack(base_preds)
                meta = joblib.load(config['model_path'])
                y_pred = meta.predict(X_stack)
        except Exception as e:
            print(f"❌ Error modelo: {e}"); continue

        # ==================== GENERACIÓN DE FIGURAS ====================
        FIG_SIZE_IND = (6, 6)

        # 1. Figura Individual: Métricas
        fig1 = plt.figure(figsize=FIG_SIZE_IND)
        gs1 = gridspec.GridSpec(1, 1, figure=fig1)
        draw_combined_metrics(fig1, gs1[0], df_test, y_test, y_pred, df_train, periodos, config)
        out1 = OUTPUT_DIR / f"{target}_{model_name}_metrics"
        fig1.savefig(f"{out1}.png", bbox_inches='tight')
        fig1.savefig(f"{out1}.pdf", bbox_inches='tight')
        plt.close(fig1)
        print(f"   💾 {out1}")

        # 2. Figura Individual: Densidad (la que se usa en Fig 6.2)
        fig2 = plt.figure(figsize=FIG_SIZE_IND)
        gs2 = gridspec.GridSpec(1, 1, figure=fig2)
        result = draw_density(fig2, gs2[0], y_test, y_pred, config['unit'], config['density_scale'])
        bias_results[f"{target}_{model_name}"] = result
        out2 = OUTPUT_DIR / f"{target}_{model_name}_density"
        fig2.savefig(f"{out2}.png", bbox_inches='tight')
        fig2.savefig(f"{out2}.pdf", bbox_inches='tight')
        plt.close(fig2)
        print(f"   💾 {out2}")
        print(f"   [rango mostrado, n={result['n_total'] - result['n_excluded']}] "
              f"a={result['intercept']:.3f} (p={result['p_intercept_vs_0']:.4f}), "
              f"b={result['slope']:.3f} (p={result['p_slope_vs_1']:.4f}) "
              f"-> {'UNBIASED' if result['unbiased'] else 'BIASED'}")
        rf = result['full_range']
        print(f"   [test set COMPLETO, n={result['n_total']}, "
              f"{result['pct_excluded']:.2f}% excluido del rango mostrado] "
              f"a={rf['intercept']:.3f} (p={rf['p_intercept_vs_0']:.4f}), "
              f"b={rf['slope']:.3f} (p={rf['p_slope_vs_1']:.4f}) "
              f"-> {'UNBIASED' if rf['unbiased'] else 'BIASED'}")

        # Fig 6.2 usa solo el panel de densidad (`{target}_{model_name}_
        # density.png/pdf`, punto 2 arriba); el panel de métricas (error vs.
        # periodo) es una figura distinta, por lo que no se genera una figura
        # combinada metrics|density.

        del df, X_test, y_test, y_pred
        gc.collect()

    print("\n=== RESUMEN DEL TEST DE SESGO (Fig 6.2) ===")
    for name, r in bias_results.items():
        verdict = 'UNBIASED' if r['unbiased'] else 'BIASED'
        rf = r['full_range']
        verdict_full = 'UNBIASED' if rf['unbiased'] else 'BIASED'
        print(f"{name} [rango mostrado, n={r['n_total'] - r['n_excluded']}]: "
              f"a={r['intercept']:.3f} (p={r['p_intercept_vs_0']:.4f}), "
              f"b={r['slope']:.3f} (p={r['p_slope_vs_1']:.4f}) -> {verdict}")
        print(f"{name} [TEST SET COMPLETO, n={r['n_total']}, {r['pct_excluded']:.2f}% excluido]: "
              f"a={rf['intercept']:.3f} (p={rf['p_intercept_vs_0']:.4f}), "
              f"b={rf['slope']:.3f} (p={rf['p_slope_vs_1']:.4f}) -> {verdict_full}")

    rows = []
    for name, r in bias_results.items():
        for scope, res in (('displayed_range', r), ('full_test_set', r['full_range'])):
            rows.append({'model': name, 'scope': scope,
                         **{k: v for k, v in res.items() if not isinstance(v, dict)}})
    if rows:
        bias_csv = OUTPUT_DIR / 'bias_test.csv'
        pd.DataFrame(rows).to_csv(bias_csv, index=False)
        print(f"   💾 {bias_csv}")
