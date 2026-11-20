# -*- coding: utf-8 -*-
"""
Generador de Estadísticas a Múltiples Escalas
Calcula R², RMSE, MAE, wMAPE, SMAPE, MAPE, Bias y Moda_Abs
para todos los modelos y rangos de valores.
"""
import numpy as np
import pandas as pd
import joblib
import os
from pathlib import Path
import warnings
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GroupShuffleSplit
from tabulate import tabulate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ==================== RUTAS (relativas al repositorio) ====================
REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "models" / "artifacts" / "IFN2-3"

# ==================== FUNCIONES AUXILIARES ====================
def cast_to_str(X):
    """Convierte la entrada a string para el OneHotEncoder."""
    return pd.DataFrame(X).astype(str).values

def make_ohe():
    """Crea el encoder compatible con distintas versiones de sklearn."""
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)

# ==================== ESCALAS A EVALUAR ====================
ESCALAS = [20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800]

# ==================== CLASES PYTORCH (para cargar modelos) ====================
class PyTorchMLPDropout(nn.Module):
    """MLP con Dropout y BatchNorm para regularización adicional."""
    def __init__(self, input_dim, hidden_layer_sizes=(100,), activation='relu', 
                 dropout_rate=0.2, use_batchnorm=True):
        super(PyTorchMLPDropout, self).__init__()
        
        layers = []
        in_dim = input_dim
        
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'tanh':
            act_fn = nn.Tanh
        elif activation == 'leaky_relu':
            act_fn = lambda: nn.LeakyReLU(0.1)
        else:
            act_fn = nn.ReLU
        
        for i, h_dim in enumerate(hidden_layer_sizes):
            layers.append(nn.Linear(in_dim, h_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(act_fn())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, 1))
        self.model = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.model(x)

# Alias para compatibilidad con modelos antiguos
PyTorchMLP = PyTorchMLPDropout

class MLPRegressorTorch(BaseEstimator, RegressorMixin):
    """
    Wrapper sklearn-compatible para MLP con Dropout y BatchNorm.
    Parámetros adicionales: dropout_rate, use_batchnorm
    """
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='adam', 
                 alpha=0.0001, learning_rate_init=0.001, max_iter=200, batch_size=256,
                 dropout_rate=0.2, use_batchnorm=True,
                 device='cuda' if torch.cuda.is_available() else 'cpu', verbose=False,
                 X_val=None, y_val=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.use_batchnorm = use_batchnorm
        self.device = device
        self.verbose = verbose
        self.X_val = X_val
        self.y_val = y_val
        self.model_ = None
        self.history_ = {'loss': [], 'train_r2': [], 'val_r2': [], 'epochs': []}
        
    def set_validation_data(self, X_val, y_val):
        self.X_val = X_val
        self.y_val = y_val
    
    def fit(self, X, y):
        # fit vacío: esta clase solo se usa para cargar modelos ya entrenados
        return self

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("Model not fitted")
            
        if isinstance(X, pd.DataFrame): X = X.values
        if hasattr(X, 'values'): X = X.values
        
        # Asegurar device
        if not hasattr(self, 'device') or self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        self.model_.eval()
        
        # Mover modelo a device si no está (CRÍTICO)
        self.model_.to(self.device)
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            preds = self.model_(X_tensor).cpu().numpy().flatten()
            
        return preds

# Alias para compatibilidad con modelos guardados
PyTorchMLP = PyTorchMLPDropout

# ==================== CONFIGURACIÓN ====================
ruta_csv = str(REPO_ROOT / "data" / "processed" / "ifn_entrenamiento_year_agrupado_periodo_completado_2.csv")
output_dir = str(REPO_ROOT / "data" / "estadisticas_por_rango")

MAPA_PENDIENTE = {'muy suave (<5°)': 0, 'suave (5-10°)': 1, 'moderada (10-15°)': 2,
                  'fuerte (15-20°)': 3, 'muy fuerte (20-30°)': 4, 'escarpada (30-50°)': 5, 'extrema (>50°)': 6}
MAPA_ORIENTACION = {'N': 0, 'NE': 1, 'E': 2, 'SE': 3, 'S': 4, 'SO': 5, 'O': 6, 'NO': 7}

# Features por target
features_c4 = [
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

features_carbono_bruto4 = [
    'especie_id', 'tipo_especie', 'periodo_agrupado', 'radio', 'ocupa',
    'npies_1', 'npies_2', 'npies_5', 'npies_10', 'npies_15', 'npies_20',
    'npies_25', 'npies_30', 'npies_35', 'npies_40', 'npies_45', 'npies_50',
    'npies_55', 'npies_60', 'npies_65', 'npies_70',
    'estado_id', 'fccarb', 'disesp_id', 'modcomb_id', 'nivel2_id', 'tratmasa_id',
    'rocosidad_id', 'orientacion_cat', 'elevacion', 'pendiente_cat', 'martonneidx_id',
    'gndvi_mean_verano', 'ndii_mean_primavera', 'gndvi_std_primavera', 'evi_mean_primavera'
]

# Configuración de modelos
base_c4_dir = str(ARTIFACTS_DIR / 'c4' / 'base')
base_cb4_dir = str(ARTIFACTS_DIR / 'carbono_bruto4' / 'base')
stack_c4_dir = str(ARTIFACTS_DIR / 'c4' / 'stackings')
stack_cb4_dir = str(ARTIFACTS_DIR / 'carbono_bruto4' / 'stackings')

base_models = ['BaggedDT', 'BayesianNN', 'CatBoost', 'GBDT', 'LightGBM', 
               'MLP_PyTorch', 'Random_Forest', 'SVR', 'XGBoost']
meta_learners = ['GradientBoosting', 'LinearRegression', 'MLP', 'RandomForest', 'Ridge', 'SVR']

STACK_CONFIGS = [
    ['LightGBM', 'Random_Forest'],
    ['LightGBM', 'XGBoost', 'GBDT'],
    ['CatBoost', 'Random_Forest', 'GBDT'],
    ['CatBoost', 'LightGBM', 'Random_Forest', 'GBDT'],
    ['CatBoost', 'LightGBM', 'XGBoost', 'Random_Forest', 'GBDT', 'BaggedDT'],
]

# Construir lista de configuraciones
configs = []

# Modelos Base C4
for model_name in base_models:
    configs.append({
        'target': 'c4',
        'model_name': model_name,
        'model_path': f'{base_c4_dir}/{model_name}.joblib',
        'features': features_c4,
        'model_type': 'base'
    })

# Modelos Base Carbono Bruto 4
for model_name in base_models:
    configs.append({
        'target': 'carbono_bruto4',
        'model_name': model_name,
        'model_path': f'{base_cb4_dir}/{model_name}.joblib',
        'features': features_carbono_bruto4,
        'model_type': 'base'
    })

# Stacking C4
for stack_idx, base_names in enumerate(STACK_CONFIGS, start=1):
    for meta in meta_learners:
        model_name = f'Stack{stack_idx}_{meta}'
        configs.append({
            'target': 'c4',
            'model_name': model_name,
            'model_path': f'{stack_c4_dir}/{model_name}.joblib',
            'features': features_c4,
            'model_type': 'stacking',
            'base_models_dir': base_c4_dir,
            'base_models': base_names
        })

# Stacking Carbono Bruto 4
for stack_idx, base_names in enumerate(STACK_CONFIGS, start=1):
    for meta in meta_learners:
        model_name = f'Stack{stack_idx}_{meta}'
        configs.append({
            'target': 'carbono_bruto4',
            'model_name': model_name,
            'model_path': f'{stack_cb4_dir}/{model_name}.joblib',
            'features': features_carbono_bruto4,
            'model_type': 'stacking',
            'base_models_dir': base_cb4_dir,
            'base_models': base_names
        })

# ==================== FUNCIÓN DE MÉTRICAS ====================
def calcular_metricas_rango(y_true, y_pred, max_val=None, min_val=0):
    """
    Calcula todas las métricas para datos dentro del rango [min_val, max_val].
    Si max_val es None, usa todos los datos >= min_val (sin límite superior).
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Aplicar filtros según los parámetros
    if max_val is not None:
        # Rango cerrado: [min_val, max_val]
        mask = (y_true >= min_val) & (y_true <= max_val)
    else:
        # Solo límite inferior: [min_val, +inf)
        mask = (y_true >= min_val)
    
    n_total = len(y_true)
    n_rango = mask.sum()
    
    if n_rango < 10:
        return {
            'N_total': n_total,
            'N_rango': n_rango,
            'R2': np.nan,
            'RMSE': np.nan,
            'MAE': np.nan,
            'wMAPE': np.nan,
            'SMAPE': np.nan,
            'MAPE': np.nan,
            'Bias': np.nan,
            'Moda_Abs': np.nan
        }
    
    y_t = y_true[mask]
    y_p = y_pred[mask]
    
    errores = y_t - y_p
    errores_abs = np.abs(errores)
    
    # Métricas estándar
    mae = mean_absolute_error(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    r2 = r2_score(y_t, y_p)
    bias = np.mean(errores)
    
    # MAPE (Clásico) - evitar división por 0
    y_t_safe = np.where(y_t == 0, 1e-6, y_t)
    pct_error = (errores / y_t_safe) * 100
    mape = np.mean(np.abs(pct_error))
    
    # wMAPE (Ponderado)
    if np.sum(y_t) != 0:
        wmape = (np.sum(errores_abs) / np.sum(y_t)) * 100
    else:
        wmape = 0.0
    
    # SMAPE (Simétrico)
    numerador = np.abs(y_t - y_p)
    denominador = np.abs(y_t) + np.abs(y_p) + 1e-6
    smape_vector = 200 * numerador / denominador
    smape = np.mean(smape_vector)
    
    # Moda Absoluta
    errores_abs_round = np.round(errores_abs, 0).astype(int)
    vals, counts = np.unique(errores_abs_round, return_counts=True)
    try:
        moda_abs = float(vals[np.argmax(counts)])
    except:
        moda_abs = 0.0
    
    return {
        'N_total': n_total,
        'N_rango': n_rango,
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'wMAPE': wmape,
        'SMAPE': smape,
        'MAPE': mape,
        'Bias': bias,
        'Moda_Abs': moda_abs
    }

# ==================== ESCALAS A EVALUAR ====================
ESCALAS_C4 = [20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800]
ESCALAS_CB4 = [20, 40, 70, 100, 150, 200, 300, 400, 500, 600, 700, 800] # Personalizado para 6 bines hasta 200

# ==================== CLASES PYTORCH (para cargar modelos) ====================
# PyTorchMLPDropout se define más arriba.

# ==================== CONFIGURACIÓN ====================
ruta_csv = str(REPO_ROOT / "data" / "processed" / "ifn_entrenamiento_year_agrupado_periodo_completado_2.csv")
output_dir = str(REPO_ROOT / "data" / "estadisticas_por_rango")

MAPA_PENDIENTE = {'muy suave (<5°)': 0, 'suave (5-10°)': 1, 'moderada (10-15°)': 2,
                  'fuerte (15-20°)': 3, 'muy fuerte (20-30°)': 4, 'escarpada (30-50°)': 5, 'extrema (>50°)': 6}
MAPA_ORIENTACION = {'N': 0, 'NE': 1, 'E': 2, 'SE': 3, 'S': 4, 'SO': 5, 'O': 6, 'NO': 7}

# Features por target
features_c4 = [
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

features_carbono_bruto4 = [
    'especie_id', 'tipo_especie', 'periodo_agrupado', 'radio', 'ocupa',
    'npies_1', 'npies_2', 'npies_5', 'npies_10', 'npies_15', 'npies_20',
    'npies_25', 'npies_30', 'npies_35', 'npies_40', 'npies_45', 'npies_50',
    'npies_55', 'npies_60', 'npies_65', 'npies_70',
    'estado_id', 'fccarb', 'disesp_id', 'modcomb_id', 'nivel2_id', 'tratmasa_id',
    'rocosidad_id', 'orientacion_cat', 'elevacion', 'pendiente_cat', 'martonneidx_id',
    'gndvi_mean_verano', 'ndii_mean_primavera', 'gndvi_std_primavera', 'evi_mean_primavera'
]

# Configuración de modelos
base_c4_dir = str(ARTIFACTS_DIR / 'c4' / 'base')
base_cb4_dir = str(ARTIFACTS_DIR / 'carbono_bruto4' / 'base')
stack_c4_dir = str(ARTIFACTS_DIR / 'c4' / 'stackings')
stack_cb4_dir = str(ARTIFACTS_DIR / 'carbono_bruto4' / 'stackings')

base_models = ['BaggedDT', 'BayesianNN', 'CatBoost', 'GBDT', 'LightGBM', 
               'MLP_PyTorch', 'Random_Forest', 'SVR', 'XGBoost']
meta_learners = ['GradientBoosting', 'LinearRegression', 'MLP', 'RandomForest', 'Ridge', 'SVR']

STACK_CONFIGS = [
    ['LightGBM', 'Random_Forest'],
    ['LightGBM', 'XGBoost', 'GBDT'],
    ['CatBoost', 'Random_Forest', 'GBDT'],
    ['CatBoost', 'LightGBM', 'XGBoost', 'Random_Forest', 'GBDT', 'BaggedDT'],
]

# Construir lista de configuraciones
configs = []

# Modelos Base C4
for model_name in base_models:
    configs.append({
        'target': 'c4',
        'model_name': model_name,
        'model_path': f'{base_c4_dir}/{model_name}.joblib',
        'features': features_c4,
        'model_type': 'base'
    })

# Modelos Base Carbono Bruto 4
for model_name in base_models:
    configs.append({
        'target': 'carbono_bruto4',
        'model_name': model_name,
        'model_path': f'{base_cb4_dir}/{model_name}.joblib',
        'features': features_carbono_bruto4,
        'model_type': 'base'
    })

# Stacking C4
for stack_idx, base_names in enumerate(STACK_CONFIGS, start=1):
    for meta in meta_learners:
        model_name = f'Stack{stack_idx}_{meta}'
        configs.append({
            'target': 'c4',
            'model_name': model_name,
            'model_path': f'{stack_c4_dir}/{model_name}.joblib',
            'features': features_c4,
            'model_type': 'stacking',
            'base_models_dir': base_c4_dir,
            'base_models': base_names
        })

# Stacking Carbono Bruto 4
for stack_idx, base_names in enumerate(STACK_CONFIGS, start=1):
    for meta in meta_learners:
        model_name = f'Stack{stack_idx}_{meta}'
        configs.append({
            'target': 'carbono_bruto4',
            'model_name': model_name,
            'model_path': f'{stack_cb4_dir}/{model_name}.joblib',
            'features': features_carbono_bruto4,
            'model_type': 'stacking',
            'base_models_dir': base_cb4_dir,
            'base_models': base_names
        })

# The published CSVs (data/estadisticas_por_rango/) contain the 9 global base
# models only: the stacking folders referenced by the original run
# (modelos_stacking_*_round4_v2) were not available, so those configurations
# were skipped. Set INCLUDE_STACKING = True to add the global stacking models.
INCLUDE_STACKING = False
if not INCLUDE_STACKING:
    configs = [c for c in configs if c['model_type'] == 'base']

# ==================== EJECUCIÓN PRINCIPAL ====================
print("=" * 80)
print("📊 GENERACIÓN DE ESTADÍSTICAS A MÚLTIPLES ESCALAS")
print("=" * 80)
# print(f"Escalas a evaluar: {ESCALAS}") -> Variable por target
print(f"Total de modelos: {len(configs)}")
print("=" * 80)

# Crear directorio de salida
os.makedirs(output_dir, exist_ok=True)

# Cargar datos
print("\n📂 Cargando datos...")
df = pd.read_csv(ruta_csv)
df['pendiente_cat'] = df['pendiente_cat'].map(MAPA_PENDIENTE)
df['orientacion_cat'] = df['orientacion_cat'].map(MAPA_ORIENTACION)
print(f"   Shape original: {df.shape}")

# Almacenar resultados
all_results = []

for idx, config in enumerate(configs, 1):
    target = config['target']
    model_name = config['model_name']
    model_path = config['model_path']
    features = config['features']
    model_type = config.get('model_type', 'base')
    
    print(f"\n[{idx}/{len(configs)}] {target.upper()} - {model_name} ({model_type})")
    
    # Seleccionar escala adecuada
    if target == 'c4':
        ESCALAS = ESCALAS_C4
    else:
        ESCALAS = ESCALAS_CB4
            
    # Verificar que existe el modelo
    if not os.path.exists(model_path):
        print(f"   ❌ Modelo no encontrado: {model_path}")
        continue
    
    try:
        # Filtrar datos
        df_work = df[(df[target].notna()) & (df[target] != 0)].copy()
        df_work = df_work[(df_work['c'].isna()) | (df_work['c'] < df_work[target])].copy()
        df_work = df_work[~(df_work['fccarb'].notna() & (df_work['fccarb'] < 20))].copy()
        
        # Usar split de test (mismo que entrenamiento)
        present_cols = [c for c in features if c in df_work.columns]
        X = df_work[present_cols].astype('float32')
        y = df_work[target].astype('float32')
        
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        _, te_idx = next(gss.split(X, y, groups=df_work['parcela_id']))
        X_test = X.iloc[te_idx]
        y_test = y.iloc[te_idx].values
        
        # Generar predicciones según el tipo de modelo
        if model_type == 'stacking':
            base_models_dir = config['base_models_dir']
            base_model_names = config['base_models']
            
            base_predictions = []
            for base_name in base_model_names:
                base_path = f"{base_models_dir}/{base_name}.joblib"
                if os.path.exists(base_path):
                    base_model = joblib.load(base_path)
                    base_pred = base_model.predict(X_test)
                    base_predictions.append(base_pred)
                else:
                    print(f"   ⚠️ Base model no encontrado: {base_name}")
            
            if len(base_predictions) != len(base_model_names):
                print(f"   ❌ Faltan modelos base, saltando...")
                continue
            
            X_stack = np.column_stack(base_predictions)
            stacking_model = joblib.load(model_path)
            y_pred = stacking_model.predict(X_stack)
        else:
            model = joblib.load(model_path)
            y_pred = model.predict(X_test)
        
        # ===============================================
        # TIPO 1: Rangos Acumulados desde 0 (0-20, 0-50, 0-100, ...)
        # ===============================================
        print("   📈 Rangos Acumulados (desde 0):")
        for escala in ESCALAS:
            metrics = calcular_metricas_rango(y_test, y_pred, max_val=escala, min_val=0)
            metrics['Target'] = target
            metrics['Modelo'] = model_name
            metrics['Tipo'] = model_type
            metrics['TipoRango'] = 'Acumulado'
            metrics['Rango'] = f'0-{escala}'
            all_results.append(metrics)
            if not np.isnan(metrics['R2']):
                print(f"      [0-{escala}]: R²={metrics['R2']:.4f} | RMSE={metrics['RMSE']:.2f} | MAE={metrics['MAE']:.2f} | wMAPE={metrics['wMAPE']:.2f}% | SMAPE={metrics['SMAPE']:.2f}% | Bias={metrics['Bias']:.2f} | Moda={metrics['Moda_Abs']:.0f} | n={metrics['N_rango']}")
        
        # Rango completo
        metrics = calcular_metricas_rango(y_test, y_pred, max_val=None, min_val=0)
        metrics['Target'] = target
        metrics['Modelo'] = model_name
        metrics['Tipo'] = model_type
        metrics['TipoRango'] = 'Acumulado'
        metrics['Rango'] = 'Completo'
        all_results.append(metrics)
        print(f"      [Completo]: R²={metrics['R2']:.4f} | RMSE={metrics['RMSE']:.2f} | MAE={metrics['MAE']:.2f} | wMAPE={metrics['wMAPE']:.2f}% | SMAPE={metrics['SMAPE']:.2f}% | Bias={metrics['Bias']:.2f} | Moda={metrics['Moda_Abs']:.0f} | n={metrics['N_rango']}")
        
        # ===============================================
        # TIPO 2: Bins Separados (0-20, 20-50, 50-100, ...)
        # ===============================================
        print("   📊 Bins Separados:")
        bins = [0] + ESCALAS  # [0, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800]
        for i in range(len(bins) - 1):
            min_bin = bins[i]
            max_bin = bins[i + 1]
            
            metrics = calcular_metricas_rango(y_test, y_pred, max_val=max_bin, min_val=min_bin)
            metrics['Target'] = target
            metrics['Modelo'] = model_name
            metrics['Tipo'] = model_type
            metrics['TipoRango'] = 'Bin'
            metrics['Rango'] = f'{min_bin}-{max_bin}'
            all_results.append(metrics)
            
            if not np.isnan(metrics['R2']):
                print(f"      [{min_bin}-{max_bin}]: R²={metrics['R2']:.4f} | RMSE={metrics['RMSE']:.2f} | MAE={metrics['MAE']:.2f} | wMAPE={metrics['wMAPE']:.2f}% | SMAPE={metrics['SMAPE']:.2f}% | Bias={metrics['Bias']:.2f} | Moda={metrics['Moda_Abs']:.0f} | n={metrics['N_rango']}")
            else:
                print(f"      [{min_bin}-{max_bin}]: ⚠️ Pocos datos (n={metrics['N_rango']})")
        
        # Bin final: desde el último límite hasta infinito (>800)
        metrics = calcular_metricas_rango(y_test, y_pred, max_val=None, min_val=bins[-1])
        metrics['Target'] = target
        metrics['Modelo'] = model_name
        metrics['Tipo'] = model_type
        metrics['TipoRango'] = 'Bin'
        metrics['Rango'] = f'{bins[-1]}+'
        all_results.append(metrics)
        if not np.isnan(metrics['R2']):
            print(f"      [{bins[-1]}+]: R²={metrics['R2']:.4f} | RMSE={metrics['RMSE']:.2f} | MAE={metrics['MAE']:.2f} | wMAPE={metrics['wMAPE']:.2f}% | SMAPE={metrics['SMAPE']:.2f}% | Bias={metrics['Bias']:.2f} | Moda={metrics['Moda_Abs']:.0f} | n={metrics['N_rango']}")
        else:
            print(f"      [{bins[-1]}+]: ⚠️ Pocos datos (n={metrics['N_rango']})")
            
        # ===============================================
        # TIPO 3: Percentiles (Basados en distribución completa del dataset filtrado)
        # ===============================================
        print("   📊 Percentiles (Cortes del dataset completo):")
        # Usamos df_work[target] (todos los datos válidos para este target)
        y_ref = df_work[target].dropna().values
        quantiles = np.linspace(0, 1, 11) # 0, 0.1, ... 1.0
        cortes = np.quantile(y_ref, quantiles)
        
        # Asegurar cobertura total en extremos
        cortes[0] = -np.inf
        cortes[-1] = np.inf
        
        labels_pct = [f'P{int(q*100):02d}-P{int((q+0.1)*100):02d}' for q in quantiles[:-1]]
        
        for i in range(len(cortes) - 1):
            min_val = cortes[i]
            max_val = cortes[i+1]
            label = labels_pct[i] 
            
            metrics = calcular_metricas_rango(y_test, y_pred, max_val=max_val, min_val=min_val)
            metrics['Target'] = target
            metrics['Modelo'] = model_name
            metrics['Tipo'] = model_type
            metrics['TipoRango'] = 'Percentil'
            metrics['Rango'] = label
            
            all_results.append(metrics)
            
            if not np.isnan(metrics['R2']):
                min_s = f"{min_val:.1f}" if min_val != -np.inf else "-inf"
                max_s = f"{max_val:.1f}" if max_val != np.inf else "inf"
                print(f"      [{label}]: ({min_s}-{max_s}) R²={metrics['R2']:.4f} | RMSE={metrics['RMSE']:.2f} | SMAPE={metrics['SMAPE']:.2f}% | n={metrics['N_rango']}")
            else:
                pass # Silencioso para no ensuciar mucho log

        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()

# ==================== GUARDAR RESULTADOS ====================
print("\n" + "=" * 80)
print("💾 GUARDANDO RESULTADOS...")

results_df = pd.DataFrame(all_results)

# Reordenar columnas
cols_order = ['Target', 'Modelo', 'Tipo', 'TipoRango', 'Rango', 'N_total', 'N_rango', 
              'R2', 'RMSE', 'MAE', 'wMAPE', 'SMAPE', 'MAPE', 'Bias', 'Moda_Abs']
results_df = results_df[[c for c in cols_order if c in results_df.columns]]

# Guardar CSV general
csv_path = os.path.join(output_dir, 'estadisticas_por_rango_todos.csv')
results_df.to_csv(csv_path, index=False)
print(f"   📄 CSV general: {csv_path}")

# Guardar CSVs separados por target
for target_name in ['c4', 'carbono_bruto4']:
    df_target = results_df[results_df['Target'] == target_name]
    csv_target_path = os.path.join(output_dir, f'estadisticas_{target_name}_todos.csv')
    df_target.to_csv(csv_target_path, index=False)
    print(f"   📄 CSV {target_name}: {csv_target_path}")

# Guardar CSVs separados por TipoRango
for tipo_rango in ['Acumulado', 'Bin']:
    df_tipo = results_df[results_df['TipoRango'] == tipo_rango]
    csv_tipo_path = os.path.join(output_dir, f'estadisticas_tipo_{tipo_rango.lower()}.csv')
    df_tipo.to_csv(csv_tipo_path, index=False)
    print(f"   📄 CSV tipo {tipo_rango}: {csv_tipo_path}")

# Guardar CSVs por target y tipo de rango
for target_name in ['c4', 'carbono_bruto4']:
    for tipo_rango in ['Acumulado', 'Bin']:
        df_subset = results_df[(results_df['Target'] == target_name) & (results_df['TipoRango'] == tipo_rango)]
        csv_subset_path = os.path.join(output_dir, f'estadisticas_{target_name}_{tipo_rango.lower()}.csv')
        df_subset.to_csv(csv_subset_path, index=False)
        print(f"   📄 CSV {target_name} {tipo_rango}: {csv_subset_path}")

# ==================== GENERAR TABLAS FORMATEADAS ====================
print("\n" + "=" * 100)
print("📊 TABLAS DE RESULTADOS")
print("=" * 100)

# --- TABLA 1: Ranking de modelos (Rango Completo) con TODAS las métricas ---
for target_name in ['c4', 'carbono_bruto4']:
    print(f"\n{'='*120}")
    print(f"📈 RANKING DE MODELOS - {target_name.upper()} (Rango Completo)")
    print("="*120)
    
    df_complete = results_df[(results_df['Target'] == target_name) & (results_df['Rango'] == 'Completo')]
    df_complete = df_complete.sort_values('R2', ascending=False)
    
    table_data = []
    for i, (_, row) in enumerate(df_complete.iterrows(), 1):
        table_data.append([
            i,
            row['Modelo'],
            f"{row['R2']:.4f}",
            f"{row['RMSE']:.2f}",
            f"{row['MAE']:.2f}",
            f"{row['wMAPE']:.2f}",
            f"{row['SMAPE']:.2f}",
            f"{row['MAPE']:.2f}" if row['MAPE'] < 10000 else ">10k",
            f"{row['Bias']:.2f}",
            f"{row['Moda_Abs']:.0f}",
            int(row['N_rango'])
        ])
    
    headers = ['#', 'Modelo', 'R²', 'RMSE', 'MAE', 'wMAPE(%)', 'SMAPE(%)', 'MAPE', 'Bias', 'Moda_Abs', 'N']
    print(tabulate(table_data, headers=headers, tablefmt='fancy_grid'))

# --- TABLA 2: Rendimiento por Rango Acumulado (mejores modelos) ---
for target_name in ['c4', 'carbono_bruto4']:
    print(f"\n{'='*140}")
    print(f"📊 RENDIMIENTO POR RANGO ACUMULADO - {target_name.upper()} (Mejor modelo por rango)")
    print("="*140)
    
    df_acum = results_df[(results_df['Target'] == target_name) & (results_df['TipoRango'] == 'Acumulado')]
    
    table_data = []
    for rango in ['0-20', '0-50', '0-100', '0-150', '0-200', '0-300', '0-400', '0-500', '0-600', '0-700', '0-800', 'Completo']:
        df_rango = df_acum[df_acum['Rango'] == rango]
        if len(df_rango) > 0 and not df_rango['R2'].isna().all():
            best_idx = df_rango['R2'].idxmax()
            best = df_rango.loc[best_idx]
            table_data.append([
                rango,
                best['Modelo'],
                f"{best['R2']:.4f}",
                f"{best['RMSE']:.2f}",
                f"{best['MAE']:.2f}",
                f"{best['wMAPE']:.2f}",
                f"{best['SMAPE']:.2f}",
                f"{best['Bias']:.2f}",
                f"{best['Moda_Abs']:.0f}",
                int(best['N_rango'])
            ])
    
    headers = ['Rango', 'Mejor Modelo', 'R²', 'RMSE', 'MAE', 'wMAPE(%)', 'SMAPE(%)', 'Bias', 'Moda_Abs', 'N']
    print(tabulate(table_data, headers=headers, tablefmt='fancy_grid'))

# --- TABLA 3: Rendimiento por Bin Separado ---
for target_name in ['c4', 'carbono_bruto4']:
    print(f"\n{'='*140}")
    print(f"📊 RENDIMIENTO POR BIN SEPARADO - {target_name.upper()} (Mejor modelo por bin)")
    print("="*140)
    
    df_bin = results_df[(results_df['Target'] == target_name) & (results_df['TipoRango'] == 'Bin')]
    
    table_data = []
    rangos_bins = ['0-20', '20-50', '50-100', '100-150', '150-200', '200-300', '300-400', '400-500', '500-600', '600-700', '700-800', '800+']
    
    for rango in rangos_bins:
        df_rango = df_bin[df_bin['Rango'] == rango]
        if len(df_rango) > 0 and not df_rango['R2'].isna().all():
            best_idx = df_rango['R2'].idxmax()
            best = df_rango.loc[best_idx]
            table_data.append([
                rango,
                best['Modelo'],
                f"{best['R2']:.4f}",
                f"{best['RMSE']:.2f}",
                f"{best['MAE']:.2f}",
                f"{best['wMAPE']:.2f}",
                f"{best['SMAPE']:.2f}",
                f"{best['Bias']:.2f}",
                f"{best['Moda_Abs']:.0f}",
                int(best['N_rango'])
            ])
        else:
            n_rango = df_rango['N_rango'].iloc[0] if len(df_rango) > 0 else 0
            table_data.append([rango, '-', '-', '-', '-', '-', '-', '-', '-', int(n_rango)])
    
    headers = ['Bin', 'Mejor Modelo', 'R²', 'RMSE', 'MAE', 'wMAPE(%)', 'SMAPE(%)', 'Bias', 'Moda_Abs', 'N']
    print(tabulate(table_data, headers=headers, tablefmt='fancy_grid'))

# --- TABLA 4: Matriz de métricas por modelo y rango (para modelos top) ---
for target_name in ['c4', 'carbono_bruto4']:
    print(f"\n{'='*100}")
    print(f"📋 MATRIZ R² POR MODELO Y RANGO ACUMULADO - {target_name.upper()}")
    print("="*100)
    
    # Seleccionar top 10 modelos por R² completo
    df_complete = results_df[(results_df['Target'] == target_name) & (results_df['Rango'] == 'Completo')]
    top_modelos = df_complete.nlargest(10, 'R2')['Modelo'].tolist()
    
    df_acum = results_df[(results_df['Target'] == target_name) & (results_df['TipoRango'] == 'Acumulado')]
    
    rangos = ['0-20', '0-50', '0-100', '0-200', '0-400', 'Completo']
    
    table_data = []
    for modelo in top_modelos:
        row_data = [modelo]
        for rango in rangos:
            df_cell = df_acum[(df_acum['Modelo'] == modelo) & (df_acum['Rango'] == rango)]
            if len(df_cell) > 0 and not np.isnan(df_cell['R2'].iloc[0]):
                row_data.append(f"{df_cell['R2'].iloc[0]:.3f}")
            else:
                row_data.append('-')
        table_data.append(row_data)
    
    headers = ['Modelo'] + rangos
    print(tabulate(table_data, headers=headers, tablefmt='fancy_grid'))

# --- TABLA 5: Matriz wMAPE por modelo y bin ---
for target_name in ['c4', 'carbono_bruto4']:
    print(f"\n{'='*100}")
    print(f"📋 MATRIZ wMAPE(%) POR MODELO Y BIN SEPARADO - {target_name.upper()}")
    print("="*100)
    
    # Seleccionar top 10 modelos
    df_complete = results_df[(results_df['Target'] == target_name) & (results_df['Rango'] == 'Completo')]
    top_modelos = df_complete.nlargest(10, 'R2')['Modelo'].tolist()
    
    df_bin = results_df[(results_df['Target'] == target_name) & (results_df['TipoRango'] == 'Bin')]
    
    bins = ['0-20', '20-50', '50-100', '100-150', '200-300', '800+']
    
    table_data = []
    for modelo in top_modelos:
        row_data = [modelo]
        for bin_name in bins:
            df_cell = df_bin[(df_bin['Modelo'] == modelo) & (df_bin['Rango'] == bin_name)]
            if len(df_cell) > 0 and not np.isnan(df_cell['wMAPE'].iloc[0]):
                row_data.append(f"{df_cell['wMAPE'].iloc[0]:.1f}")
            else:
                row_data.append('-')
        table_data.append(row_data)
    
    headers = ['Modelo'] + bins
    print(tabulate(table_data, headers=headers, tablefmt='fancy_grid'))

# ==================== GENERAR GRÁFICOS ====================
print("\n" + "=" * 100)
print("📈 GENERANDO GRÁFICOS DE EVOLUCIÓN DE MÉTRICAS")
print("=" * 100)

graficos_dir = os.path.join(output_dir, 'graficos')
os.makedirs(graficos_dir, exist_ok=True)

# Configuración de colores
colores_base = plt.cm.Set1(np.linspace(0, 1, 10))
colores_stack = plt.cm.tab20(np.linspace(0, 1, 30))

metricas_a_graficar = ['R2', 'RMSE', 'MAE', 'wMAPE', 'SMAPE', 'Bias']
metricas_labels = {
    'R2': 'R²',
    'RMSE': 'RMSE',
    'MAE': 'MAE',
    'wMAPE': 'wMAPE (%)',
    'SMAPE': 'SMAPE (%)',
    'Bias': 'Bias'
}

rangos_acum = ['0-20', '0-50', '0-100', '0-150', '0-200', '0-300', '0-400', '0-500', '0-600', '0-700', '0-800', 'Completo']
rangos_bin = ['0-20', '20-50', '50-100', '100-150', '150-200', '200-300', '300-400', '400-500', '500-600', '600-700', '700-800', '800+']

for target_name in ['c4', 'carbono_bruto4']:
    print(f"\n   Generando gráficos para {target_name.upper()}...")
    
    target_graficos_dir = os.path.join(graficos_dir, target_name)
    os.makedirs(target_graficos_dir, exist_ok=True)
    
    # Separar modelos base y stacking
    df_target = results_df[results_df['Target'] == target_name]
    modelos_base = df_target[df_target['Tipo'] == 'base']['Modelo'].unique().tolist()
    modelos_stack = df_target[df_target['Tipo'] == 'stacking']['Modelo'].unique().tolist()
    
    df_acum = df_target[df_target['TipoRango'] == 'Acumulado']
    df_bin = df_target[df_target['TipoRango'] == 'Bin']
    
    x_pos = range(len(rangos_acum))
    x_pos_bin = range(len(rangos_bin))
    
    # =====================================================
    # GRÁFICOS COMPARATIVOS - MODELOS BASE
    # =====================================================
    print(f"      📊 Gráficos comparativos - Modelos Base...")
    base_dir = os.path.join(target_graficos_dir, 'modelos_base')
    os.makedirs(base_dir, exist_ok=True)
    
    for metrica in metricas_a_graficar:
        # Rango acumulado
        fig, ax = plt.subplots(figsize=(14, 8))
        for i, modelo in enumerate(modelos_base):
            df_modelo = df_acum[df_acum['Modelo'] == modelo]
            valores = []
            for rango in rangos_acum:
                df_r = df_modelo[df_modelo['Rango'] == rango]
                if len(df_r) > 0 and not np.isnan(df_r[metrica].iloc[0]):
                    valores.append(df_r[metrica].iloc[0])
                else:
                    valores.append(np.nan)
            ax.plot(x_pos, valores, marker='o', label=modelo, color=colores_base[i % len(colores_base)], linewidth=2.5, markersize=8)
        
        ax.set_xlabel('Rango', fontsize=12)
        ax.set_ylabel(metricas_labels[metrica], fontsize=12)
        ax.set_title(f'{target_name.upper()} - MODELOS BASE\nEvolución de {metricas_labels[metrica]} por Rango Acumulado', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(rangos_acum, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, f'{metrica}_acumulado.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Bins separados
        fig, ax = plt.subplots(figsize=(14, 8))
        for i, modelo in enumerate(modelos_base):
            df_modelo = df_bin[df_bin['Modelo'] == modelo]
            valores = []
            for rango in rangos_bin:
                df_r = df_modelo[df_modelo['Rango'] == rango]
                if len(df_r) > 0 and not np.isnan(df_r[metrica].iloc[0]):
                    valores.append(df_r[metrica].iloc[0])
                else:
                    valores.append(np.nan)
            ax.plot(x_pos_bin, valores, marker='s', label=modelo, color=colores_base[i % len(colores_base)], linewidth=2.5, markersize=8)
        
        ax.set_xlabel('Bin', fontsize=12)
        ax.set_ylabel(metricas_labels[metrica], fontsize=12)
        ax.set_title(f'{target_name.upper()} - MODELOS BASE\nEvolución de {metricas_labels[metrica]} por Bin Separado', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos_bin)
        ax.set_xticklabels(rangos_bin, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, f'{metrica}_bins.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"         ✅ {len(metricas_a_graficar)*2} gráficos de modelos base")
    
    # =====================================================
    # GRÁFICOS COMPARATIVOS - MODELOS STACKING
    # =====================================================
    print(f"      📊 Gráficos comparativos - Modelos Stacking...")
    stack_dir = os.path.join(target_graficos_dir, 'modelos_stacking')
    os.makedirs(stack_dir, exist_ok=True)
    
    # Agrupar stacks por número (Stack1, Stack2, etc.)
    for stack_num in range(1, 6):  # Stack1 a Stack5
        stack_modelos = [m for m in modelos_stack if m.startswith(f'Stack{stack_num}_')]
        if not stack_modelos:
            continue
        
        stack_subdir = os.path.join(stack_dir, f'Stack{stack_num}')
        os.makedirs(stack_subdir, exist_ok=True)
        
        for metrica in metricas_a_graficar:
            # Rango acumulado
            fig, ax = plt.subplots(figsize=(14, 8))
            for i, modelo in enumerate(stack_modelos):
                df_modelo = df_acum[df_acum['Modelo'] == modelo]
                valores = []
                for rango in rangos_acum:
                    df_r = df_modelo[df_modelo['Rango'] == rango]
                    if len(df_r) > 0 and not np.isnan(df_r[metrica].iloc[0]):
                        valores.append(df_r[metrica].iloc[0])
                    else:
                        valores.append(np.nan)
                ax.plot(x_pos, valores, marker='o', label=modelo.replace(f'Stack{stack_num}_', ''), 
                       color=colores_base[i % len(colores_base)], linewidth=2.5, markersize=8)
            
            ax.set_xlabel('Rango', fontsize=12)
            ax.set_ylabel(metricas_labels[metrica], fontsize=12)
            ax.set_title(f'{target_name.upper()} - Stack {stack_num}\nEvolución de {metricas_labels[metrica]} por Rango Acumulado', fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(rangos_acum, rotation=45, ha='right')
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(stack_subdir, f'{metrica}_acumulado.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            # Bins separados
            fig, ax = plt.subplots(figsize=(14, 8))
            for i, modelo in enumerate(stack_modelos):
                df_modelo = df_bin[df_bin['Modelo'] == modelo]
                valores = []
                for rango in rangos_bin:
                    df_r = df_modelo[df_modelo['Rango'] == rango]
                    if len(df_r) > 0 and not np.isnan(df_r[metrica].iloc[0]):
                        valores.append(df_r[metrica].iloc[0])
                    else:
                        valores.append(np.nan)
                ax.plot(x_pos_bin, valores, marker='s', label=modelo.replace(f'Stack{stack_num}_', ''), 
                       color=colores_base[i % len(colores_base)], linewidth=2.5, markersize=8)
            
            ax.set_xlabel('Bin', fontsize=12)
            ax.set_ylabel(metricas_labels[metrica], fontsize=12)
            ax.set_title(f'{target_name.upper()} - Stack {stack_num}\nEvolución de {metricas_labels[metrica]} por Bin Separado', fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos_bin)
            ax.set_xticklabels(rangos_bin, rotation=45, ha='right')
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(stack_subdir, f'{metrica}_bins.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"         ✅ Stack{stack_num}: {len(metricas_a_graficar)*2} gráficos")
    
    # =====================================================
    # PANELES INDIVIDUALES - TODOS LOS MODELOS
    # =====================================================
    print(f"      📊 Generando paneles individuales para todos los modelos...")
    
    # Paneles modelos base
    base_panel_dir = os.path.join(base_dir, 'paneles')
    os.makedirs(base_panel_dir, exist_ok=True)
    
    for modelo in modelos_base:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'{target_name.upper()} - {modelo}\nEvolución de Métricas por Rango Acumulado', fontsize=14, fontweight='bold')
        df_modelo = df_acum[df_acum['Modelo'] == modelo]
        
        for idx, metrica in enumerate(metricas_a_graficar):
            ax = axes[idx // 3, idx % 3]
            valores = []
            for rango in rangos_acum:
                df_r = df_modelo[df_modelo['Rango'] == rango]
                if len(df_r) > 0 and not np.isnan(df_r[metrica].iloc[0]):
                    valores.append(df_r[metrica].iloc[0])
                else:
                    valores.append(np.nan)
            
            color = 'green' if metrica == 'R2' else 'steelblue'
            ax.bar(x_pos, valores, color=color, alpha=0.7, edgecolor='black')
            ax.plot(x_pos, valores, 'ro-', markersize=5)
            ax.set_title(metricas_labels[metrica], fontsize=12)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(rangos_acum, rotation=45, ha='right', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(base_panel_dir, f'panel_{modelo.replace("/", "_")}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"         ✅ {len(modelos_base)} paneles de modelos base")
    
    # Paneles modelos stacking
    stack_panel_dir = os.path.join(stack_dir, 'paneles')
    os.makedirs(stack_panel_dir, exist_ok=True)
    
    for modelo in modelos_stack:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'{target_name.upper()} - {modelo}\nEvolución de Métricas por Rango Acumulado', fontsize=14, fontweight='bold')
        df_modelo = df_acum[df_acum['Modelo'] == modelo]
        
        for idx, metrica in enumerate(metricas_a_graficar):
            ax = axes[idx // 3, idx % 3]
            valores = []
            for rango in rangos_acum:
                df_r = df_modelo[df_modelo['Rango'] == rango]
                if len(df_r) > 0 and not np.isnan(df_r[metrica].iloc[0]):
                    valores.append(df_r[metrica].iloc[0])
                else:
                    valores.append(np.nan)
            
            color = 'purple' if metrica == 'R2' else 'darkorange'
            ax.bar(x_pos, valores, color=color, alpha=0.7, edgecolor='black')
            ax.plot(x_pos, valores, 'ro-', markersize=5)
            ax.set_title(metricas_labels[metrica], fontsize=12)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(rangos_acum, rotation=45, ha='right', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(stack_panel_dir, f'panel_{modelo.replace("/", "_")}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"         ✅ {len(modelos_stack)} paneles de modelos stacking")
    
    # =====================================================
    # HEATMAPS
    # =====================================================
    print(f"      📊 Generando heatmaps...")
    
    # Heatmap modelos base
    fig, ax = plt.subplots(figsize=(14, 8))
    heatmap_data = []
    for modelo in modelos_base:
        row = []
        df_modelo = df_acum[df_acum['Modelo'] == modelo]
        for rango in rangos_acum:
            df_r = df_modelo[df_modelo['Rango'] == rango]
            if len(df_r) > 0 and not np.isnan(df_r['R2'].iloc[0]):
                row.append(df_r['R2'].iloc[0])
            else:
                row.append(np.nan)
        heatmap_data.append(row)
    
    heatmap_array = np.array(heatmap_data)
    im = ax.imshow(heatmap_array, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(rangos_acum)))
    ax.set_xticklabels(rangos_acum, rotation=45, ha='right')
    ax.set_yticks(range(len(modelos_base)))
    ax.set_yticklabels(modelos_base)
    ax.set_xlabel('Rango', fontsize=12)
    ax.set_ylabel('Modelo', fontsize=12)
    ax.set_title(f'{target_name.upper()} - Heatmap R² - MODELOS BASE', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('R²', fontsize=12)
    for i in range(len(modelos_base)):
        for j in range(len(rangos_acum)):
            val = heatmap_array[i, j]
            if not np.isnan(val):
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'heatmap_R2.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"         ✅ Heatmap modelos base")
    
    # Heatmaps por stack
    for stack_num in range(1, 6):
        stack_modelos = [m for m in modelos_stack if m.startswith(f'Stack{stack_num}_')]
        if not stack_modelos:
            continue
        
        fig, ax = plt.subplots(figsize=(14, 6))
        heatmap_data = []
        for modelo in stack_modelos:
            row = []
            df_modelo = df_acum[df_acum['Modelo'] == modelo]
            for rango in rangos_acum:
                df_r = df_modelo[df_modelo['Rango'] == rango]
                if len(df_r) > 0 and not np.isnan(df_r['R2'].iloc[0]):
                    row.append(df_r['R2'].iloc[0])
                else:
                    row.append(np.nan)
            heatmap_data.append(row)
        
        heatmap_array = np.array(heatmap_data)
        im = ax.imshow(heatmap_array, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
        ax.set_xticks(range(len(rangos_acum)))
        ax.set_xticklabels(rangos_acum, rotation=45, ha='right')
        ax.set_yticks(range(len(stack_modelos)))
        ax.set_yticklabels([m.replace(f'Stack{stack_num}_', '') for m in stack_modelos])
        ax.set_xlabel('Rango', fontsize=12)
        ax.set_ylabel('Meta-Modelo', fontsize=12)
        ax.set_title(f'{target_name.upper()} - Heatmap R² - Stack {stack_num}', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('R²', fontsize=12)
        for i in range(len(stack_modelos)):
            for j in range(len(rangos_acum)):
                val = heatmap_array[i, j]
                if not np.isnan(val):
                    color = 'white' if abs(val) > 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(stack_dir, f'Stack{stack_num}', 'heatmap_R2.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"         ✅ Heatmaps de stacking")

print("\n" + "=" * 100)
print("✅ GENERACIÓN COMPLETADA")
print(f"📁 Resultados guardados en: {output_dir}")
print(f"📊 Gráficos guardados en: {graficos_dir}")
print("=" * 100)
