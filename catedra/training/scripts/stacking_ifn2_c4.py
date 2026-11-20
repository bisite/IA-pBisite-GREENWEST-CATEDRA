# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import joblib
import os
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import warnings
import logging
import sys
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, KFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from tabulate import tabulate

# ========= CONFIGURACIÓN DE LOGGING =========
LOG_FORMAT = '%(asctime)s | %(message)s'
LOG_DATEFMT = '%Y-%m-%d %H:%M:%S'

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    filename='stacking_ifn2_c4.log',
    filemode='w',
    force=True
)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
logging.getLogger().addHandler(console_handler)

try:
    import cuml
    from cuml import SVR as CuSVR
    from cuml.ensemble import RandomForestRegressor as CuRF
    logging.info("✅ Librería cuML detectada.")
except ImportError:
    logging.warning("⚠️ No se detectó cuML.")

warnings.filterwarnings("ignore")

import threading
import time
import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_GPU_MONITOR = True
except:
    HAS_GPU_MONITOR = False

# ========= VARIABLES DE CONFIGURACIÓN =========
# --- Rutas y archivos ---
RUTA_CARPETA_MODELOS = os.path.join(REPO_ROOT, "models/artifacts/IFN2/c4/base")
RUTA_CSV = os.path.join(REPO_ROOT, "data/processed/ifn_entrenamiento_year_agrupado_periodo_completado_2.csv")
RUTA_SALIDA = os.path.join(REPO_ROOT, "models/artifacts/IFN2/c4/stackings")
TARGET = 'c4'

# --- Archivos de modelos a cargar ---
MODEL_FILES = {
    'AdaBoost': 'AdaBoost.joblib',
    'BaggedDT': 'BaggedDT.joblib',
    'BayesianNN': 'BayesianNN.joblib',
    'CatBoost': 'CatBoost.joblib',
    'GBDT': 'GBDT.joblib',
    'KNN': 'KNN.joblib',
    'LightGBM': 'LightGBM.joblib',
    'MLP_PyTorch': 'MLP_PyTorch.joblib',
    'Random_Forest': 'Random_Forest.joblib',
    'SVR': 'SVR.joblib',
    'XGBoost': 'XGBoost.joblib'
}

MAP_MODELS_NAMES = {
    "CatBoost": "CatBoost", "Random_Forest": "RandomForest", "AdaBoost": "AdaBoost",
    "BaggedDT": "BaggedDT", "BayesianNN": "BayesianNN", "GBDT": "GBDT", "KNN": "KNN",
    "LightGBM": "LightGBM", "MLP_PyTorch": "MLP_Torch", "XGBoost": "XGBoost", "SVR": "SVR"
}

FEATURES = [
    'especie_id', 'tipo_especie', 'periodo_agrupado', 'radio', 'ocupa',
    'npies_1', 'npies_2', 'npies_5', 'npies_10', 'npies_15', 'npies_20',
    'npies_25', 'npies_30', 'npies_35', 'npies_40', 'npies_45', 'npies_50',
    'npies_55', 'npies_60', 'npies_65', 'npies_70',
    'estado_id', 'fccarb', 'disesp_id',
    'modcomb_id', 'nivel2_id', 'tratmasa_id',
    'rocosidad_id', 'orientacion_cat', 'elevacion', 'pendiente_cat',
    'martonneidx_id',
    'skt_mean_primavera', 'pr_sum_primavera', 'skt_std_verano', 'skt_std_primavera',
    'pr_sum_otoño', 'skt_mean_verano', 'pr_sum_verano', 'pr_sum_invierno',
    'gndvi_mean_verano', 'ndii_mean_primavera', 'gndvi_std_primavera', 'evi_mean_primavera'
]

MAPA_PENDIENTE_INT = {
    'muy suave (<5°)': 0, 'suave (5-10°)': 1, 'moderada (10-15°)': 2,
    'fuerte (15-20°)': 3, 'muy fuerte (20-30°)': 4, 'escarpada (30-50°)': 5, 'extrema (>50°)': 6
}
MAPA_ORIENTACION_INT = {
    'N': 0, 'NE': 1, 'E': 2, 'SE': 3, 'S': 4, 'SO': 5, 'O': 6, 'NO': 7
}

STACK_CONFIGS = [
    ['LightGBM', 'Random_Forest'],
    ['LightGBM', 'XGBoost', 'GBDT'],
    ['CatBoost', 'Random_Forest', 'GBDT'],
    ['CatBoost', 'LightGBM', 'Random_Forest', 'GBDT'],
    ['CatBoost', 'LightGBM', 'XGBoost', 'Random_Forest', 'GBDT', 'BaggedDT'],
]

# 1. DEFINICIÓN DE FUNCIONES PERSONALIZADAS
def cast_to_str(X): return pd.DataFrame(X).astype(str).values
def make_ohe():
    try: return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except: return OneHotEncoder(handle_unknown='ignore', sparse=False)

# DEFINICIÓN DE CLASES PYTORCH
import torch.optim as optim

class PyTorchMLPDropout(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes=(100,), activation='relu', 
                 dropout_rate=0.2, use_batchnorm=True):
        super(PyTorchMLPDropout, self).__init__()
        layers = []
        in_dim = input_dim
        if activation == 'relu': act_fn = nn.ReLU
        elif activation == 'tanh': act_fn = nn.Tanh
        elif activation == 'leaky_relu': act_fn = lambda: nn.LeakyReLU(0.1)
        else: act_fn = nn.ReLU
        
        for i, h_dim in enumerate(hidden_layer_sizes):
            layers.append(nn.Linear(in_dim, h_dim))
            if use_batchnorm: layers.append(nn.BatchNorm1d(h_dim))
            layers.append(act_fn())
            if dropout_rate > 0: layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.model = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, x): return self.model(x)

class MLPRegressorTorch(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='adam', 
                 alpha=0.0001, learning_rate_init=0.001, max_iter=200, batch_size=256,
                 dropout_rate=0.2, use_batchnorm=True,
                 device='cuda' if torch.cuda.is_available() else 'cpu', verbose=False):
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
        self.model_ = None
        
    def fit(self, X, y): return self
    def predict(self, X):
        if self.model_ is None: raise RuntimeError("Model not fitted")
        if isinstance(X, pd.DataFrame): X = X.values
        if not hasattr(self, 'device') or self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_.eval()
        self.model_.to(self.device)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model_(X_tensor).cpu().numpy().flatten()
        return preds

# =============================================================================
# 2. PREPARACIÓN DE DATOS
# =============================================================================
logging.info("➤ Cargando y preparando datos...")
df_base = pd.read_csv(RUTA_CSV)

# --- FILTRO EXCLUSIVO PARA IFN2 ---
logging.info("🔎 Filtrando solo datos del IFN2...")
df_base['ifn_str'] = df_base['ifn_id'].astype(str)
df_base = df_base[df_base['ifn_str'].str.contains('2')].copy() # <--- CLAVE
logging.info(f"✅ Registros tras filtrar IFN2: {len(df_base)}")

df_base['pendiente_cat'] = df_base['pendiente_cat'].map(MAPA_PENDIENTE_INT)
df_base['orientacion_cat'] = df_base['orientacion_cat'].map(MAPA_ORIENTACION_INT)
df = df_base[(df_base[TARGET].notna()) & (df_base[TARGET] != 0) & ((df_base['c'].isna()) | (df_base['c'] < df_base[TARGET]))].copy()
df = df[~(df['fccarb'].notna() & (df['fccarb'] < 20))].copy()

present_cols = [c for c in FEATURES if c in df.columns]
X = df[present_cols].copy().astype('float32')
y = df[TARGET].copy().astype('float32')

indices_path = os.path.join(RUTA_CARPETA_MODELOS, 'test_indices_real.joblib')

if os.path.exists(indices_path):
    logging.info(f"✅ Cargando índices de Test (IFN2) desde: {indices_path}")
    test_indices = joblib.load(indices_path)
    valid_indices = [i for i in test_indices if i in X.index]
    X_test = X.loc[valid_indices]
    y_test = y.loc[valid_indices]
    X_train = X.drop(valid_indices)
    y_train = y.drop(valid_indices)
    
    if 'parcela_id' in df.columns:
        groups_train = df.loc[X_train.index, 'parcela_id']
        cv = GroupKFold(n_splits=5)
    else:
        groups_train = None
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
    logging.info(f"   Train: {len(X_train)} | Test: {len(X_test)}")
else:
    logging.warning("⚠️ No se encontraron índices guardados. Haciendo split automático.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    groups_train = None
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

# =============================================================================
# 3. FUNCIÓN DE MÉTRICAS
# =============================================================================
metricas_output = []
def calcular_metricas(nombre, y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    errores = y_true - y_pred
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    if np.sum(y_true) != 0: wmape = (np.sum(np.abs(errores)) / np.sum(y_true)) * 100
    else: wmape = 0.0
    return {'Modelo': nombre, 'R2': r2, 'RMSE': rmse, 'MAE': mae, 'wMAPE (%)': wmape}

# =============================================================================
# 4. CARGA MODELOS BASE
# =============================================================================
loaded_models = {}
logging.info("\n➤ Cargando modelos base de IFN2...")
for name_key, filename in MODEL_FILES.items():
    path = os.path.join(RUTA_CARPETA_MODELOS, filename)
    if os.path.exists(path):
        try:
            model = joblib.load(path)
            loaded_models[name_key] = model
            preds = model.predict(X_test)
            metrics = calcular_metricas(name_key, y_test, preds)
            metricas_output.append(metrics)
            logging.info(f"   ✅ {name_key}: R2={metrics['R2']:.4f}")
        except Exception as e: logging.info(f"   ❌ Error {name_key}: {e}")

# =============================================================================
# 5. STACKING
# =============================================================================
if not os.path.exists(RUTA_SALIDA): os.makedirs(RUTA_SALIDA)

meta_models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42, verbose=1),
    'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42, verbose=1),
    'SVR': SVR(kernel='linear', verbose=True),
    'MLP': MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=42, verbose=True)
}

def get_splits(cv_obj, X, y, groups):
    if isinstance(cv_obj, GroupKFold): return cv_obj.split(X, y, groups)
    return cv_obj.split(X, y)

logging.info("\n➤ Ejecutando Stacking IFN2...")

for cfg_idx, base_names in enumerate(STACK_CONFIGS, start=1):
    current_bases = [b for b in base_names if b in loaded_models]
    if len(current_bases) < 2: continue
    
    logging.info(f"\n--- CONFIGURACIÓN {cfg_idx} --- Models: {current_bases}")
    n_train, n_test_len = len(X_train), len(X_test)
    n_bases = len(current_bases)
    oof_train = np.zeros((n_train, n_bases))
    oof_test = np.zeros((n_test_len, n_bases))

    # OOF
    for i, name in enumerate(current_bases):
        pipeline = loaded_models[name]
        try:
            oof_test[:, i] = pipeline.predict(X_test)
            splits = get_splits(cv, X_train, y_train, groups_train)
            for tr_idx, val_idx in splits:
                clone_est = clone(pipeline)
                X_tr = X_train.iloc[tr_idx] if isinstance(X_train, pd.DataFrame) else X_train[tr_idx]
                y_tr = y_train.iloc[tr_idx] if isinstance(y_train, pd.Series) else y_train[tr_idx]
                clone_est.fit(X_tr, y_tr)
                X_val_fold = X_train.iloc[val_idx] if isinstance(X_train, pd.DataFrame) else X_train[val_idx]
                oof_train[val_idx, i] = clone_est.predict(X_val_fold)
        except Exception as e:
            logging.error(f"Error OOF {name}: {e}")

    # Meta-Training
    for meta_name, meta_est in meta_models.items():
        stack_id = f"Stack{cfg_idx}_{meta_name}"
        meta_pipe = Pipeline([('scaler', StandardScaler()), ('meta', clone(meta_est))])
        meta_pipe.fit(oof_train, y_train)
        
        joblib.dump(meta_pipe, os.path.join(RUTA_SALIDA, f"{stack_id}.joblib"))
        
        y_pred_stack = meta_pipe.predict(oof_test)
        metrics = calcular_metricas(stack_id, y_test, y_pred_stack)
        metricas_output.append(metrics)
        logging.info(f"   ✅ {stack_id}: R2={metrics['R2']:.4f}")

# RESUMEN FINAL
logging.info("\n" + "="*80)
logging.info("RESUMEN FINAL STACKING IFN2")
metricas_output.sort(key=lambda x: x['wMAPE (%)'])
logging.info(tabulate(metricas_output, headers="keys", tablefmt="fancy_grid", floatfmt=".4f"))
