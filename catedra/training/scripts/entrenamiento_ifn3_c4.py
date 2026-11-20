# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy import stats
import warnings
import logging
import sys
import os
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import joblib
import time
import threading
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- LIBRERÍAS DE MONITORIZACIÓN ---
# 1. CPU / RAM
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("⚠️ No se encontró 'psutil'. Instálalo con: pip install psutil")

# 2. GPU (NVIDIA)
try:
    import pynvml
    pynvml.nvmlInit()
    HAS_GPU_MONITOR = True
except ImportError:
    HAS_GPU_MONITOR = False
    print("⚠️ No se encontró 'nvidia-ml-py'.")
except Exception as e:
    HAS_GPU_MONITOR = False
    print(f"⚠️ Error inicializando NVML: {e}")

warnings.filterwarnings("ignore")

# --- MODELOS SKLEARN ---
from sklearn.ensemble import (
    RandomForestRegressor as SklearnRF, # Renamed to avoid conflict if needed, though we use cuML mostly
    AdaBoostRegressor, 
    BaggingRegressor,
    GradientBoostingRegressor
)
from sklearn.neighbors import KNeighborsRegressor as SklearnKNN
from sklearn.neural_network import MLPRegressor as SklearnMLP
from sklearn.svm import SVR as SklearnSVR
from sklearn.linear_model import BayesianRidge
from sklearn.base import BaseEstimator, RegressorMixin

# --- MODELOS CUML (GPU) ---
try:
    from cuml.ensemble import RandomForestRegressor
    from cuml.neighbors import KNeighborsRegressor
    from cuml.svm import SVR
    HAS_CUML = True
except ImportError:
    print("⚠️ No se encontró 'cuml'. Usando versiones de scikit-learn donde sea posible.")
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    HAS_CUML = False

# --- OTROS MODELOS ---
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import xgboost as xgb

# --- SPLIT / MÉTRICAS / PREPROC ---
from sklearn.model_selection import (train_test_split, GroupShuffleSplit, GroupKFold,
                                     KFold, GridSearchCV)
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ========= 0) CONFIGURACIÓN DE LOGGING =========
LOG_FORMAT = '%(asctime)s | %(message)s'
LOG_DATEFMT = '%Y-%m-%d %H:%M:%S'

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    filename='entrenamiento_ifn3_c4.log',
    filemode='w',
    force=True
)

# Mostrar también en consola CON timestamps
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
logging.getLogger().addHandler(console_handler)

# ========= VARIABLES DE CONFIGURACIÓN (HARDCODEADAS) =========
# --- Rutas y archivos ---
LOG_FILENAME = 'entrenamiento_ifn3_c4.log'
OUTPUT_DIR = os.path.join(REPO_ROOT, "models/artifacts/IFN3/c4/base")
RUTA_DATOS = os.path.join(REPO_ROOT, "data/processed/ifn_entrenamiento_year_agrupado_periodo_completado_2.csv")
TARGET = 'c4'
BEST_CFG_NAME = "features"

# --- Features del modelo ---
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

BASE_CAT_FEATURES = [
    'especie_id', 'tipo_especie', 'estado_id', 'tratmasa_id',
    'rocosidad_id', 'nivel2_id',
    'modcomb_id', 'disesp_id', 'pendiente_cat', 'orientacion_cat', 'martonneidx_id'
]

NPIES_ALL = [
    'npies_1', 'npies_2', 'npies_5', 'npies_10', 'npies_15', 'npies_20',
    'npies_25', 'npies_30', 'npies_35', 'npies_40', 'npies_45',
    'npies_50', 'npies_55', 'npies_60', 'npies_65', 'npies_70'
]

# --- Mapeos de variables categóricas ---
# Asignamos 0 a la más plana y subimos hasta la más extrema.
MAPA_PENDIENTE_INT = {
    'muy suave (<5°)': 0,
    'suave (5-10°)': 1,
    'moderada (10-15°)': 2,
    'fuerte (15-20°)': 3,
    'muy fuerte (20-30°)': 4,
    'escarpada (30-50°)': 5,
    'extrema (>50°)': 6
}

# Mapeo para ORIENTACIÓN (Siguiendo las agujas del reloj)
# Asignamos 0 al Norte y vamos girando hasta cerrar el círculo.
MAPA_ORIENTACION_INT = {
    'N': 0,
    'NE': 1,
    'E': 2,
    'SE': 3,
    'S': 4,
    'SO': 5,
    'O': 6,
    'NO': 7
}

# ========= FIN VARIABLES DE CONFIGURACIÓN =========

os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.info(f"Los modelos se guardarán en: {OUTPUT_DIR}")

logging.info("--- INICIO DEL SCRIPT DE ENTRENAMIENTO COMBINADO ---")

# ========= 1) FUNCIONES AUXILIARES =========
def cast_to_str(X):
    """Convierte la entrada a string para el OneHotEncoder."""
    return pd.DataFrame(X).astype(str).values

def make_ohe():
    """Crea el encoder compatible con distintas versiones de sklearn."""
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)

# ========= 2) CLASE MONITOR DE RECURSOS =========
class ResourceMonitor:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.stop_flag = False
        self.gpu_handles = []
        
        # Estructura base
        self.stats = {
            "cpu_util": [], 
            "ram_util": []
        }

        # Detectar todas las GPUs disponibles
        if HAS_GPU_MONITOR:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                logging.info(f"ResourceMonitor: Detectadas {device_count} GPUs.")
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    self.gpu_handles.append(handle)
                    # Crear listas para cada GPU encontrada
                    self.stats[f"gpu_{i}_util"] = []
                    self.stats[f"gpu_{i}_mem"] = []
            except Exception as e:
                logging.warning(f"Error detectando GPUs: {e}")

    def _loop(self):
        while not self.stop_flag:
            # 1. Loop por cada GPU detectada
            for i, handle in enumerate(self.gpu_handles):
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_mb = mem_info.used / 1024 / 1024 
                    self.stats[f"gpu_{i}_util"].append(util)
                    self.stats[f"gpu_{i}_mem"].append(mem_mb)
                except: 
                    pass
            
            # 2. CPU / RAM Stats
            if HAS_PSUTIL:
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().percent
                self.stats["cpu_util"].append(cpu)
                self.stats["ram_util"].append(ram)
            
            time.sleep(self.interval)

    def start(self):
        self.stop_flag = False
        # Limpiar stats anteriores
        for k in self.stats: self.stats[k] = []
        if HAS_PSUTIL: psutil.cpu_percent(interval=None)
        self.thread = threading.Thread(target=self._loop)
        self.thread.start()

    def stop(self):
        self.stop_flag = True
        self.thread.join()
        
        summary = {}
        # Calcular medias dinámicamente para todo lo que haya en stats
        for k, v in self.stats.items():
            if v:
                summary[f"avg_{k}"] = np.mean(v)
                summary[f"max_{k}"] = np.max(v)
            else:
                summary[f"avg_{k}"] = 0.0
                summary[f"max_{k}"] = 0.0
        return summary

# ========= 3) CLASES PYTORCH MLP CON DROPOUT Y BATCHNORM =========
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
        if isinstance(X, pd.DataFrame): X = X.values
        if isinstance(y, pd.Series): y = y.values
        
        if self.verbose:
            print(f"🔥 Training on device: {self.device} | Dropout: {self.dropout_rate} | BN: {self.use_batchnorm}")

        input_dim = X.shape[1]
        self.model_ = PyTorchMLPDropout(
            input_dim, self.hidden_layer_sizes, self.activation,
            self.dropout_rate, self.use_batchnorm
        ).to(self.device)
        
        criterion = nn.MSELoss()
        
        if self.solver == 'adam':
            optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate_init, weight_decay=self.alpha)
        elif self.solver == 'adamw':
            optimizer = optim.AdamW(self.model_.parameters(), lr=self.learning_rate_init, weight_decay=self.alpha)
        elif self.solver == 'lbfgs':
            optimizer = optim.LBFGS(self.model_.parameters(), lr=self.learning_rate_init, line_search_fn=None)
        else:
            optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate_init, weight_decay=self.alpha)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        
        if self.solver == 'lbfgs':
            loader = torch.utils.data.DataLoader(dataset, batch_size=len(X), shuffle=True)
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
        for epoch in range(self.max_iter):
            self.model_.train()  # Activa Dropout y BatchNorm en modo training
            epoch_loss = 0.0
            batches = 0
            for batch_X, batch_y in loader:
                def closure():
                    optimizer.zero_grad()
                    outputs = self.model_(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    return loss
                
                if self.solver == 'lbfgs':
                    loss = optimizer.step(closure)
                    epoch_loss += loss.item()
                    batches += 1
                else:
                    optimizer.zero_grad()
                    outputs = self.model_(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    batches += 1
            
            if self.verbose and (epoch % 50 == 0 or epoch == self.max_iter - 1):
                avg_loss = epoch_loss / batches if batches > 0 else 0
                print(f"Epoch {epoch}/{self.max_iter} - Loss: {avg_loss:.6f}")
                
        return self

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("Model not fitted")
            
        if isinstance(X, pd.DataFrame): X = X.values
        self.model_.eval()  # Desactiva Dropout en modo evaluación
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            preds = self.model_(X_tensor).cpu().numpy().flatten()
            
        return preds

# ========= 4) PREPARACIÓN DE DATOS =========
logging.info(f"➤ Configuración manual de explicativas cargada: {BEST_CFG_NAME}")
logging.info(f"Cargando datos desde: {RUTA_DATOS}")

df_base = pd.read_csv(RUTA_DATOS)  #[:1000]
# --- COMPROBACIÓN DE INVENTARIOS PRESENTES ---
inventarios_unicos = df_base['ifn_id'].unique()
logging.info(f"📋 Inventarios presentes en datos cargados: {len(inventarios_unicos)} -> {sorted(inventarios_unicos)}")
# --- FILTRO EXCLUSIVO PARA IFN3 ---
logging.info("🔎 Filtrando solo datos del IFN3...")
df_base['ifn_str'] = df_base['ifn_id'].astype(str)
df_base = df_base[df_base['ifn_str'].str.contains('3')].copy()
logging.info(f"✅ Registros tras filtrar IFN3: {len(df_base)}")
# --- COMPROBACIÓN POST-FILTRO ---
inventarios_post = df_base['ifn_id'].unique()
logging.info(f"📋 Inventarios tras filtro: {len(inventarios_post)} -> {sorted(inventarios_post)}")
# ----------------------------------
df_base['pendiente_cat'] = df_base['pendiente_cat'].map(MAPA_PENDIENTE_INT)
df_base['orientacion_cat'] = df_base['orientacion_cat'].map(MAPA_ORIENTACION_INT)
if TARGET not in df_base.columns:
    raise ValueError(f"Target '{TARGET}' no está en el CSV.")

df_work = df_base[(df_base[TARGET].notna()) & (df_base[TARGET] != 0) & ((df_base['c'].isna()) | (df_base['c'] < df_base[TARGET]))].copy()
logging.info(f"Shape tras filtro df_work: {df_work.shape}")
df = df_work.copy()
df = df[~(df['fccarb'].notna() & (df['fccarb'] < 20))].copy()
logging.info(f"Shape tras filtro df: {df.shape}")

def present(cols, df_): return [c for c in cols if c in df_.columns]

present_cols = present(FEATURES, df)
present_cat    = [c for c in BASE_CAT_FEATURES if c in present_cols]
present_npies  = [c for c in NPIES_ALL if c in present_cols]
present_num    = [c for c in present_cols if c not in present_cat + present_npies]

logging.info(f"Variables presentes: {len(present_cols)}")

X = df[present_cols].copy().astype('float32')
y = df[TARGET].copy().astype('float32')

# Split
if 'parcela_id' in df.columns:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups=df['parcela_id']))
    X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
    y_train, y_test = y.iloc[tr_idx], y.iloc[te_idx]
    groups_train = df['parcela_id'].iloc[tr_idx]
    cv = GroupKFold(n_splits=5)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # En caso de train_test_split simple, no tenemos índices explícitos fácil si no reseteamos índice.
    # Pero como el df original tiene índice, podemos recuperar los índices del split.
    tr_idx = X_train.index
    te_idx = X_test.index
    groups_train = None
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

# --- GUARDAR INDICES DE TEST PARA CONSISTENCIA EN STACKING ---
# Guardamos los índices REALES del DataFrame (no los posicionales) para ser robustos ante reordenamientos
indices_path = os.path.join(OUTPUT_DIR, 'test_indices_real.joblib')
try:
    joblib.dump(X_test.index, indices_path)
    logging.info(f"✅ Índices de Test (Reales) guardados en: {indices_path}")
except Exception as e:
    logging.error(f"❌ Error guardando índices de test: {e}")
# -------------------------------------------------------------

# ========= 5) Preprocesador =========
preprocessor = ColumnTransformer([
    ('num',  Pipeline([('imputer', SimpleImputer(strategy='median')),
                          ('scaler', StandardScaler())]), present_num),
    ('npies',Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                          ('scaler', StandardScaler())]), present_npies),
    ('cat',  Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('to_str', FunctionTransformer(cast_to_str, validate=False)),
        ('encoder', make_ohe())
    ]), present_cat)
])

# ========= 6) Modelos y GRIDs (COMBINADOS) =========
estimadores = {
    # --- MODELOS GPU (cuML) - NO usar n_jobs ni verbose ---
    'LightGBM': LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1),  # LightGBM SÍ soporta
    'Random Forest': RandomForestRegressor(random_state=42),  # cuML - NO soporta n_jobs/verbose
    'XGBoost': xgb.XGBRegressor(random_state=42, verbosity=0, tree_method='gpu_hist' if HAS_GPU_MONITOR else 'hist', n_jobs=-1),
    'CatBoost': CatBoostRegressor(random_state=42, verbose=0, task_type='GPU' if HAS_GPU_MONITOR else 'CPU', thread_count=-1),
    'KNN': KNeighborsRegressor(),  # cuML - NO soporta n_jobs/verbose
    'SVR': SVR(),  # cuML - NO soporta verbose
    
    # --- MODELOS CPU ---
    'GBDT': GradientBoostingRegressor(random_state=42, n_estimators=100, warm_start=False),
    'BaggedDT': BaggingRegressor(random_state=42, n_jobs=-1),
    'AdaBoost': AdaBoostRegressor(random_state=42),
    'BayesianNN': BayesianRidge(compute_score=True),
    
    # --- MLP PyTorch (GPU) ---
    'MLP_PyTorch': MLPRegressorTorch(device='cuda' if torch.cuda.is_available() else 'cpu', verbose=False)
}

# ========== PARAM_GRIDS FIJOS - MEJORES PARÁMETROS DE ROUND3 (C4) - COMENTADOS ==========
# param_grids = {
#     'LightGBM': {
#         'regressor__n_estimators': [600],
#         'regressor__learning_rate': [0.05],
#         'regressor__num_leaves': [63],
#         'regressor__max_depth': [20],
#         'regressor__min_child_samples': [20],
#         'regressor__reg_lambda': [0.1],
#     },
#     
#     'Random Forest': {
#         'regressor__n_estimators': [600],
#         'regressor__max_depth': [40],
#         'regressor__min_samples_split': [5],
#         'regressor__min_samples_leaf': [2],
#         'regressor__max_features': [0.7],
#     },
#     
#     'XGBoost': {
#         'regressor__n_estimators': [600],
#         'regressor__learning_rate': [0.05],
#         'regressor__max_depth': [8],
#         'regressor__subsample': [0.8],
#         'regressor__colsample_bytree': [0.6],
#         'regressor__reg_alpha': [0],
#     },
#     
#     'CatBoost': {
#         'regressor__iterations': [800],
#         'regressor__depth': [8],
#         'regressor__learning_rate': [0.1],
#         'regressor__l2_leaf_reg': [5],
#         'regressor__bagging_temperature': [0],
#     },
#     
#     'SVR': {
#         'regressor__kernel': ['rbf'],
#         'regressor__C': [1],
#         'regressor__epsilon': [0.5],
#     },
#     
#     'GBDT': {
#         'regressor__n_estimators': [600],
#         'regressor__learning_rate': [0.05],
#         'regressor__max_depth': [8],
#         'regressor__subsample': [0.8],
#         'regressor__min_samples_leaf': [2],
#     },
#     
#     'BaggedDT': {
#         'regressor__n_estimators': [400],
#         'regressor__max_samples': [1.0],
#         'regressor__max_features': [0.7],
#     },
#     
#     'BayesianNN': {
#         'regressor__max_iter': [300],
#         'regressor__alpha_1': [1e-5],
#         'regressor__lambda_1': [1e-6],
#     },
#     
#     'MLP_PyTorch': {
#         'regressor__hidden_layer_sizes': [(100,)],
#         'regressor__activation': ['relu'],
#         'regressor__alpha': [0.0001],
#         'regressor__learning_rate_init': [0.001],
#         'regressor__max_iter': [400],
#         'regressor__dropout_rate': [0.2],
#         'regressor__use_batchnorm': [False],
#     },
# }

# ========== PARAM_GRIDS EXHAUSTIVOS (GRID SEARCH ACTIVO - ~3+ días) ==========
# ========== PARAM_GRIDS OPTIMIZADOS (VERSIÓN RÁPIDA) ==========
param_grids = {
    'LightGBM': {
        'regressor__n_estimators': [400, 600],
        'regressor__learning_rate': [0.05, 0.1],
        'regressor__num_leaves': [63],
        'regressor__max_depth': [20],
        'regressor__min_child_samples': [20],
    },
    'XGBoost': {
        'regressor__n_estimators': [400, 600],
        'regressor__learning_rate': [0.05, 0.1],
        'regressor__max_depth': [8],
        'regressor__subsample': [0.8],
        'regressor__colsample_bytree': [0.8],
    },
    'CatBoost': {
        'regressor__iterations': [600, 800],
        'regressor__depth': [8],
        'regressor__learning_rate': [0.05, 0.1],
        'regressor__l2_leaf_reg': [3],
    },
    'Random Forest': {
        'regressor__n_estimators': [400, 600],
        'regressor__max_depth': [20, 30],
        'regressor__min_samples_split': [5],
    },
    'SVR': {
        'regressor__kernel': ['rbf'],
        'regressor__C': [1, 10],
        'regressor__epsilon': [0.1],
    },
    'KNN': {
        'regressor__n_neighbors': [5, 11],
        'regressor__weights': ['distance'],
    },
    'GBDT': {
        'regressor__n_estimators': [400],
        'regressor__learning_rate': [0.05],
        'regressor__max_depth': [5],
    },
    'BaggedDT': {
        'regressor__n_estimators': [200],
        'regressor__max_samples': [0.85],
    },
    'AdaBoost': {
        'regressor__n_estimators': [200],
        'regressor__learning_rate': [0.1, 0.5],
    },
    'BayesianNN': {
        'regressor__max_iter': [300],
    },
    'MLP_PyTorch': {
        'regressor__hidden_layer_sizes': [(256, 128)],
        'regressor__activation': ['relu'],
        'regressor__learning_rate_init': [0.001],
        'regressor__max_iter': [400],
        'regressor__dropout_rate': [0.2],
        'regressor__use_batchnorm': [True],
    },
}

# ========= 7) Entrenamiento con Monitorización =========
monitor = ResourceMonitor(interval=0.5)
slow_results_rows = []
slow_best_models  = {}
slow_best_params  = {}

def fit_with_cv(name, estimator, grid):
    # --- A) HEADER VISUAL ---
    header_len = 60
    print(f"\n{'='*header_len}")
    print(f"🚀 INICIANDO: {name.upper()}")
    print(f"{'='*header_len}")
    
    logging.info(f"--> Iniciando: {name}")
    
    # 1. Iniciar monitor
    monitor.start()
    start_time = time.time()
    
    # 2. Crear Pipeline y Grid
    pipe = Pipeline([('preprocessor', preprocessor), ('regressor', estimator)])
    
    # Ajustar n_jobs para evitar conflictos con modelos que ya paralelizan
    n_jobs_grid = 1 if name in ['LightGBM', 'XGBoost', 'CatBoost', 'Random Forest', 'BaggedDT', 'MLP_PyTorch'] else -1
    
    gs = GridSearchCV(pipe, param_grid=grid, scoring='r2', cv=cv, verbose=2, n_jobs=n_jobs_grid)
    
    fit_kwargs = {}
    if isinstance(cv, GroupKFold):
        fit_kwargs['groups'] = groups_train
        
    # Ejecución del entrenamiento
    gs.fit(X_train, y_train, **fit_kwargs)

    # 3. Detener monitor y calcular tiempos
    elapsed = time.time() - start_time
    metrics = monitor.stop()
    
    # 4. Evaluar
    best_est = gs.best_estimator_
    y_pred = best_est.predict(X_test)
    
    # Métricas básicas
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Métricas detalladas
    err = y_test - y_pred
    rmse_ind = np.sqrt(err**2)
    
    media_rmse = float(np.mean(rmse_ind))
    mediana_rmse = float(np.median(rmse_ind))
    
    if len(rmse_ind) > 0:
        moda_res = stats.mode(np.round(rmse_ind), keepdims=True)
        moda_rmse = float(moda_res.mode[0])
    else:
        moda_rmse = np.nan

    # 5. PRESENTACIÓN EN CONSOLA
    print(f"\n✅ {name} FINALIZADO en {elapsed:.2f}s")
    print("-" * header_len)
    
    print(f"  📊 RENDIMIENTO:")
    print(f"     • CV R² (Validación):  {gs.best_score_:.4f}")
    print(f"     • Test R²:             {r2:.4f}")
    print(f"     • Test RMSE:           {rmse:.4f}")
    
    print(f"  💻 RECURSOS:")
    print(f"     • CPU: {metrics['avg_cpu_util']:.1f}%")
    
    gpu_utils_list = []
    gpu_mems_list = []
    num_gpus = 0
    
    while f"avg_gpu_{num_gpus}_util" in metrics:
        avg_u = metrics[f"avg_gpu_{num_gpus}_util"]
        max_m = metrics[f"max_gpu_{num_gpus}_mem"]
        
        gpu_utils_list.append(avg_u)
        gpu_mems_list.append(max_m)
        
        print(f"     • GPU {num_gpus}: Util: {avg_u:.1f}% | Mem Max: {max_m:.0f} MB")
        num_gpus += 1

    if num_gpus == 0:
        print("     • GPU: No detectada o sin datos.")
        global_gpu_util = 0.0
        global_gpu_mem = 0.0
    else:
        global_gpu_util = np.mean(gpu_utils_list)
        global_gpu_mem = np.max(gpu_mems_list)

    print(f"  🛠  MEJORES PARÁMETROS:")
    for k, v in gs.best_params_.items():
        clean_k = k.replace('regressor__', '')
        print(f"     • {clean_k:<25}: {v}")

    # 6. LOGUEAR
    log_msg = (f"    [FIN {name}] T: {elapsed:.1f}s | R2: {r2:.4f} | RMSE: {rmse:.4f} | "
               f"GPUs_Avg: {global_gpu_util:.1f}% Max_Mem: {global_gpu_mem:.0f}MB | "
               f"CPU_Avg: {metrics['avg_cpu_util']:.1f}%")
    logging.info(log_msg)

    # 7. ACTUALIZAR OBJETOS GLOBALES
    slow_best_models[name] = best_est
    slow_best_params[name] = gs.best_params_

    # 8. GUARDAR MODELO EN DISCO
    filename = f"{name.replace(' ', '_').replace('.', '')}.joblib"
    filepath = os.path.join(OUTPUT_DIR, filename)
    try:
        joblib.dump(best_est, filepath)
        print(f"  💾 GUARDADO: {filename}")
        logging.info(f"    Modelo guardado: {filename}")
    except Exception as e:
        print(f"  ❌ ERROR AL GUARDAR: {e}")
        logging.error(f"    [ERROR] No se pudo guardar {name}: {e}")

    print(f"{'='*header_len}\n")

    # 9. TABLA FINAL
    slow_results_rows.append({
        'Configuración': BEST_CFG_NAME,
        'Modelo': name,
        'CV_R2': float(gs.best_score_),
        'Test_R2': float(r2),
        'Test_RMSE': float(rmse),
        'Media_RMSE_ind': media_rmse,
        'Mediana_RMSE_ind': mediana_rmse,
        'Moda_RMSE_ind': moda_rmse,
        'Best_Params': gs.best_params_,
        'Tiempo_s': elapsed,
        'GPU_Util_Avg': global_gpu_util,
        'GPU_Mem_Max': global_gpu_mem,
        'CPU_Util_Avg': metrics['avg_cpu_util']
    })


logging.info(f"Entrenando {len(estimadores)} modelos...")

for name, est in estimadores.items():
    try:
        fit_with_cv(name, est, param_grids[name])
    except Exception as e:
        monitor.stop() # Asegurarnos de parar el hilo si hay crash
        logging.error(f"  > [CRASH] {name}: {e}", exc_info=True)
        print(f"❌ Error crítico en {name}: {e}")

# ========= 8) Resultados =========
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'left')

slow_results_df = pd.DataFrame(slow_results_rows)

if not slow_results_df.empty:
    # --- A) LIMPIEZA Y FORMATO ---
    summary_cols = ['Modelo', 'CV_R2', 'Test_R2', 'Test_RMSE', 
                    'Tiempo_s', 'GPU_Util_Avg', 'CPU_Util_Avg']
    
    leaderboard = slow_results_df.sort_values(['Test_R2', 'CV_R2'], ascending=[False, False]).reset_index(drop=True)
    
    format_dict = {
        'CV_R2': '{:.4f}'.format,
        'Test_R2': '{:.4f}'.format,
        'Test_RMSE': '{:.2f}'.format,
        'Tiempo_s': '{:.1f}s'.format,
        'GPU_Util_Avg': '{:.1f}%'.format,
        'CPU_Util_Avg': '{:.1f}%'.format
    }

    print("\n" + "="*80)
    print(f" >>> TABLA DE LÍDERES (Ordenado por Test R2) <<<")
    print("="*80)
    
    print(leaderboard[summary_cols].to_string(index=False, formatters=format_dict))

    # --- B) SELECCIÓN DEL MEJOR ---
    best_row = leaderboard.iloc[0]
    best_name = best_row['Modelo']
    
    print("\n" + "-"*60)
    print(f"🏆 GANADOR: {best_name.upper()}")
    print("-"*60)
    print(f"  ➤ R2 Test:     {best_row['Test_R2']:.4f}")
    print(f"  ➤ RMSE Test:   {best_row['Test_RMSE']:.4f}")
    
    if 'Media_RMSE_ind' in best_row:
        print(f"  ➤ RMSE Indiv:  Media={best_row['Media_RMSE_ind']:.2f} | Mediana={best_row['Mediana_RMSE_ind']:.2f}")
    
    print(f"\n  ➤ Mejores Parámetros ({best_name}):")
    params = best_row['Best_Params']
    for k, v in params.items():
        clean_k = k.replace('regressor__', '')
        print(f"     • {clean_k:<25}: {v}")

else:
    print("⚠️ No hay resultados. Revisa los logs de error anteriores.")

# ========= 9) GUARDAR CONTEXTO PARA ENSEMBLE =========
logging.info("Guardando contexto de ejecución para ensemble.py...")
context_path = os.path.join(OUTPUT_DIR, "training_context.joblib")

execution_context = {
    'best_cfg_name': BEST_CFG_NAME,
    'slow_best_models': slow_best_models,
    'X_train': X_train,
    'y_train': y_train,
    'X_test': X_test,
    'y_test': y_test,
    'cv': cv,
    'groups_train': groups_train
}

try:
    joblib.dump(execution_context, context_path)
    print(f"\n✅ Contexto guardado exitosamente en: {context_path}")
    logging.info(f"Contexto guardado en: {context_path}")
except Exception as e:
    print(f"\n❌ Error al guardar el contexto: {e}")
    logging.error(f"Error guardando contexto: {e}")

print("\n--- FIN DEL SCRIPT ---")
