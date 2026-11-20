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
    RandomForestRegressor as SklearnRF,
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
    filename='entrenamiento_ifn2_c4.log',
    filemode='w',
    force=True
)

# Mostrar también en consola CON timestamps
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
logging.getLogger().addHandler(console_handler)

# ========= VARIABLES DE CONFIGURACIÓN (HARDCODEADAS) =========
# --- Rutas y archivos ---
LOG_FILENAME = 'entrenamiento_ifn2_c4.log'
OUTPUT_DIR = os.path.join(REPO_ROOT, "models/artifacts/IFN2/c4/base")
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
MAPA_PENDIENTE_INT = {
    'muy suave (<5°)': 0, 'suave (5-10°)': 1, 'moderada (10-15°)': 2,
    'fuerte (15-20°)': 3, 'muy fuerte (20-30°)': 4, 'escarpada (30-50°)': 5, 'extrema (>50°)': 6
}

MAPA_ORIENTACION_INT = {
    'N': 0, 'NE': 1, 'E': 2, 'SE': 3, 'S': 4, 'SO': 5, 'O': 6, 'NO': 7
}

# ========= FIN VARIABLES DE CONFIGURACIÓN =========

os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.info(f"Los modelos se guardarán en: {OUTPUT_DIR}")

logging.info("--- INICIO DEL SCRIPT DE ENTRENAMIENTO IFN2 ---")

# ========= 1) FUNCIONES AUXILIARES =========
def cast_to_str(X):
    return pd.DataFrame(X).astype(str).values

def make_ohe():
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
        self.stats = {"cpu_util": [], "ram_util": []}
        if HAS_GPU_MONITOR:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                logging.info(f"ResourceMonitor: Detectadas {device_count} GPUs.")
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    self.gpu_handles.append(handle)
                    self.stats[f"gpu_{i}_util"] = []
                    self.stats[f"gpu_{i}_mem"] = []
            except Exception as e:
                logging.warning(f"Error detectando GPUs: {e}")

    def _loop(self):
        while not self.stop_flag:
            for i, handle in enumerate(self.gpu_handles):
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_mb = mem_info.used / 1024 / 1024 
                    self.stats[f"gpu_{i}_util"].append(util)
                    self.stats[f"gpu_{i}_mem"].append(mem_mb)
                except: pass
            if HAS_PSUTIL:
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().percent
                self.stats["cpu_util"].append(cpu)
                self.stats["ram_util"].append(ram)
            time.sleep(self.interval)

    def start(self):
        self.stop_flag = False
        for k in self.stats: self.stats[k] = []
        if HAS_PSUTIL: psutil.cpu_percent(interval=None)
        self.thread = threading.Thread(target=self._loop)
        self.thread.start()

    def stop(self):
        self.stop_flag = True
        self.thread.join()
        summary = {}
        for k, v in self.stats.items():
            if v:
                summary[f"avg_{k}"] = np.mean(v)
                summary[f"max_{k}"] = np.max(v)
            else:
                summary[f"avg_{k}"] = 0.0
                summary[f"max_{k}"] = 0.0
        return summary

# ========= 3) CLASES PYTORCH MLP =========
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
        
    def fit(self, X, y):
        if isinstance(X, pd.DataFrame): X = X.values
        if isinstance(y, pd.Series): y = y.values
        input_dim = X.shape[1]
        self.model_ = PyTorchMLPDropout(input_dim, self.hidden_layer_sizes, self.activation,
                                        self.dropout_rate, self.use_batchnorm).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate_init, weight_decay=self.alpha)
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
        for epoch in range(self.max_iter):
            self.model_.train()
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        if self.model_ is None: raise RuntimeError("Model not fitted")
        if isinstance(X, pd.DataFrame): X = X.values
        self.model_.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model_(X_tensor).cpu().numpy().flatten()
        return preds

# ========= 4) PREPARACIÓN DE DATOS =========
logging.info(f"➤ Configuración manual: {BEST_CFG_NAME}")
logging.info(f"Cargando datos desde: {RUTA_DATOS}")
df_base = pd.read_csv(RUTA_DATOS)
inventarios_unicos = df_base['ifn_id'].unique()
logging.info(f"📋 Inventarios presentes: {len(inventarios_unicos)} -> {sorted(inventarios_unicos)}")

# --- FILTRO EXCLUSIVO PARA IFN2 ---
logging.info("🔎 Filtrando solo datos del IFN2...")
df_base['ifn_str'] = df_base['ifn_id'].astype(str)
df_base = df_base[df_base['ifn_str'].str.contains('2')].copy() # solo IFN2
logging.info(f"✅ Registros tras filtrar IFN2: {len(df_base)}")

df_base['pendiente_cat'] = df_base['pendiente_cat'].map(MAPA_PENDIENTE_INT)
df_base['orientacion_cat'] = df_base['orientacion_cat'].map(MAPA_ORIENTACION_INT)

df_work = df_base[(df_base[TARGET].notna()) & (df_base[TARGET] != 0) & ((df_base['c'].isna()) | (df_base['c'] < df_base[TARGET]))].copy()
df = df_work[~(df_work['fccarb'].notna() & (df_work['fccarb'] < 20))].copy()
logging.info(f"Shape para entrenamiento: {df.shape}")

def present(cols, df_): return [c for c in cols if c in df_.columns]
present_cols = present(FEATURES, df)
present_cat = [c for c in BASE_CAT_FEATURES if c in present_cols]
present_npies = [c for c in NPIES_ALL if c in present_cols]
present_num = [c for c in present_cols if c not in present_cat + present_npies]

X = df[present_cols].copy().astype('float32')
y = df[TARGET].copy().astype('float32')

# Split
if 'parcela_id' in df.columns:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    try:
        tr_idx, te_idx = next(gss.split(X, y, groups=df['parcela_id']))
        X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
        y_train, y_test = y.iloc[tr_idx], y.iloc[te_idx]
        groups_train = df['parcela_id'].iloc[tr_idx]
        cv = GroupKFold(n_splits=5)
    except Exception as e:
        logging.error(f"Error en GroupShuffleSplit (posiblemente pocos datos): {e}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        groups_train = None
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    groups_train = None
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Guardar índices de test para el Stacking posterior
indices_path = os.path.join(OUTPUT_DIR, 'test_indices_real.joblib')
try:
    joblib.dump(X_test.index, indices_path)
    logging.info(f"✅ Índices de Test (IFN2) guardados en: {indices_path}")
except Exception as e:
    logging.error(f"❌ Error guardando índices de test: {e}")

# ========= 5) Preprocesador =========
preprocessor = ColumnTransformer([
    ('num',  Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), present_num),
    ('npies',Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value=0)), ('scaler', StandardScaler())]), present_npies),
    ('cat',  Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('to_str', FunctionTransformer(cast_to_str, validate=False)), ('encoder', make_ohe())]), present_cat)
])

# ========= 6) Modelos y GRIDs =========
estimadores = {
    'LightGBM': LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1),
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': xgb.XGBRegressor(random_state=42, verbosity=0, tree_method='gpu_hist' if HAS_GPU_MONITOR else 'hist', n_jobs=-1),
    'CatBoost': CatBoostRegressor(random_state=42, verbose=0, task_type='GPU' if HAS_GPU_MONITOR else 'CPU', thread_count=-1),
    'KNN': KNeighborsRegressor(),
    'SVR': SVR(),
    'GBDT': GradientBoostingRegressor(random_state=42, n_estimators=100),
    'BaggedDT': BaggingRegressor(random_state=42, n_jobs=-1),
    'AdaBoost': AdaBoostRegressor(random_state=42),
    'BayesianNN': BayesianRidge(compute_score=True),
    'MLP_PyTorch': MLPRegressorTorch(device='cuda' if torch.cuda.is_available() else 'cpu', verbose=False)
}

param_grids = {
    'LightGBM': {'regressor__n_estimators': [400, 600], 'regressor__learning_rate': [0.05, 0.1], 'regressor__num_leaves': [63], 'regressor__max_depth': [20]},
    'XGBoost': {'regressor__n_estimators': [400, 600], 'regressor__learning_rate': [0.05, 0.1], 'regressor__max_depth': [8]},
    'CatBoost': {'regressor__iterations': [600, 800], 'regressor__depth': [8], 'regressor__learning_rate': [0.05, 0.1]},
    'Random Forest': {'regressor__n_estimators': [400, 600], 'regressor__max_depth': [20, 30]},
    'SVR': {'regressor__kernel': ['rbf'], 'regressor__C': [1, 10], 'regressor__epsilon': [0.1]},
    'KNN': {'regressor__n_neighbors': [5, 11]},
    'GBDT': {'regressor__n_estimators': [400], 'regressor__learning_rate': [0.05]},
    'BaggedDT': {'regressor__n_estimators': [200]},
    'AdaBoost': {'regressor__n_estimators': [200]},
    'BayesianNN': {'regressor__max_iter': [300]},
    'MLP_PyTorch': {'regressor__hidden_layer_sizes': [(256, 128)], 'regressor__learning_rate_init': [0.001], 'regressor__max_iter': [400], 'regressor__dropout_rate': [0.2]}
}

monitor = ResourceMonitor(interval=0.5)

def fit_with_cv(name, estimator, grid):
    print(f"\n🚀 {name.upper()}")
    monitor.start()
    start_time = time.time()
    
    pipe = Pipeline([('preprocessor', preprocessor), ('regressor', estimator)])
    n_jobs_grid = 1 if name in ['LightGBM', 'XGBoost', 'CatBoost', 'Random Forest', 'BaggedDT', 'MLP_PyTorch'] else -1
    
    gs = GridSearchCV(pipe, param_grid=grid, scoring='r2', cv=cv, verbose=1, n_jobs=n_jobs_grid)
    
    fit_kwargs = {'groups': groups_train} if isinstance(cv, GroupKFold) else {}
    gs.fit(X_train, y_train, **fit_kwargs)
    
    elapsed = time.time() - start_time
    monitor.stop()
    
    best_est = gs.best_estimator_
    y_pred = best_est.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"✅ {name} | Time: {elapsed:.1f}s | Test R2: {r2:.4f} | RMSE: {rmse:.4f}")
    logging.info(f"MODELO: {name} | R2: {r2:.4f} | RMSE: {rmse:.4f} | Params: {gs.best_params_}")
    
    filename = f"{name.replace(' ', '_').replace('.', '')}.joblib"
    joblib.dump(best_est, os.path.join(OUTPUT_DIR, filename))
    print(f"💾 Guardado: {filename}")

logging.info(f"Entrenando {len(estimadores)} modelos en IFN2...")
print(f"Entrenando en {len(X_train)} muestras, validando en {len(X_test)} muestras (IFN2).")

for name, est in estimadores.items():
    try:
        fit_with_cv(name, est, param_grids.get(name, {}))
    except Exception as e:
        monitor.stop()
        logging.error(f"❌ Error en {name}: {e}")
        print(f"❌ Error en {name}: {e}")
