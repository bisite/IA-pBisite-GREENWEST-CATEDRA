# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import joblib
import os
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import warnings
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
from tabulate import tabulate  # <--- IMPORTANTE

# --- AÑADE ESTO ---
try:
    import cuml
    from cuml import SVR as CuSVR
    from cuml.ensemble import RandomForestRegressor as CuRF
    print("✅ Librería cuML detectada y cargada.")
except ImportError:
    print("⚠️ ADVERTENCIA: No se detectó cuML. Si tus .joblib contienen modelos cuML, la carga fallará.")
# ------------------

warnings.filterwarnings("ignore")

import threading
import time
import psutil

# Intentar importar pynvml para la GPU
try:
    import pynvml
    pynvml.nvmlInit()
    HAS_GPU_MONITOR = True
except:
    HAS_GPU_MONITOR = False

# ========= CLASES PYTORCH MLP (necesarias para cargar modelos) =========
import torch.optim as optim

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
    """Wrapper sklearn-compatible para MLP con Dropout y BatchNorm."""
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
        if hasattr(X, 'values'): X = X.values
        if hasattr(y, 'values'): y = y.values
        
        input_dim = X.shape[1]
        self.model_ = PyTorchMLPDropout(
            input_dim, self.hidden_layer_sizes, self.activation,
            self.dropout_rate, self.use_batchnorm
        ).to(self.device)
        
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
        if self.model_ is None:
            raise RuntimeError("Model not fitted")
        if hasattr(X, 'values'): X = X.values
        
        self.model_.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            preds = self.model_(X_tensor).cpu().numpy().flatten()
        return preds

# ========= VARIABLES DE CONFIGURACIÓN (HARDCODEADAS) =========
# --- Rutas y archivos ---
RUTA_CARPETA_MODELOS = os.path.join(REPO_ROOT, "models/artifacts/IFN2-3/carbono_bruto4/base")
RUTA_CSV = os.path.join(REPO_ROOT, "data/processed/ifn_entrenamiento_year_agrupado_periodo_completado_2.csv")
RUTA_SALIDA = os.path.join(REPO_ROOT, "models/artifacts/IFN2-3/carbono_bruto4/stackings")
TARGET = 'carbono_bruto4'

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

# --- Mapeo de nombres para visualización ---
MAP_MODELS_NAMES = {
    "CatBoost": "CatBoost", "Random_Forest": "RandomForest", "AdaBoost": "AdaBoost",
    "BaggedDT": "BaggedDT", "BayesianNN": "BayesianNN", "GBDT": "GBDT", "KNN": "KNN",
    "LightGBM": "LightGBM", "MLP_PyTorch": "MLP_Torch", "XGBoost": "XGBoost", "SVR": "SVR"
}

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
    'gndvi_mean_verano', 'ndii_mean_primavera', 'gndvi_std_primavera', 'evi_mean_primavera'
]

# --- Mapeos de variables categóricas ---
MAPA_PENDIENTE_INT = {
    'muy suave (<5°)': 0,
    'suave (5-10°)': 1,
    'moderada (10-15°)': 2,
    'fuerte (15-20°)': 3,
    'muy fuerte (20-30°)': 4,
    'escarpada (30-50°)': 5,
    'extrema (>50°)': 6
}

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

# --- Configuraciones de Stacking ---
STACK_CONFIGS = [
    ['LightGBM', 'Random_Forest'],
    ['LightGBM', 'XGBoost', 'GBDT'],
    ['CatBoost', 'Random_Forest', 'GBDT'],
    ['CatBoost', 'LightGBM', 'Random_Forest', 'GBDT'],
    ['CatBoost', 'LightGBM', 'XGBoost', 'Random_Forest', 'GBDT', 'BaggedDT'],
]

# ========= FIN VARIABLES DE CONFIGURACIÓN =========

class RecursosMonitor:
    def __init__(self, etiqueta="Proceso", expect_gpu=False):
        self.etiqueta = etiqueta
        self.expect_gpu = expect_gpu
        self.keep_running = False
        self.thread = None
        self.stats = {
            'cpu_max': 0, 'cpu_avg': [],
            'ram_max': 0, 'ram_avg': [],
            'gpu_mem_max': 0, 'gpu_mem_avg': [],
            'gpu_util_max': 0, 'gpu_util_avg': []
        }
        self.start_time = 0
        self.gpu_handle = None
        self.gpu_name = "N/A"
        self.gpu_id = -1
        self.total_gpus = 0

        if HAS_GPU_MONITOR:
            try:
                # 1. Obtener PID y Total de GPUs
                my_pid = os.getpid()
                self.total_gpus = pynvml.nvmlDeviceGetCount() # <--- GUARDAMOS EL TOTAL
                
                found = False
                # 2. Buscar en todas las GPUs
                for i in range(self.total_gpus):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    try:
                        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                        for p in procs:
                            if p.pid == my_pid:
                                self.gpu_id = i
                                self.gpu_handle = handle
                                name_bytes = pynvml.nvmlDeviceGetName(handle)
                                self.gpu_name = name_bytes.decode('utf-8') if isinstance(name_bytes, bytes) else name_bytes
                                found = True
                                break
                    except pynvml.NVMLError:
                        continue 
                    if found: break
                
                # Fallback
                if not found and self.expect_gpu:
                     env_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
                     fallback_id = int(env_gpu.split(',')[0]) if env_gpu else 0
                     self.gpu_id = fallback_id
                     self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(fallback_id)
                     name_bytes = pynvml.nvmlDeviceGetName(self.gpu_handle)
                     self.gpu_name = name_bytes.decode('utf-8') if isinstance(name_bytes, bytes) else name_bytes

            except Exception as e:
                print(f"⚠️ Error detectando GPU: {e}")

    def _monitor_loop(self):
        while self.keep_running:
            # CPU
            cpu_p = psutil.cpu_percent(interval=None)
            ram_info = psutil.virtual_memory()
            self.stats['cpu_avg'].append(cpu_p)
            self.stats['cpu_max'] = max(self.stats['cpu_max'], cpu_p)
            self.stats['ram_avg'].append(ram_info.used / (1024**3))
            self.stats['ram_max'] = max(self.stats['ram_max'], ram_info.used / (1024**3))
            
            # GPU
            if self.gpu_handle:
                try:
                    mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    g_mem = mem.used / (1024**3)
                    g_util = util.gpu
                    self.stats['gpu_mem_avg'].append(g_mem)
                    self.stats['gpu_mem_max'] = max(self.stats['gpu_mem_max'], g_mem)
                    self.stats['gpu_util_avg'].append(g_util)
                    self.stats['gpu_util_max'] = max(self.stats['gpu_util_max'], g_util)
                except: pass
            time.sleep(0.5)

    def __enter__(self):
        self.start_time = time.time()
        self.keep_running = True
        
        # LOG INFORMATIVO CON EL TOTAL DE GPUS
        if self.expect_gpu:
            if self.gpu_handle:
                # AQUÍ MOSTRAMOS EL TOTAL
                print(f"   [Monitor] 🟢 GPU Detectada: ID {self.gpu_id} (de {self.total_gpus} disponibles) | {self.gpu_name}", flush=True)
            else:
                print(f"   [Monitor] ⚠️  Esperando GPU pero no detectada (Total GPUs en nodo: {self.total_gpus})", flush=True)
        
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.keep_running = False
        self.thread.join()
        duration = time.time() - self.start_time
        
        def calc_avg(l): return sum(l)/len(l) if l else 0.0
        
        print(f"\n   📊 [Recursos {self.etiqueta}] Duración: {duration:.1f}s")
        print(f"      CPU: Media {calc_avg(self.stats['cpu_avg']):.1f}% | Max {self.stats['cpu_max']:.1f}%")
        print(f"      RAM: Media {calc_avg(self.stats['ram_avg']):.1f} GB | Max {self.stats['ram_max']:.1f} GB")
        
        avg_util = calc_avg(self.stats['gpu_util_avg'])
        if self.gpu_handle and (self.expect_gpu or avg_util > 1.0):
            print(f"      GPU ({self.gpu_id}): Carga {avg_util:.1f}% | VRAM {calc_avg(self.stats['gpu_mem_avg']):.1f} GB")
            
        print("   " + "-"*60)

# 1. DEFINICIÓN DE FUNCIONES PERSONALIZADAS (OBLIGATORIO)

# ==============================================================================

def cast_to_str(X):
    """Convierte la entrada a string para el OneHotEncoder."""
    return pd.DataFrame(X).astype(str).values


def make_ohe():
    """Crea el encoder compatible con distintas versiones de sklearn."""
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)

# =============================================================================
# 1. DEFINICIÓN DE CLASES PYTORCH (Para cargar .joblib)
# =============================================================================
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
        # Dummy fit para compatibilidad si fuera necesario, pero aquí solo queremos cargar
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

# =============================================================================
# 2. PREPARACIÓN DE DATOS
# =============================================================================
print("➤ Cargando y preparando datos...")
df_base = pd.read_csv(RUTA_CSV) #[:1000] para pruebas rápidas
df_base['pendiente_cat'] = df_base['pendiente_cat'].map(MAPA_PENDIENTE_INT)
df_base['orientacion_cat'] = df_base['orientacion_cat'].map(MAPA_ORIENTACION_INT)
df = df_base[(df_base[TARGET].notna()) & (df_base[TARGET] != 0) & ((df_base['c'].isna()) | (df_base['c'] < df_base[TARGET]))].copy()
# Filtro adicional para sincronizar con entrenamiento (fccarb < 20 = datos problemáticos)
df = df[~(df['fccarb'].notna() & (df['fccarb'] < 20))].copy()
print(f"   Shape tras filtros: {df.shape}")

present_cols = [c for c in FEATURES if c in df.columns]
X = df[present_cols].copy().astype('float32')
y = df[TARGET].copy().astype('float32')

indices_path = os.path.join(RUTA_CARPETA_MODELOS, 'test_indices_real.joblib')

if os.path.exists(indices_path):
    print(f"✅ Cargando índices de Test PRE-DEFINIDOS desde: {indices_path}")
    test_indices = joblib.load(indices_path)
    
    # Verificar que los índices existen en el DataFrame actual
    valid_indices = [i for i in test_indices if i in X.index]
    
    if len(valid_indices) != len(test_indices):
        print(f"⚠️ ADVERTENCIA: {len(test_indices) - len(valid_indices)} índices de test no se encontraron en el DataFrame actual.")
    
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
        
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

elif 'parcela_id' in df.columns:
    print("⚠️ No se encontraron índices guardados. Realizando split automático (GroupShuffleSplit).")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups=df['parcela_id']))
    X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
    y_train, y_test = y.iloc[tr_idx], y.iloc[te_idx]
    groups_train = df['parcela_id'].iloc[tr_idx]
    cv = GroupKFold(n_splits=5)
else:
    print("⚠️ No se encontraron índices guardados. Realizando split automático (train_test_split).")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    groups_train = None
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

# =============================================================================
# 3. FUNCIÓN DE MÉTRICAS (TU CÓDIGO)
# =============================================================================

# Inicialización de diccionarios globales
errores_abs_dict = {}
smape_vector_dict = {}
pct_err_vector_dict = {}
real_values_dict = {}
metricas_output = []

def calcular_metricas(nombre, y_true, y_pred):
    # Asegurar arrays numpy planos
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    errores = y_true - y_pred
    errores_abs = np.abs(errores)

    # Métricas estándar
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    bias = np.mean(errores)

    # --- MAPE (Clásico) ---
    y_true_safe = np.where(y_true == 0, 1e-6, y_true)
    pct_error_vector = (errores / y_true_safe) * 100 
    mape = np.mean(np.abs(pct_error_vector))

    # --- wMAPE (Ponderado) ---
    if np.sum(y_true) != 0:
        wmape = (np.sum(errores_abs) / np.sum(y_true)) * 100
    else:
        wmape = 0.0

    # --- SMAPE (Simétrico) ---
    numerador = np.abs(y_true - y_pred)
    denominador = np.abs(y_true) + np.abs(y_pred) + 1e-6
    smape_vector = 200 * numerador / denominador
    smape_val = np.mean(smape_vector)

    # --- Moda Absoluta ---
    errores_abs_round = np.round(errores_abs, 0).astype(int)
    vals, counts = np.unique(errores_abs_round, return_counts=True)
    try: moda_abs_val = float(vals[np.argmax(counts)])
    except: moda_abs_val = 0.0

    # Guardar vectores
    errores_abs_dict[nombre] = errores 
    smape_vector_dict[nombre] = smape_vector
    pct_err_vector_dict[nombre] = pct_error_vector
    real_values_dict[nombre] = y_true

    return {
        'Modelo': nombre,
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'wMAPE (%)': wmape,
        'SMAPE (%)': smape_val,
        'MAPE (Viejo)': mape,
        'Bias': bias,
        'Moda_Abs': moda_abs_val
    }

# =============================================================================
# 4. CARGA Y EVALUACIÓN DE MODELOS BASE
# =============================================================================
loaded_models = {}
print("\n➤ Cargando y evaluando modelos base...")

for name_key, filename in MODEL_FILES.items():
    path = os.path.join(RUTA_CARPETA_MODELOS, filename)
    if os.path.exists(path):
        try:
            model = joblib.load(path)
            loaded_models[name_key] = model
            
            # --- EVALUAR BASE ---
            preds = model.predict(X_test)
            name_display = MAP_MODELS_NAMES.get(name_key, name_key)
            metricas_output.append(calcular_metricas(name_display, y_test, preds))
            
        except Exception as e:
            print(f"  ❌ Error {name_key}: {e}")
    else:
        print(f"  ⚠️ No encontrado: {filename}")

# --- Mostrar métricas base preliminares para verificar R2 ---
print("\n" + "="*80)
print(" 📊 MÉTRICAS PRELIMINARES DE MODELOS BASE (Test Set Actual)")
print("="*80)
bases_sorted = sorted(metricas_output, key=lambda x: x['R2'], reverse=True)
print(tabulate(bases_sorted, headers="keys", tablefmt="fancy_grid", floatfmt=".4f"))
print("="*80 + "\n")
# ------------------------------------------------------------------


print("\n➤ Verificando motor de ejecución (CPU vs GPU)...")
for name, model in loaded_models.items():
    # Si es un Pipeline, miramos el paso final (el estimador)
    if hasattr(model, 'steps'):
        final_step = model.steps[-1][1]
        tipo = str(type(final_step))
    else:
        tipo = str(type(model))
    
    # Comprobar si 'cuml' está en el nombre del tipo
    donde = "🟢 GPU (cuML)" if 'cuml' in tipo else "🔴 CPU (Sklearn)"
    print(f"  Modelo {name}: {donde}  [{tipo}]")

# =============================================================================
# 5. STACKING (OOF + META) Y EVALUACIÓN Y GUARDADO
# =============================================================================

# 1. CREAR CARPETA DE SALIDA
if not os.path.exists(RUTA_SALIDA):
    os.makedirs(RUTA_SALIDA)
    print(f"📁 Carpeta creada: {RUTA_SALIDA}")



# =============================================================================
# MODIFICACIÓN 1: Activar verbose en los Meta-Modelos
# =============================================================================
meta_models = {
    'GradientBoosting': GradientBoostingRegressor(random_state=42, verbose=1), # <--- verbose=1
    'LinearRegression': LinearRegression(), # No tiene verbose útil
    'Ridge'           : RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5),
    'RandomForest'    : RandomForestRegressor(n_estimators=50, random_state=42, verbose=1), # <--- verbose=1
    'SVR'             : SVR(kernel='linear', verbose=True), # <--- verbose=True
    'MLP'             : MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=42, verbose=True) # <--- verbose=True
}

def get_splits(cv_obj, X, y, groups):
    if isinstance(cv_obj, GroupKFold): return cv_obj.split(X, y, groups)
    return cv_obj.split(X, y)

# =============================================================================
# MODIFICACIÓN 2: Añadir prints en el bucle de Stacking
# =============================================================================

print("\n➤ Ejecutando Stacking...")

for cfg_idx, base_names in enumerate(STACK_CONFIGS, start=1):
    current_bases = [b for b in base_names if b in loaded_models]
    if len(current_bases) < 2: continue
    
    print(f"\n{'='*80}")
    print(f" CONFIGURACIÓN {cfg_idx}/{len(STACK_CONFIGS)}")
    print(f" Modelos Base: {', '.join(current_bases)}")
    print(f"{'='*80}\n")

    # 1. Preparar matrices (Igual que antes)
    n_train, n_test_len = len(X_train), len(X_test)
    n_bases = len(current_bases)
    oof_train = np.zeros((n_train, n_bases))
    oof_test = np.zeros((n_test_len, n_bases))
    
    # Métricas de los modelos base individuales para comparar luego
    base_scores = [] 

    # 2. Generar OOF
    print("➤ Generando predicciones base (OOF)...")
    for i, name in enumerate(current_bases):
        print(f"   > {name}...", end=" ", flush=True)
        pipeline = loaded_models[name]

        # --- [BLOQUE FALTANTE] DETECCIÓN AUTOMÁTICA DE GPU ---
        es_modelo_gpu = False
        tipo_str = "Desconocido"
        try:
            # Si es un pipeline, miramos el paso final (el estimador)
            if hasattr(pipeline, 'steps'):
                final_step = pipeline.steps[-1][1]
                tipo_str = str(type(final_step))
            else:
                tipo_str = str(type(pipeline))
            
            # Si el tipo contiene 'cuml' o 'xgboost' (y está config para gpu), asumimos GPU
            if 'cuml' in tipo_str:
                es_modelo_gpu = True
        except Exception as e:
            es_modelo_gpu = False
        # -----------------------------------------------------

        if name == 'Random_Forest':
            print(f"(DEBUG: {name} ¿Es GPU?: {es_modelo_gpu} | Tipo: {tipo_str})", end=" ")
        
        with RecursosMonitor(f"Base: {name}", expect_gpu=es_modelo_gpu):
            # Test directo y guardar métrica individual para comparar
            y_pred_base = pipeline.predict(X_test)
            oof_test[:, i] = y_pred_base
            
            # Calculamos wMAPE de este modelo base individualmente
            score_base = calcular_metricas(name, y_test, y_pred_base)
            base_scores.append((name, score_base['wMAPE (%)'], score_base['R2']))
            
            # OOF Train
            splits = get_splits(cv, X_train, y_train, groups_train)
            for tr_idx, val_idx in splits:
                print(".", end="", flush=True)
                clone_est = clone(pipeline)
                if isinstance(X_train, pd.DataFrame):
                    X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
                else:
                    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
                clone_est.fit(X_tr, y_tr)
                # Nota: Aseguramos X_val correspondiente
                if isinstance(X_train, pd.DataFrame): X_val_fold = X_train.iloc[val_idx]
                else: X_val_fold = X_train[val_idx]
                
                oof_train[val_idx, i] = clone_est.predict(X_val_fold)
        print(" [OK]")

    # Encontrar el mejor modelo base de este grupo
    best_base_name, best_base_wmape, best_base_r2 = sorted(base_scores, key=lambda x: x[1])[0]
    
    print(f"\n➤ Entrenando Meta-Modelos (Ingredientes: {len(current_bases)} modelos)")
    print(f"   (Referencia: El mejor base es {best_base_name} con wMAPE: {best_base_wmape:.2f}%)")
    print("-" * 60)

    # 3. Entrenar Metas
    iter_results = []
    
    for meta_name, meta_est in meta_models.items():
        print(f"\n🔸 Entrenando {meta_name}:") # Salto de línea antes del log del verbose
        
        stack_id = f"Stack{cfg_idx}_{meta_name}"
        archivo_salida = os.path.join(RUTA_SALIDA, f"{stack_id}.joblib")
        meta_pipe = Pipeline([('scaler', StandardScaler()), ('meta', clone(meta_est))])
        
        # AQUÍ SALE EL VERBOSE DEL MODELO (Loss, iteraciones...)
        meta_pipe.fit(oof_train, y_train)

        # --- GUARDAR MODELO ---
        try:
            joblib.dump(meta_pipe, archivo_salida)
            msg_guardado = "💾 Guardado"
        except Exception as e:
            msg_guardado = f"⚠️ Error guardando: {e}"
        # -----------------------------
        
        # Predicción y Métricas
        y_pred_stack = meta_pipe.predict(oof_test)
        metrics = calcular_metricas(stack_id, y_test, y_pred_stack)
        metricas_output.append(metrics)
        
        # Guardamos para la tabla local
        iter_results.append({
            'Meta': meta_name,
            'wMAPE': metrics['wMAPE (%)'],
            'R2': metrics['R2'],
            'Mejora': best_base_wmape - metrics['wMAPE (%)'] # Cuánto baja el error respecto al mejor base
        })

        print(f"   --> {msg_guardado} en: {archivo_salida}")

    # --- 4. REPORTE COMPARATIVO DE LA CONFIGURACIÓN ---
    print(f"\n📊 RESUMEN CONFIGURACIÓN {cfg_idx} (Ordenado por Error wMAPE):")
    print(f"{'Modelo (Meta o Base)':<25} | {'wMAPE':<10} | {'R²':<8} | {'Diferencia vs Mejor Base':<22}")
    print("-" * 75)
    
    # Añadimos el mejor base a la lista para comparar
    lista_comparativa = iter_results + [{'Meta': f"[BASE] {best_base_name}", 'wMAPE': best_base_wmape, 'R2': best_base_r2, 'Mejora': 0.0}]
    
    # Ordenamos por wMAPE
    lista_comparativa.sort(key=lambda x: x['wMAPE'])
    
    for res in lista_comparativa:
        # Colorear visualmente si es mejor o peor
        es_stacking = "[BASE]" not in res['Meta']
        
        if es_stacking and res['Mejora'] > 0:
            diff_str = f"✅ -{res['Mejora']:.2f}% (MEJORA)"
            prefix = "⚡ "
        elif es_stacking and res['Mejora'] <= 0:
            diff_str = f"❌ +{abs(res['Mejora']):.2f}% (PEORA)"
            prefix = "  "
        else: # Es el base
            diff_str = "--- REFERENCIA ---"
            prefix = "🔹 "
            
        print(f"{prefix}{res['Meta']:<23} | {res['wMAPE']:6.2f}%    | {res['R2']:6.4f}   | {diff_str}")
    print("\n")

# =============================================================================
# 6. TABLA FINAL
# =============================================================================
print("\n" + "="*100)
print(" COMPARATIVA FINAL DE TODOS LOS MODELOS (ORDENADO POR wMAPE)")
print("="*100)

# Ordenar por wMAPE (Menor es mejor)
metricas_output.sort(key=lambda x: x['wMAPE (%)'])

print(tabulate(metricas_output, headers="keys", tablefmt="fancy_grid", floatfmt=".4f"))