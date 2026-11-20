# -*- coding: utf-8 -*-
"""
Experimento de subsampling controlado.

Comparar el modelo "global" (IFN2+IFN3) contra el modelo "local" (IFN3 solo)
mezcla dos efectos, porque el global entrena con más filas.
Para aislar el efecto de mezclar IFN2+IFN3 del efecto de tener más datos,
este script:

  1. Calcula N = nº de filas del dataset IFN3-solo tras los mismos filtros
     que usa `entrenamiento_ifn3_<target>.py`.
  2. Sobre el pool IFN2+IFN3 (mismos filtros que `entrenamiento_ifn2_3_<target>.py`,
     sin filtrar por ifn_id), toma N_REPS submuestras aleatorias SIN reemplazo
     de tamaño N cada una (semillas distintas).
  3. Entrena los mismos 11 modelos con los mismos hiperparámetros fijos que
     ya se usaron para el modelo global de tamaño completo (misma
     configuración, solo cambia el tamaño de los datos).
  4. Reporta métricas de test media +/- std sobre las N_REPS repeticiones.
  5. Construye la tabla comparativa final: IFN3-puro vs IFN2+3-global
     (tamaño completo) vs IFN2+3-subsampleado (tamaño igualado), reutilizando
     los `training_context.joblib` ya guardados por los scripts existentes
     (no hace falta re-entrenar los otros dos).

Uso:
    python subsampling_experiment.py --target carbono_bruto4 --n-reps 5
    python subsampling_experiment.py --target c4 --n-reps 5

Nota de diseño: a diferencia de los scripts `entrenamiento_*`, aquí NO se usa
GridSearchCV. Los param_grids ya son un único valor fijo (los mejores
hiperparámetros encontrados en la ronda 3), así que un GridSearchCV con un
solo combo solo aporta las 5 fits de CV + 1 refit sin buscar nada — se
sustituye por un único `.fit` sobre train y evaluación sobre test, igual que
se compara después. Esto reduce el cómputo ~6x. Si se quiere reportar CV_R2
además de Test_R2, hay un flag `--with-cv` que restaura ese comportamiento.
"""
import argparse
import copy
import gc
import logging
import subprocess
import sys
import time
import types
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import BayesianRidge
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import xgboost as xgb

try:
    from cuml.ensemble import RandomForestRegressor
    from cuml.neighbors import KNeighborsRegressor
    from cuml.svm import SVR
    HAS_CUML = True
except Exception:  # Catches ImportError, CUDARuntimeError, etc.
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    HAS_CUML = False

HAS_GPU = torch.cuda.is_available()


def patch_pandas_legacy_index_module():
    """
    pandas.core.indexes.numeric fue eliminado en pandas 2.0 (Int64Index/
    UInt64Index/Float64Index/NumericIndex se unificaron en el pd.Index
    generico). El X_test/y_test dentro de training_context.joblib se
    pickle-o con pandas <2.0 y referencia ese modulo por nombre exacto ->
    sin este shim, joblib.load falla con ModuleNotFoundError antes incluso
    de llegar a deserializar ningun modelo. Registra un modulo falso en
    sys.modules con esas clases apuntando al pd.Index actual (que ya
    maneja cualquier dtype numerico de forma generica).
    """
    if 'pandas.core.indexes.numeric' in sys.modules:
        return
    shim = types.ModuleType('pandas.core.indexes.numeric')
    for name in ('Int64Index', 'UInt64Index', 'Float64Index', 'NumericIndex'):
        setattr(shim, name, pd.Index)
    sys.modules['pandas.core.indexes.numeric'] = shim


patch_pandas_legacy_index_module()


def patch_cuml_setstate_verbose():
    """
    Los .joblib de cuml (Random Forest, SVR, KNN) guardados con una version
    anterior de cuml no tienen la clave `_verbose` en su estado pickleado,
    que el `__setstate__` de la version instalada espera encontrar. Sin
    esto, joblib.load() falla con `KeyError: '_verbose'` al cargar el
    training_context.joblib del baseline. No-op si cuml no esta instalado.
    """
    try:
        from cuml.ensemble.randomforestregressor import RandomForestRegressor as _CumlRF
        from cuml.svm.svm_base import SVMBase as _CumlSVMBase
        from cuml.neighbors.kneighbors_regressor import KNeighborsRegressor as _CumlKNN
    except ImportError:
        return
    for cls in (_CumlRF, _CumlSVMBase, _CumlKNN):
        orig_setstate = cls.__setstate__

        def patched_setstate(self, state, _orig=orig_setstate):
            if isinstance(state, dict) and '_verbose' not in state:
                state['_verbose'] = state.get('verbose', False)
            return _orig(self, state)

        cls.__setstate__ = patched_setstate


patch_cuml_setstate_verbose()


def patch_torch_load_cpu_fallback():
    """
    Los tensores de MLP_PyTorch se guardaron en un entorno con GPU real
    (torch.cuda.is_available()=True en su momento). Al deserializar aqui
    (donde torch.cuda.is_available() puede ser False, p.ej. por un conflicto
    de librerias CUDA 12/13 en el entorno), torch.load falla con "Attempting to
    deserialize object on a CUDA device but torch.cuda.is_available() is
    False" salvo que se fuerce map_location='cpu'. joblib.load() dispara
    esto internamente (los Storage de torch tienen su propio __reduce__ que
    llama a torch.load), asi que se sustituye torch.load aqui, no en la
    llamada externa.
    """
    orig_load = torch.load
    def patched_load(*args, **kwargs):
        kwargs.setdefault('map_location', torch.device('cpu'))
        return orig_load(*args, **kwargs)
    torch.load = patched_load


patch_torch_load_cpu_fallback()


def patch_xgboost_legacy_attrs(pipeline_or_estimator):
    """Ver scripts/revision/generar_figuras_publicacion.py — mismos atributos
    legacy (`gpu_id`, `predictor`) que XGBModel.get_params() espera y que
    faltan en pickles de una version distinta de XGBoost. No-op para
    modelos que no son XGBoost."""
    steps = pipeline_or_estimator.steps if hasattr(pipeline_or_estimator, 'steps') else [(None, pipeline_or_estimator)]
    for _, step in steps:
        if type(step).__module__.startswith('xgboost'):
            for attr in ('gpu_id', 'predictor'):
                if not hasattr(step, attr):
                    setattr(step, attr, None)
    return pipeline_or_estimator

# ========= CONFIGURACIÓN =========
RUTA_DATOS = Path(__file__).resolve().parents[2] / "data/processed/ifn_entrenamiento_year_agrupado_periodo_completado_2.csv"
OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "models/metrics/subsampling"
BASELINE_ROOT = Path(__file__).resolve().parents[2] / "models/artifacts"

MAPA_PENDIENTE_INT = {
    'muy suave (<5°)': 0, 'suave (5-10°)': 1, 'moderada (10-15°)': 2,
    'fuerte (15-20°)': 3, 'muy fuerte (20-30°)': 4,
    'escarpada (30-50°)': 5, 'extrema (>50°)': 6,
}
MAPA_ORIENTACION_INT = {'N': 0, 'NE': 1, 'E': 2, 'SE': 3, 'S': 4, 'SO': 5, 'O': 6, 'NO': 7}

BASE_CAT_FEATURES = [
    'especie_id', 'tipo_especie', 'estado_id', 'tratmasa_id',
    'rocosidad_id', 'nivel2_id', 'modcomb_id', 'disesp_id',
    'pendiente_cat', 'orientacion_cat', 'martonneidx_id',
]
NPIES_ALL = [
    'npies_1', 'npies_2', 'npies_5', 'npies_10', 'npies_15', 'npies_20',
    'npies_25', 'npies_30', 'npies_35', 'npies_40', 'npies_45',
    'npies_50', 'npies_55', 'npies_60', 'npies_65', 'npies_70',
]

# Features y param_grids EXACTAMENTE como en entrenamiento_ifn2_3_<target>.py
# (mismos hiperparámetros fijos que el modelo global de tamaño completo:
# el único cambio en este experimento es el TAMAÑO de los datos de entrenamiento)
TARGET_CONFIG = {
    'carbono_bruto4': {
        'features': [
            'especie_id', 'tipo_especie', 'periodo_agrupado', 'radio', 'ocupa',
            *NPIES_ALL, 'estado_id', 'fccarb', 'disesp_id', 'modcomb_id',
            'nivel2_id', 'tratmasa_id', 'rocosidad_id', 'orientacion_cat',
            'elevacion', 'pendiente_cat', 'martonneidx_id',
            'gndvi_mean_verano', 'ndii_mean_primavera', 'gndvi_std_primavera', 'evi_mean_primavera',
        ],
        'param_grids': {
            'LightGBM': {'regressor__n_estimators': [600], 'regressor__learning_rate': [0.05],
                         'regressor__num_leaves': [63], 'regressor__max_depth': [10],
                         'regressor__min_child_samples': [20], 'regressor__reg_lambda': [0]},
            'Random Forest': {'regressor__n_estimators': [600], 'regressor__max_depth': [40],
                               'regressor__min_samples_split': [5], 'regressor__min_samples_leaf': [2],
                               'regressor__max_features': [0.7]},
            'XGBoost': {'regressor__n_estimators': [600], 'regressor__learning_rate': [0.05],
                        'regressor__max_depth': [8], 'regressor__subsample': [1.0],
                        'regressor__colsample_bytree': [0.6], 'regressor__reg_alpha': [0.1]},
            'CatBoost': {'regressor__iterations': [800], 'regressor__depth': [8],
                         'regressor__learning_rate': [0.1], 'regressor__l2_leaf_reg': [1],
                         'regressor__bagging_temperature': [0]},
            'SVR': {'regressor__kernel': ['rbf'], 'regressor__C': [1], 'regressor__epsilon': [0.1]},
            'GBDT': {'regressor__n_estimators': [600], 'regressor__learning_rate': [0.05],
                     'regressor__max_depth': [8], 'regressor__subsample': [0.8],
                     'regressor__min_samples_leaf': [2]},
            'BaggedDT': {'regressor__n_estimators': [400], 'regressor__max_samples': [1.0],
                         'regressor__max_features': [0.7]},
            'BayesianNN': {'regressor__max_iter': [300], 'regressor__alpha_1': [1e-6],
                           'regressor__lambda_1': [1e-5]},
            'MLP_PyTorch': {'regressor__hidden_layer_sizes': [(100,)], 'regressor__activation': ['relu'],
                            'regressor__alpha': [0.0001], 'regressor__learning_rate_init': [0.001],
                            'regressor__max_iter': [400], 'regressor__dropout_rate': [0.2],
                            'regressor__use_batchnorm': [True]},
        },
    },
    'c4': {
        'features': [
            'especie_id', 'tipo_especie', 'periodo_agrupado', 'radio', 'ocupa',
            *NPIES_ALL, 'estado_id', 'fccarb', 'disesp_id', 'modcomb_id',
            'nivel2_id', 'tratmasa_id', 'rocosidad_id', 'orientacion_cat',
            'elevacion', 'pendiente_cat', 'martonneidx_id',
            'skt_mean_primavera', 'pr_sum_primavera', 'skt_std_verano', 'skt_std_primavera',
            'pr_sum_otoño', 'skt_mean_verano', 'pr_sum_verano', 'pr_sum_invierno',
            'gndvi_mean_verano', 'ndii_mean_primavera', 'gndvi_std_primavera', 'evi_mean_primavera',
        ],
        'param_grids': {
            'LightGBM': {'regressor__n_estimators': [600], 'regressor__learning_rate': [0.05],
                         'regressor__num_leaves': [63], 'regressor__max_depth': [20],
                         'regressor__min_child_samples': [20], 'regressor__reg_lambda': [0.1]},
            'Random Forest': {'regressor__n_estimators': [600], 'regressor__max_depth': [40],
                               'regressor__min_samples_split': [5], 'regressor__min_samples_leaf': [2],
                               'regressor__max_features': [0.7]},
            'XGBoost': {'regressor__n_estimators': [600], 'regressor__learning_rate': [0.05],
                        'regressor__max_depth': [8], 'regressor__subsample': [0.8],
                        'regressor__colsample_bytree': [0.6], 'regressor__reg_alpha': [0]},
            'CatBoost': {'regressor__iterations': [800], 'regressor__depth': [8],
                         'regressor__learning_rate': [0.1], 'regressor__l2_leaf_reg': [5],
                         'regressor__bagging_temperature': [0]},
            'SVR': {'regressor__kernel': ['rbf'], 'regressor__C': [1], 'regressor__epsilon': [0.5]},
            'GBDT': {'regressor__n_estimators': [600], 'regressor__learning_rate': [0.05],
                     'regressor__max_depth': [8], 'regressor__subsample': [0.8],
                     'regressor__min_samples_leaf': [2]},
            'BaggedDT': {'regressor__n_estimators': [400], 'regressor__max_samples': [1.0],
                         'regressor__max_features': [0.7]},
            'BayesianNN': {'regressor__max_iter': [300], 'regressor__alpha_1': [1e-5],
                           'regressor__lambda_1': [1e-6]},
            'MLP_PyTorch': {'regressor__hidden_layer_sizes': [(100,)], 'regressor__activation': ['relu'],
                            'regressor__alpha': [0.0001], 'regressor__learning_rate_init': [0.001],
                            'regressor__max_iter': [400], 'regressor__dropout_rate': [0.2],
                            'regressor__use_batchnorm': [False]},
        },
    },
}


# ========= CLASES COMPARTIDAS (idénticas a entrenamiento_ifn2_3_<target>.py) =========
class PyTorchMLPDropout(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes=(100,), activation='relu',
                 dropout_rate=0.2, use_batchnorm=True):
        super().__init__()
        layers = []
        in_dim = input_dim
        act_fn = {'relu': nn.ReLU, 'tanh': nn.Tanh}.get(activation, nn.ReLU)
        for h_dim in hidden_layer_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(act_fn())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.model = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)


class MLPRegressorTorch(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='adam',
                 alpha=0.0001, learning_rate_init=0.001, max_iter=200, batch_size=256,
                 dropout_rate=0.2, use_batchnorm=True,
                 device='cuda' if HAS_GPU else 'cpu', verbose=False):
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

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        input_dim = X.shape[1]
        self.model_ = PyTorchMLPDropout(
            input_dim, self.hidden_layer_sizes, self.activation,
            self.dropout_rate, self.use_batchnorm,
        ).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate_init, weight_decay=self.alpha)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_tensor, y_tensor),
            batch_size=self.batch_size, shuffle=True,
        )
        for _ in range(self.max_iter):
            self.model_.train()
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                loss = criterion(self.model_(batch_X), batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("Model not fitted")
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.model_.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.model_(X_tensor).cpu().numpy().flatten()


def cast_to_str(X):
    return pd.DataFrame(X).astype(str).values


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)


def build_one_estimator(name):
    """Factory: create a single fresh estimator instance by name."""
    return {
        'LightGBM': LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1),
        'Random Forest': RandomForestRegressor(random_state=42),
        'XGBoost': xgb.XGBRegressor(random_state=42, verbosity=0,
                                     tree_method='gpu_hist' if HAS_GPU else 'hist', n_jobs=-1),
        'CatBoost': CatBoostRegressor(random_state=42, verbose=0,
                                        task_type='GPU' if HAS_GPU else 'CPU', thread_count=-1),
        'KNN': KNeighborsRegressor(),
        'SVR': SVR(),
        'GBDT': GradientBoostingRegressor(random_state=42, n_estimators=100, warm_start=False),
        'BaggedDT': BaggingRegressor(random_state=42, n_jobs=-1),
        'AdaBoost': AdaBoostRegressor(random_state=42),
        'BayesianNN': BayesianRidge(compute_score=True),
        'MLP_PyTorch': MLPRegressorTorch(device='cuda' if HAS_GPU else 'cpu', verbose=False),
    }[name]


def build_estimators():
    """Instancia fresca de los 11 modelos base (misma config que los scripts existentes)."""
    return {
        'LightGBM': LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1),
        'Random Forest': RandomForestRegressor(random_state=42),
        'XGBoost': xgb.XGBRegressor(random_state=42, verbosity=0,
                                     tree_method='gpu_hist' if HAS_GPU else 'hist', n_jobs=-1),
        'CatBoost': CatBoostRegressor(random_state=42, verbose=0,
                                        task_type='GPU' if HAS_GPU else 'CPU', thread_count=-1),
        'KNN': KNeighborsRegressor(),
        'SVR': SVR(),
        'GBDT': GradientBoostingRegressor(random_state=42, n_estimators=100, warm_start=False),
        'BaggedDT': BaggingRegressor(random_state=42, n_jobs=-1),
        'AdaBoost': AdaBoostRegressor(random_state=42),
        'BayesianNN': BayesianRidge(compute_score=True),
        'MLP_PyTorch': MLPRegressorTorch(device='cuda' if HAS_GPU else 'cpu', verbose=False),
    }


# ========= PREPARACIÓN DE DATOS (mismos filtros que los scripts existentes) =========
def load_and_filter(target, restrict_to_ifn3):
    """Reproduce EXACTAMENTE el pipeline de filtrado de entrenamiento_ifn3_<target>.py
    (si restrict_to_ifn3=True) o entrenamiento_ifn2_3_<target>.py (si False)."""
    df_base = pd.read_csv(RUTA_DATOS)
    if restrict_to_ifn3:
        df_base['ifn_str'] = df_base['ifn_id'].astype(str)
        df_base = df_base[df_base['ifn_str'].str.contains('3')].copy()
    df_base['pendiente_cat'] = df_base['pendiente_cat'].map(MAPA_PENDIENTE_INT)
    df_base['orientacion_cat'] = df_base['orientacion_cat'].map(MAPA_ORIENTACION_INT)
    if target not in df_base.columns:
        raise ValueError(f"Target '{target}' no está en el CSV.")
    df_work = df_base[
        (df_base[target].notna()) & (df_base[target] != 0)
        & ((df_base['c'].isna()) | (df_base['c'] < df_base[target]))
    ].copy()
    df = df_work[~(df_work['fccarb'].notna() & (df_work['fccarb'] < 20))].copy()
    return df


def split_by_group(X, y, groups, test_size=0.2, seed=42, n_splits=5):
    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr_idx, te_idx = next(gss.split(X, y, groups=groups))
        cv = GroupKFold(n_splits=n_splits)
        groups_train = groups.iloc[tr_idx]
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr_idx, te_idx = next(gss.split(X, y))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        groups_train = None
    return X.iloc[tr_idx], X.iloc[te_idx], y.iloc[tr_idx], y.iloc[te_idx], groups_train, cv


def make_preprocessor(present_num, present_npies, present_cat):
    return ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')),
                           ('scaler', StandardScaler())]), present_num),
        ('npies', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                             ('scaler', StandardScaler())]), present_npies),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                           ('to_str', FunctionTransformer(cast_to_str, validate=False)),
                           ('encoder', make_ohe())]), present_cat),
    ])


def fit_eval_one(name, estimator_factory, grid, X_train, y_train, X_test, y_test, cv, groups_train,
                  present_num, present_npies, present_cat, with_cv):
    start = time.time()
    fit_kwargs = {'groups': groups_train} if isinstance(cv, GroupKFold) else {}

    if with_cv:
        # Mismo patrón que entrenamiento_ifn2_3_<target>.py (GridSearchCV con un
        # único combo): más lento (5 fits de CV + 1 refit) pero deja CV_R2 también.
        preprocessor = make_preprocessor(present_num, present_npies, present_cat)
        pipe = Pipeline([('preprocessor', preprocessor), ('regressor', estimator_factory())])
        gs = GridSearchCV(pipe, param_grid=grid, scoring='r2', cv=cv, n_jobs=1)
        gs.fit(X_train, y_train, **fit_kwargs)
        best_est = gs.best_estimator_
        cv_r2 = float(gs.best_score_)
    else:
        # Create a fresh pipeline + estimator for each call to avoid:
        # 1) LightGBM feature count mismatch (OneHotEncoder produces different
        #    column counts per subsample).
        # 2) CatBoost "can't change params of fitted model" on set_params().
        preprocessor = make_preprocessor(present_num, present_npies, present_cat)
        pipe = Pipeline([('preprocessor', preprocessor), ('regressor', estimator_factory())])
        single_params = {k: v[0] for k, v in grid.items()}
        pipe.set_params(**single_params)
        pipe.fit(X_train, y_train)
        best_est = pipe
        cv_r2 = np.nan

    elapsed = time.time() - start

    y_pred = best_est.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    return {'Modelo': name, 'CV_R2': cv_r2, 'Test_R2': float(r2), 'Test_RMSE': rmse, 'Tiempo_s': elapsed}


def run_subsampling_experiment(target, n_reps, seed_base, with_cv, models_subset=None):
    logging.info(f"=== Experimento de subsampling — target={target} ===")

    # 1) Tamaño objetivo: nº de filas de IFN3-solo tras filtros
    df_ifn3 = load_and_filter(target, restrict_to_ifn3=True)
    n_target = len(df_ifn3)
    logging.info(f"N objetivo (IFN3-solo filtrado): {n_target}")
    del df_ifn3
    gc.collect()

    # 2) Pool combinado IFN2+IFN3 tras los mismos filtros que el modelo global
    df_pool = load_and_filter(target, restrict_to_ifn3=False)
    logging.info(f"Pool IFN2+IFN3 filtrado: {len(df_pool)} filas (>= N objetivo: {len(df_pool) >= n_target})")
    if len(df_pool) < n_target:
        raise RuntimeError(
            "El pool IFN2+IFN3 es más pequeño que IFN3-solo tras filtrar; "
            "revisar los filtros antes de continuar."
        )

    cfg = TARGET_CONFIG[target]
    features = cfg['features']
    param_grids = cfg['param_grids']

    # Determinar nombres de modelos válidos (fuera del loop para logging único).
    # KNN y AdaBoost no tienen param_grids en entrenamiento_ifn2_3_<target>.py
    # (excluidos ahí del modelo global de tamaño completo, ver comentario original
    # "KNN y AdaBoost eliminados"). Se excluyen también aquí por consistencia: la
    # tabla comparativa final debe comparar los mismos modelos en las 3 variantes
    # (IFN3-puro / global-completo / global-subsampleado).
    estimators_check = build_estimators()
    skipped = [name for name in estimators_check if name not in param_grids]
    if skipped:
        logging.info(f"Modelos excluidos (sin param_grid, igual que el script global de referencia): {skipped}")
    valid_model_names = set(k for k in estimators_check if k in param_grids)
    del estimators_check  # liberar memoria

    if models_subset:
        valid_model_names = {k for k in valid_model_names if k in models_subset}

    present_cols = [c for c in features if c in df_pool.columns]
    present_cat = [c for c in BASE_CAT_FEATURES if c in present_cols]
    present_npies = [c for c in NPIES_ALL if c in present_cols]
    present_num = [c for c in present_cols if c not in present_cat + present_npies]

    all_reps_rows = []
    for rep in range(n_reps):
        seed = seed_base + rep
        logging.info(f"--- Repetición {rep + 1}/{n_reps} (seed={seed}) ---")
        df_sample = df_pool.sample(n=n_target, random_state=seed).copy()

        X = df_sample[present_cols].astype('float32')
        y = df_sample[target].astype('float32')
        groups = df_sample['parcela_id'] if 'parcela_id' in df_sample.columns else None

        X_train, X_test, y_train, y_test, groups_train, cv = split_by_group(
            X, y, groups, seed=seed,
        )

        # CRITICAL: Build fresh estimators each repetition.
        # Reusing fitted estimators causes:
        # 1) LightGBM feature mismatch (OneHotEncoder may produce different column
        #    counts across subsamples due to 'drop="if_binary"', causing
        #    "X has N features, but expecting M").
        # 2) CatBoost "You can't change params of fitted model" on set_params().
        estimators_all = {k: v for k, v in build_estimators().items() if k in valid_model_names}

        for name in estimators_all:
            try:
                row = fit_eval_one(
                    name, lambda n=name: build_one_estimator(n), param_grids[name],
                    X_train, y_train, X_test, y_test, cv, groups_train,
                    present_num, present_npies, present_cat, with_cv,
                )
                row['rep'] = rep
                row['seed'] = seed
                all_reps_rows.append(row)
                logging.info(f"  [{name}] rep={rep} Test_R2={row['Test_R2']:.4f} "
                             f"Test_RMSE={row['Test_RMSE']:.3f} ({row['Tiempo_s']:.1f}s)")
            except Exception as e:
                logging.error(f"  [CRASH] {name} rep={rep}: {e}", exc_info=True)

        del X, y, X_train, X_test, y_train, y_test
        gc.collect()

    results_df = pd.DataFrame(all_reps_rows)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    raw_path = OUTPUT_ROOT / f"subsampling_{target}_raw_reps.csv"
    results_df.to_csv(raw_path, index=False)
    logging.info(f"Resultados por repetición guardados en: {raw_path}")

    summary = (
        results_df.groupby('Modelo')
        .agg(Test_R2_mean=('Test_R2', 'mean'), Test_R2_std=('Test_R2', 'std'),
             Test_RMSE_mean=('Test_RMSE', 'mean'), Test_RMSE_std=('Test_RMSE', 'std'),
             n_reps=('Test_R2', 'count'))
        .reset_index()
        .sort_values('Test_R2_mean', ascending=False)
    )
    summary_path = OUTPUT_ROOT / f"subsampling_{target}_summary.csv"
    summary.to_csv(summary_path, index=False)
    logging.info(f"Resumen (media +/- std) guardado en: {summary_path}")
    print(summary.to_string(index=False))
    return summary, results_df, n_target


def load_baseline_test_metrics(target, variant):
    """Recalcula Test_R2/RMSE de un modelo ya entrenado (IFN3 o IFN2-3 completo)
    reutilizando el training_context.joblib que cada script ya guarda.

    Devuelve None (con un warning) si el fichero no existe O si no se puede
    deserializar (p.ej. por un cambio interno de scikit-learn entre la
    versión con la que se entrenó el modelo original y la instalada ahora:
    los pipelines de sklearn NO garantizan compatibilidad entre versiones,
    ver https://scikit-learn.org/stable/model_persistence.html). Esto es
    deliberado: sin este fallback, un solo baseline no cargable tira abajo
    todo el experimento de subsampling, que sí es reproducible de forma
    independiente.
    """
    ctx_path = BASELINE_ROOT / variant / target / "base" / "training_context.joblib"
    if not ctx_path.exists():
        logging.warning(f"No se encontró {ctx_path}; ese baseline no se incluirá en la tabla comparativa.")
        return None
    try:
        ctx = joblib.load(ctx_path)
    except Exception as e:
        logging.warning(
            f"No se pudo deserializar {ctx_path} con la versión de scikit-learn instalada "
            f"({e.__class__.__name__}: {e}). Probablemente el modelo se entrenó con otra versión "
            f"de scikit-learn. Ese baseline no se incluirá en la tabla comparativa; para recuperarlo "
            f"hay que re-entrenar con `entrenamiento_<variant>_<target>.py` en este mismo entorno, "
            f"o instalar la versión exacta de sklearn usada originalmente."
        )
        return None

    model_names = list(ctx['slow_best_models'].keys())
    del ctx  # cada subproceso recarga el ctx_path por su cuenta (ver docstring de abajo)
    gc.collect()

    rows = []
    helper = Path(__file__).resolve().parent / "_predict_one_baseline_model.py"
    for name in model_names:
        # Aislado en subproceso: SVR (cuml, via cupy) puede segfaultear dentro
        # de la compilacion JIT de CUDA al predecir
        # - un segfault no se puede capturar con try/except en el mismo
        # proceso, y sin aislar, tumbaria los demas modelos que si funcionan.
        proc = subprocess.run(
            [sys.executable, str(helper), str(ctx_path), name],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            reason = "segfault u otro crash del proceso" if proc.returncode < 0 or proc.returncode == 139 \
                else proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else f"exit code {proc.returncode}"
            logging.warning(f"  No se pudo evaluar el modelo baseline '{name}' ({variant}/{target}): {reason}")
            continue
        line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line is None:
            logging.warning(f"  Modelo baseline '{name}' ({variant}/{target}): subproceso sin resultado parseable.")
            continue
        parts = dict(p.split('=') for p in line.removeprefix("RESULT ").split())
        rows.append({'Modelo': name, 'Test_R2': float(parts['R2']), 'Test_RMSE': float(parts['RMSE'])})
    return pd.DataFrame(rows) if rows else None


def fill_from_saved_predictions(table, target):
    """Completa las métricas IFN3-puro / IFN2+3-completo que no se hayan podido
    evaluar cargando el modelo (p. ej. SVR de cuml, cuya predicción puede
    abortar el proceso) a partir de las predicciones sobre el test guardadas
    por scripts/tables/generar_tablas_y_predicciones.py
    (tables/supplementary/preds_<target>_{IFN3_Locales,Global_Full}.parquet).
    Para los modelos que sí se evalúan, ambas fuentes dan el mismo R2 y RMSE."""
    pred_dir = Path(__file__).resolve().parents[2] / "tables" / "supplementary"
    sources = {
        ('R2_ifn3_pure', 'RMSE_ifn3_pure'): (pred_dir / f"preds_{target}_IFN3_Locales.parquet", "{}"),
        ('R2_global_full', 'RMSE_global_full'): (pred_dir / f"preds_{target}_Global_Full.parquet", "{}$_{{\\text{{global}}}}$"),
    }
    alias = {'MLP_PyTorch': 'MLP'}
    for (r2_col, rmse_col), (path, pattern) in sources.items():
        if r2_col not in table.columns:
            table[r2_col] = np.nan
            table[rmse_col] = np.nan
        missing = table[r2_col].isna()
        if not missing.any() or not path.exists():
            continue
        preds = pd.read_parquet(path)
        y = preds['y_true'].to_numpy(dtype=float)
        for i in table.index[missing]:
            col = pattern.format(alias.get(table.at[i, 'Modelo'], table.at[i, 'Modelo']))
            if col not in preds.columns:
                continue
            p = preds[col].to_numpy(dtype=float)
            table.at[i, r2_col] = r2_score(y, p)
            table.at[i, rmse_col] = float(np.sqrt(mean_squared_error(y, p)))
            logging.info(f"  {table.at[i, 'Modelo']}: {r2_col}/{rmse_col} tomados de {path.name}")
    return table


def build_comparison_table(target, subsampled_summary):
    """Tabla final para la Sección 6.2: IFN3-puro vs IFN2+3-global (completo)
    vs IFN2+3-subsampleado (tamaño igualado)."""
    ifn3 = load_baseline_test_metrics(target, "IFN3")
    global_full = load_baseline_test_metrics(target, "IFN2-3")

    sub = subsampled_summary[['Modelo', 'Test_R2_mean', 'Test_R2_std', 'Test_RMSE_mean', 'Test_RMSE_std']].copy()
    sub = sub.rename(columns={
        'Test_R2_mean': 'R2_subsampled', 'Test_R2_std': 'R2_subsampled_std',
        'Test_RMSE_mean': 'RMSE_subsampled', 'Test_RMSE_std': 'RMSE_subsampled_std',
    })

    table = sub
    if ifn3 is not None:
        table = table.merge(
            ifn3.rename(columns={'Test_R2': 'R2_ifn3_pure', 'Test_RMSE': 'RMSE_ifn3_pure'}),
            on='Modelo', how='left',
        )
    if global_full is not None:
        table = table.merge(
            global_full.rename(columns={'Test_R2': 'R2_global_full', 'Test_RMSE': 'RMSE_global_full'}),
            on='Modelo', how='left',
        )

    table = fill_from_saved_predictions(table, target)

    out_path = OUTPUT_ROOT / f"comparison_table_{target}.csv"
    table.to_csv(out_path, index=False)
    logging.info(f"Tabla comparativa final guardada en: {out_path}")
    print("\n=== TABLA COMPARATIVA (Sección 6.2) ===")
    print(table.to_string(index=False))
    return table


def main():
    parser = argparse.ArgumentParser(description="Experimento de subsampling controlado")
    parser.add_argument('--target', required=True, choices=['c4', 'carbono_bruto4'])
    parser.add_argument('--n-reps', type=int, default=5,
                         help="Nº de submuestras aleatorias independientes (default: 5)")
    parser.add_argument('--seed-base', type=int, default=42)
    parser.add_argument('--with-cv', action='store_true',
                         help="Además del Test_R2, calcula CV_R2 con cross_val_score (más lento)")
    parser.add_argument('--models', nargs='*', default=None,
                         help="Subconjunto de modelos a correr (por defecto, los 11)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(f"subsampling_experiment_{args.target}.log", mode='w'),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    summary, _, n_target = run_subsampling_experiment(
        target=args.target, n_reps=args.n_reps, seed_base=args.seed_base,
        with_cv=args.with_cv, models_subset=args.models,
    )
    logging.info(f"N objetivo usado en el subsampling: {n_target}")
    build_comparison_table(args.target, summary)


if __name__ == '__main__':
    main()
