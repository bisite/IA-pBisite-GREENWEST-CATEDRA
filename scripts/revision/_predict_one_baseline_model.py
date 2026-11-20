# -*- coding: utf-8 -*-
"""
Helper aislado: carga UN training_context.joblib y predice con UN solo
modelo de `slow_best_models`, imprimiendo el resultado como texto parseable.

Por qué existe este archivo aparte:
SVR (cuml, vía cupy) puede segfaultear dentro de la compilación JIT de CUDA
(cupy/cuda/compiler.py) al predecir — un segfault no se puede capturar con
try/except en el mismo proceso. Ejecutando cada modelo en un subproceso
separado, un crash de SVR no se lleva por delante los otros 8 modelos que sí
funcionan bien.

Uso: python _predict_one_baseline_model.py <ctx_path> <model_name>
Salida en stdout: "RESULT R2=<valor> RMSE=<valor>" si tuvo éxito.
Código de salida != 0 (incluye segfault) = ese modelo no se pudo evaluar.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
from subsampling_experiment import (  # noqa: E402
    cast_to_str, make_ohe, MLPRegressorTorch, PyTorchMLPDropout,  # noqa: F401
    patch_xgboost_legacy_attrs,
)
import joblib  # noqa: E402


def force_mlp_torch_cpu(pipeline_or_estimator):
    """
    MLPRegressorTorch ya trae `self.device` fijado a 'cuda' desde que se
    entrenó (con GPU real) — su propio predict() solo decide el device si
    `self.device is None`, así que sin esto intenta `.to('cuda')`
    directamente y falla cuando CUDA no está disponible en el nodo
    (torch.cuda.is_available() == False). No-op para otros modelos.
    """
    steps = pipeline_or_estimator.steps if hasattr(pipeline_or_estimator, 'steps') else [(None, pipeline_or_estimator)]
    for _, step in steps:
        if isinstance(step, MLPRegressorTorch):
            step.device = torch.device('cpu')
    return pipeline_or_estimator


def main():
    ctx_path, model_name = sys.argv[1], sys.argv[2]
    ctx = joblib.load(ctx_path)
    X_test, y_test = ctx['X_test'], ctx['y_test']
    model = ctx['slow_best_models'][model_name]
    patch_xgboost_legacy_attrs(model)
    force_mlp_torch_cpu(model)
    y_pred = model.predict(X_test)
    r2 = float(r2_score(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    print(f"RESULT R2={r2} RMSE={rmse}")


if __name__ == '__main__':
    main()
