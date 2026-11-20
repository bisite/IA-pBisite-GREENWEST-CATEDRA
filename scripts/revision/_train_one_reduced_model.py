# -*- coding: utf-8 -*-
"""
Helper aislado: entrena UN modelo (sin npies*) en un subproceso
separado, a partir de un contexto ya preparado (train/test/columnas).

Por que existe este archivo aparte: entrenar "Random Forest" (cuml, GPU)
puede producir un segfault, que no se puede capturar con try/except en el
mismo proceso. Aislando cada modelo en un subproceso, un crash de un modelo
no se lleva por delante los demas (mismo patron que
scripts/revision/_predict_one_baseline_model.py).

Uso: python _train_one_reduced_model.py <ctx_path> <model_name>
Salida en stdout: "RESULT Modelo=<n> Test_R2=<r> Test_RMSE=<r> Tiempo_s=<r>"
Codigo de salida != 0 (incluye segfault) = ese modelo no se pudo entrenar.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from subsampling_experiment import (  # noqa: E402
    TARGET_CONFIG, build_one_estimator, make_preprocessor,
)


def main():
    ctx_path, model_name = sys.argv[1], sys.argv[2]
    ctx = joblib.load(ctx_path)
    X_train, y_train = ctx['X_train'], ctx['y_train']
    X_test, y_test = ctx['X_test'], ctx['y_test']
    present_num, present_npies, present_cat = ctx['present_num'], ctx['present_npies'], ctx['present_cat']
    grid = TARGET_CONFIG[ctx['target']]['param_grids'][model_name]

    # Reutiliza make_preprocessor() de subsampling_experiment.py en vez de
    # una version paralela -- present_npies=[] en el modo
    # "sin npies" y la lista completa en el modo "control, con npies"
    # (ColumnTransformer tolera una rama con lista de columnas vacia).
    preprocessor = make_preprocessor(present_num, present_npies, present_cat)
    pipe = Pipeline([('preprocessor', preprocessor), ('regressor', build_one_estimator(model_name))])
    single_params = {k: v[0] for k, v in grid.items()}
    pipe.set_params(**single_params)

    start = time.time()
    pipe.fit(X_train, y_train)
    elapsed = time.time() - start

    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    print(f"RESULT Modelo={model_name} Test_R2={r2} Test_RMSE={rmse} Tiempo_s={elapsed}")


if __name__ == '__main__':
    main()
