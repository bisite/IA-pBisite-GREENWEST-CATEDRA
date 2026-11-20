# -*- coding: utf-8 -*-
"""
Scatterplot observado-vs-predicho con test de sesgo (Fig 6.2).

La figura sigue dos criterios:
  1. Ejes: Y = observado/real, X = predicho/estimado
     (convención estándar en evaluación de modelos, no al revés).
  2. Ajustar la regresión lineal y = a + b*x y testear si a y b son
     significativamente distintos de 0 y 1 respectivamente. Si no lo son,
     el modelo es "unbiased" en el sentido de Piñeiro et al. (2008),
     Ecological Modelling 216, 316-322 ("How to evaluate models: observed
     vs. predicted or predicted vs. observed?").

Este módulo solo contiene `test_unbiased_fit()`, separada de la generación
de la figura para poder validarla con datos sintéticos antes de aplicarla a
las predicciones reales de los modelos. El dibujado está en
`draw_density()` de `scripts/revision/generar_figuras_publicacion.py`.
"""
import numpy as np
from scipy.stats import linregress, t as t_dist


def test_unbiased_fit(y_true, y_pred, alpha=0.05):
    """
    Ajusta y_true = a + b * y_pred (Piñeiro et al. 2008: Y=observado, X=predicho)
    y testea H0: a=0 y H0: b=1 por separado (dos t-tests, no uno conjunto).

    Devuelve un dict con la pendiente/intercepto, sus errores estándar,
    estadísticos t, p-valores de cada hipótesis, y un veredicto booleano
    `unbiased` (True si NINGUNA de las dos hipótesis nulas se rechaza a
    nivel `alpha`, es decir, ni a difiere de 0 ni b difiere de 1).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n < 3:
        raise ValueError("Se necesitan al menos 3 puntos para el test de regresión.")

    fit = linregress(y_pred, y_true)
    df = n - 2

    t_intercept = fit.intercept / fit.intercept_stderr
    p_intercept = 2 * t_dist.sf(abs(t_intercept), df)

    t_slope = (fit.slope - 1.0) / fit.stderr
    p_slope = 2 * t_dist.sf(abs(t_slope), df)

    return {
        'n': n,
        'slope': fit.slope,
        'slope_stderr': fit.stderr,
        't_slope_vs_1': t_slope,
        'p_slope_vs_1': p_slope,
        'intercept': fit.intercept,
        'intercept_stderr': fit.intercept_stderr,
        't_intercept_vs_0': t_intercept,
        'p_intercept_vs_0': p_intercept,
        'r_value': fit.rvalue,
        'unbiased': (p_intercept >= alpha) and (p_slope >= alpha),
        'alpha': alpha,
    }


if __name__ == '__main__':
    # Smoke test con datos sinteticos: confirma que el test detecta sesgo
    # cuando lo hay, y no lo detecta cuando el modelo es realmente unbiased.
    rng = np.random.default_rng(42)
    n = 2000

    # Caso 1: modelo unbiased. IMPORTANTE: el ruido tiene que estar en el lado
    # de "observado" (y_true), no en el de "predicho" (y_pred) — igual que en
    # el test real, y_pred hace de X (fijo) e y_true de Y (con error). Poner
    # el ruido en y_pred en vez de y_true simularia atenuacion por error de
    # medida en X, que sesga la pendiente incluso con un modelo "perfecto" —
    # ese es justo el motivo estadistico por el que Piñeiro et al. (2008)
    # insisten en Y=observado, X=predicho y no al reves.
    y_pred_ok = rng.uniform(0, 100, n)
    y_true_ok = y_pred_ok + rng.normal(0, 5, n)
    r_ok = test_unbiased_fit(y_true_ok, y_pred_ok)
    assert r_ok['unbiased'], f"Caso unbiased mal detectado: {r_ok}"
    print(f"[OK] Caso unbiased: p(a=0)={r_ok['p_intercept_vs_0']:.4f}, "
          f"p(b=1)={r_ok['p_slope_vs_1']:.4f} -> unbiased={r_ok['unbiased']}")

    # Caso 2: sesgo claro en pendiente (el modelo subestima sistematicamente
    # a medida que sube el valor real, b=0.7) -> debe rechazar H0: b=1
    y_true_slope = rng.uniform(0, 100, n)
    y_pred_biased_slope = 0.7 * y_true_slope + rng.normal(0, 5, n)
    r_bias_slope = test_unbiased_fit(y_true_slope, y_pred_biased_slope)
    assert not r_bias_slope['unbiased'], f"Sesgo de pendiente no detectado: {r_bias_slope}"
    print(f"[OK] Caso sesgo en pendiente (b=0.7): p(b=1)={r_bias_slope['p_slope_vs_1']:.2e} "
          f"-> unbiased={r_bias_slope['unbiased']}")

    # Caso 3: sesgo claro en intercepto (el modelo sobrestima uniformemente
    # en +15) -> debe rechazar H0: a=0
    y_pred_int = rng.uniform(0, 100, n)
    y_true_biased_intercept = y_pred_int - 15 + rng.normal(0, 5, n)
    r_bias_int = test_unbiased_fit(y_true_biased_intercept, y_pred_int)
    assert not r_bias_int['unbiased'], f"Sesgo de intercepto no detectado: {r_bias_int}"
    print(f"[OK] Caso sesgo en intercepto (a=-15): p(a=0)={r_bias_int['p_intercept_vs_0']:.2e} "
          f"-> unbiased={r_bias_int['unbiased']}")

    print("\nTodos los casos de prueba pasaron.")
