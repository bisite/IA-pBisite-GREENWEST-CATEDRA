#!/usr/bin/env python3
"""Behaviour of the best global models at the extremes of the carbon range.

Complements the observed-vs-predicted bias test (Piñeiro et al., 2008;
bias_test.csv written by generar_figuras_publicacion.py) for the four models of
the observed-vs-predicted figure: LightGBM and Stack5 MLP for c4, CatBoost and
Stack5 MLP for carbono_bruto4. For each model, on the full test set:

    mean_observed                  mean observed value of the test set
    n_obs_above / pct_obs_above    test records with observed value > HIGH
    mean_obs_above, mean_pred_above  mean observed / predicted value of those records
    n_pred_above                   test records with predicted value > HIGH
    a/b_excl_pred_above, p_*       bias test after removing those records
    n_obs_below, mean_obs_below, mean_pred_below
                                   same for observed values < LOW
    n_negative / pct_negative      records with a negative prediction

Input: tables/supplementary/preds_<target>_Global_Full.parquet, the test-set
predictions written by scripts/tables/generar_tablas_y_predicciones.py (also
available in the data deposit). Output: data/revision/bias_extreme_values.csv.

Usage:
    python scripts/revision/bias_extreme_values.py [--high 300] [--low 10]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from regression_scatter import test_unbiased_fit  # noqa: E402

PRED_DIR = REPO_ROOT / "tables" / "supplementary"
OUT_CSV = REPO_ROOT / "data" / "revision" / "bias_extreme_values.csv"
MODELS = {"c4": ["LightGBM", "Stack5 MLP"], "carbono_bruto4": ["CatBoost", "Stack5 MLP"]}


def summarise(target, model, y, p, high, low):
    hi_obs, hi_pred, lo_obs, neg = y > high, p > high, y < low, p < 0
    excl = test_unbiased_fit(y[~hi_pred], p[~hi_pred])
    return {
        "target": target, "model": model, "n_test": len(y),
        "mean_observed": y.mean(),
        "n_obs_above": int(hi_obs.sum()), "pct_obs_above": 100 * hi_obs.mean(),
        "mean_obs_above": y[hi_obs].mean(), "mean_pred_above": p[hi_obs].mean(),
        "n_pred_above": int(hi_pred.sum()),
        "a_excl_pred_above": excl["intercept"], "p_a_excl_pred_above": excl["p_intercept_vs_0"],
        "b_excl_pred_above": excl["slope"], "p_b_excl_pred_above": excl["p_slope_vs_1"],
        "n_obs_below": int(lo_obs.sum()),
        "mean_obs_below": y[lo_obs].mean(), "mean_pred_below": p[lo_obs].mean(),
        "n_negative": int(neg.sum()), "pct_negative": 100 * neg.mean(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--high", type=float, default=300.0, help="upper threshold (target units)")
    ap.add_argument("--low", type=float, default=10.0, help="lower threshold (target units)")
    args = ap.parse_args()

    rows = []
    for target, models in MODELS.items():
        df = pd.read_parquet(PRED_DIR / f"preds_{target}_Global_Full.parquet")
        y = df["y_true"].to_numpy(dtype=float)
        for model in models:
            p = df[f"{model}$_{{\\text{{global}}}}$"].to_numpy(dtype=float)
            rows.append(summarise(target, model, y, p, args.high, args.low))

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, float_format="%.6g")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(out.round(3).T.to_string())
    print(f"\nSaved: {OUT_CSV.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
