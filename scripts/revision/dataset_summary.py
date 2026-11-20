#!/usr/bin/env python3
"""Dataset sizes and summary metrics quoted in the text of the article.

1. Size of the final training table (Section "Final training data"): records
   per source inventory, NFI3 records with fccarb >= 20, and the records of
   every inventory/target subset after the training filters (the same
   filters as the training scripts, see scripts/export_test_indices.py).
2. Number of distinct species/genera (especie_id) in the training table.
3. R2 and relative RMSE (RMSE / mean observed, %) of the global LightGBM model
   for c4 on its test set, overall and for prediction horizons of up to
   6 years (periodo_agrupado <= 6), used in the comparison with the literature.

Inputs: data/processed/ifn_entrenamiento_year_agrupado_periodo_completado_2.csv
and tables/supplementary/preds_c4_Global_Full.parquet (test-set predictions
written by scripts/tables/generar_tablas_y_predicciones.py).
Output: data/revision/dataset_summary.csv (metric, value).

Usage:
    python scripts/revision/dataset_summary.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from export_test_indices import CSV_FILE, select_rows  # noqa: E402

PRED_C4 = REPO_ROOT / "tables" / "supplementary" / "preds_c4_Global_Full.parquet"
OUT_CSV = REPO_ROOT / "data" / "revision" / "dataset_summary.csv"
SHORT_HORIZON = 6  # years


def r2_rmse_pct(y, p):
    rmse = np.sqrt(np.mean((y - p) ** 2))
    r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    return r2, 100 * rmse / y.mean()


def main():
    rows = []
    add = lambda metric, value: rows.append({"metric": metric, "value": value})

    df = pd.read_csv(CSV_FILE, usecols=["ifn_id", "parcela_id", "especie_id", "c",
                                        "c4", "carbono_bruto4", "fccarb"])
    inv = df["ifn_id"].astype(str)
    nfi3 = df[inv.str.contains("3")]
    add("records_total", len(df))
    add("records_NFI2", int(inv.str.contains("2").sum()))
    add("records_NFI3", len(nfi3))
    add("records_NFI3_fccarb_ge_20", int((nfi3["fccarb"] >= 20).sum()))
    for name, code in (("NFI2", "2"), ("NFI3", "3"), ("NFI2+3", None)):
        for target in ("c4", "carbono_bruto4"):
            add(f"records_{name}_{target}_after_filters", len(select_rows(df, code, target)))
    add("distinct_species", int(df["especie_id"].nunique()))

    pred = pd.read_parquet(PRED_C4)
    y = pred["y_true"].to_numpy(dtype=float)
    p = pred["LightGBM$_{\\text{global}}$"].to_numpy(dtype=float)
    short = pred["periodo_agrupado"].to_numpy() <= SHORT_HORIZON
    for label, mask in (("all", np.ones(len(y), bool)), (f"horizon_le_{SHORT_HORIZON}y", short)):
        r2, rmse_pct = r2_rmse_pct(y[mask], p[mask])
        add(f"LightGBM_global_c4_{label}_n", int(mask.sum()))
        add(f"LightGBM_global_c4_{label}_R2", round(r2, 4))
        add(f"LightGBM_global_c4_{label}_RMSE_pct", round(rmse_pct, 2))

    out = pd.DataFrame(rows, dtype=object)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved: {OUT_CSV.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
