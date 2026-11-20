#!/usr/bin/env python3
"""Final step of the data pipeline: grouped period and completion of plot-level variables.

Input:  data/processed/ifn_entrenamiento.csv  (exported by 4.Extraer_BBDD.ipynb)
Output: data/processed/ifn_entrenamiento_year_agrupado_periodo_completado_2.csv
        (the training table used by catedra/training/scripts/)

1. periodo_agrupado: the period between inventories (years) is grouped as
       periodo < 6 -> 5, 7-10 -> 10, 18-21 -> 20, >= 29 -> 30,
   other values are kept.
2. Plot-level variables (soil, terrain, climate and spectral indices, year and
   period) that are missing in some records of a plot and inventory are filled
   with the first available value of the same (parcela_id, ifn_id).

Usage:
    python catedra/pipeline/4.1.Periodo_agrupado_y_completado.py [--input PATH] [--output PATH]
"""
import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = REPO_ROOT / "data" / "processed" / "ifn_entrenamiento.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "ifn_entrenamiento_year_agrupado_periodo_completado_2.csv"

vars_a_completar = [
    'tipsuelo1_id','tipsuelo2_id','tipsuelo3_id','rocosidad_id','radio',
    'orientacion','elevacion','pendiente','martonneidx_id','ano3','ano','periodo',
    'evi_max_invierno','evi_max_otoño','evi_max_primavera','evi_max_verano',
    'evi_mean_invierno','evi_mean_otoño','evi_mean_primavera','evi_mean_verano',
    'evi_median_invierno','evi_median_otoño','evi_median_primavera','evi_median_verano',
    'evi_min_invierno','evi_min_otoño','evi_min_primavera','evi_min_verano',
    'evi_std_invierno','evi_std_otoño','evi_std_primavera','evi_std_verano',
    'gndvi_max_invierno','gndvi_max_otoño','gndvi_max_primavera','gndvi_max_verano',
    'gndvi_mean_invierno','gndvi_mean_otoño','gndvi_mean_primavera','gndvi_mean_verano',
    'gndvi_median_invierno','gndvi_median_otoño','gndvi_median_primavera','gndvi_median_verano',
    'gndvi_min_invierno','gndvi_min_otoño','gndvi_min_primavera','gndvi_min_verano',
    'gndvi_std_invierno','gndvi_std_otoño','gndvi_std_primavera','gndvi_std_verano',
    'ndii_max_invierno','ndii_max_otoño','ndii_max_primavera','ndii_max_verano',
    'ndii_mean_invierno','ndii_mean_otoño','ndii_mean_primavera','ndii_mean_verano',
    'ndii_median_invierno','ndii_median_otoño','ndii_median_primavera','ndii_median_verano',
    'ndii_min_invierno','ndii_min_otoño','ndii_min_primavera','ndii_min_verano',
    'ndii_std_invierno','ndii_std_otoño','ndii_std_primavera','ndii_std_verano',
    'ndvi_max_invierno','ndvi_max_otoño','ndvi_max_primavera','ndvi_max_verano',
    'ndvi_mean_invierno','ndvi_mean_otoño','ndvi_mean_primavera','ndvi_mean_verano',
    'ndvi_median_invierno','ndvi_median_otoño','ndvi_median_primavera','ndvi_median_verano',
    'ndvi_min_invierno','ndvi_min_otoño','ndvi_min_primavera','ndvi_min_verano',
    'ndvi_std_invierno','ndvi_std_otoño','ndvi_std_primavera','ndvi_std_verano',
    'pr_max_invierno','pr_max_otoño','pr_max_primavera','pr_max_verano',
    'pr_mean_invierno','pr_mean_otoño','pr_mean_primavera','pr_mean_verano',
    'pr_min_invierno','pr_min_otoño','pr_min_primavera','pr_min_verano',
    'pr_std_invierno','pr_std_otoño','pr_std_primavera','pr_std_verano',
    'pr_sum_invierno','pr_sum_otoño','pr_sum_primavera','pr_sum_verano',
    'skt_max_invierno','skt_max_otoño','skt_max_primavera','skt_max_verano',
    'skt_max_max_invierno','skt_max_max_otoño','skt_max_max_primavera','skt_max_max_verano',
    'skt_max_mean_invierno','skt_max_mean_otoño','skt_max_mean_primavera','skt_max_mean_verano',
    'skt_max_min_invierno','skt_max_min_otoño','skt_max_min_primavera','skt_max_min_verano',
    'skt_max_std_invierno','skt_max_std_otoño','skt_max_std_primavera','skt_max_std_verano',
    'skt_mean_invierno','skt_mean_otoño','skt_mean_primavera','skt_mean_verano',
    'skt_min_invierno','skt_min_otoño','skt_min_primavera','skt_min_verano',
    'skt_min_max_invierno','skt_min_max_otoño','skt_min_max_primavera','skt_min_max_verano',
    'skt_min_mean_invierno','skt_min_mean_otoño','skt_min_mean_primavera','skt_min_mean_verano',
    'skt_min_min_invierno','skt_min_min_otoño','skt_min_min_primavera','skt_min_min_verano',
    'skt_min_std_invierno','skt_min_std_otoño','skt_min_std_primavera','skt_min_std_verano',
    'skt_std_invierno','skt_std_otoño','skt_std_primavera','skt_std_verano',
    'stl1_max_invierno','stl1_max_otoño','stl1_max_primavera','stl1_max_verano',
    'stl1_max_max_invierno','stl1_max_max_otoño','stl1_max_max_primavera','stl1_max_max_verano',
    'stl1_max_mean_invierno','stl1_max_mean_otoño','stl1_max_mean_primavera','stl1_max_mean_verano',
    'stl1_max_min_invierno','stl1_max_min_otoño','stl1_max_min_primavera','stl1_max_min_verano',
    'stl1_max_std_invierno','stl1_max_std_otoño','stl1_max_std_primavera','stl1_max_std_verano',
    'stl1_mean_invierno','stl1_mean_otoño','stl1_mean_primavera','stl1_mean_verano',
    'stl1_min_invierno','stl1_min_otoño','stl1_min_primavera','stl1_min_verano',
    'stl1_min_max_invierno','stl1_min_max_otoño','stl1_min_max_primavera','stl1_min_max_verano',
    'stl1_min_mean_invierno','stl1_min_mean_otoño','stl1_min_mean_primavera','stl1_min_mean_verano',
    'stl1_min_min_invierno','stl1_min_min_otoño','stl1_min_min_primavera','stl1_min_min_verano',
    'stl1_min_std_invierno','stl1_min_std_otoño','stl1_min_std_primavera','stl1_min_std_verano',
    'stl1_std_invierno','stl1_std_otoño','stl1_std_primavera','stl1_std_verano',
    'stl2_max_invierno','stl2_max_otoño','stl2_max_primavera','stl2_max_verano',
    'stl2_max_max_invierno','stl2_max_max_otoño','stl2_max_max_primavera','stl2_max_max_verano',
    'stl2_max_mean_invierno','stl2_max_mean_otoño','stl2_max_mean_primavera','stl2_max_mean_verano',
    'stl2_max_min_invierno','stl2_max_min_otoño','stl2_max_min_primavera','stl2_max_min_verano',
    'stl2_max_std_invierno','stl2_max_std_otoño','stl2_max_std_primavera','stl2_max_std_verano',
    'stl2_mean_invierno','stl2_mean_otoño','stl2_mean_primavera','stl2_mean_verano',
    'stl2_min_invierno','stl2_min_otoño','stl2_min_primavera','stl2_min_verano',
    'stl2_min_max_invierno','stl2_min_max_otoño','stl2_min_max_primavera','stl2_min_max_verano',
    'stl2_min_mean_invierno','stl2_min_mean_otoño','stl2_min_mean_primavera','stl2_min_mean_verano',
    'stl2_min_min_invierno','stl2_min_min_otoño','stl2_min_min_primavera','stl2_min_min_verano',
    'stl2_min_std_invierno','stl2_min_std_otoño','stl2_min_std_primavera','stl2_min_std_verano',
    'stl2_std_invierno','stl2_std_otoño','stl2_std_primavera','stl2_std_verano',
    'stl3_max_invierno','stl3_max_otoño','stl3_max_primavera','stl3_max_verano',
    'stl3_max_max_invierno','stl3_max_max_otoño','stl3_max_max_primavera','stl3_max_max_verano',
    'stl3_max_mean_invierno','stl3_max_mean_otoño','stl3_max_mean_primavera','stl3_max_mean_verano',
    'stl3_max_min_invierno','stl3_max_min_otoño','stl3_max_min_primavera','stl3_max_min_verano',
    'stl3_max_std_invierno','stl3_max_std_otoño','stl3_max_std_primavera','stl3_max_std_verano',
    'stl3_mean_invierno','stl3_mean_otoño','stl3_mean_primavera','stl3_mean_verano',
    'stl3_min_invierno','stl3_min_otoño','stl3_min_primavera','stl3_min_verano',
    'stl3_min_max_invierno','stl3_min_max_otoño','stl3_min_max_primavera','stl3_min_max_verano',
    'stl3_min_mean_invierno','stl3_min_mean_otoño','stl3_min_mean_primavera','stl3_min_mean_verano',
    'stl3_min_min_invierno','stl3_min_min_otoño','stl3_min_min_primavera','stl3_min_min_verano',
    'stl3_min_std_invierno','stl3_min_std_otoño','stl3_min_std_primavera','stl3_min_std_verano',
    'stl3_std_invierno','stl3_std_otoño','stl3_std_primavera','stl3_std_verano',
    'stl4_max_invierno','stl4_max_otoño','stl4_max_primavera','stl4_max_verano',
    'stl4_max_max_invierno','stl4_max_max_otoño','stl4_max_max_primavera','stl4_max_max_verano',
    'stl4_max_mean_invierno','stl4_max_mean_otoño','stl4_max_mean_primavera','stl4_max_mean_verano',
    'stl4_max_min_invierno','stl4_max_min_otoño','stl4_max_min_primavera','stl4_max_min_verano',
    'stl4_max_std_invierno','stl4_max_std_otoño','stl4_max_std_primavera','stl4_max_std_verano',
    'stl4_mean_invierno','stl4_mean_otoño','stl4_mean_primavera','stl4_mean_verano',
    'stl4_min_invierno','stl4_min_otoño','stl4_min_primavera','stl4_min_verano',
    'stl4_min_max_invierno','stl4_min_max_otoño','stl4_min_max_primavera','stl4_min_max_verano',
    'stl4_min_mean_invierno','stl4_min_mean_otoño','stl4_min_mean_primavera','stl4_min_mean_verano',
    'stl4_min_min_invierno','stl4_min_min_otoño','stl4_min_min_primavera','stl4_min_min_verano',
    'stl4_min_std_invierno','stl4_min_std_otoño','stl4_min_std_primavera','stl4_min_std_verano',
    'stl4_std_invierno','stl4_std_otoño','stl4_std_primavera','stl4_std_verano',
    't2m_max_invierno','t2m_max_otoño','t2m_max_primavera','t2m_max_verano',
    't2m_max_max_invierno','t2m_max_max_otoño','t2m_max_max_primavera','t2m_max_max_verano',
    't2m_max_mean_invierno','t2m_max_mean_otoño','t2m_max_mean_primavera','t2m_max_mean_verano',
    't2m_max_min_invierno','t2m_max_min_otoño','t2m_max_min_primavera','t2m_max_min_verano',
    't2m_max_std_invierno','t2m_max_std_otoño','t2m_max_std_primavera','t2m_max_std_verano',
    't2m_mean_invierno','t2m_mean_otoño','t2m_mean_primavera','t2m_mean_verano',
    't2m_min_invierno','t2m_min_otoño','t2m_min_primavera','t2m_min_verano',
    't2m_min_max_invierno','t2m_min_max_otoño','t2m_min_max_primavera','t2m_min_max_verano',
    't2m_min_mean_invierno','t2m_min_mean_otoño','t2m_min_mean_primavera','t2m_min_mean_verano',
    't2m_min_min_invierno','t2m_min_min_otoño','t2m_min_min_primavera','t2m_min_min_verano',
    't2m_min_std_invierno','t2m_min_std_otoño','t2m_min_std_primavera','t2m_min_std_verano',
    't2m_std_invierno','t2m_std_otoño','t2m_std_primavera','t2m_std_verano',
    'orientacion_cat','pendiente_cat','periodo_agrupado'
]


def agrupar_periodo(p):
    if p < 6:
        return 5
    elif 7 <= p <= 10:
        return 10
    elif 18 <= p <= 21:
        return 20
    elif p >= 29:
        return 30
    else:
        return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=DEFAULT_IN)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--intermediate", type=Path, default=None,
                    help="optional path to also save the table after step 1")
    args = ap.parse_args()

    # 1) Grouped period
    df = pd.read_csv(args.input)
    df["periodo_agrupado"] = df["periodo"].apply(agrupar_periodo)
    tmp = args.intermediate or args.output.with_name(args.output.stem + "_tmp.csv")
    df.to_csv(tmp, index=False)

    # 2) Completion of plot-level variables within (parcela_id, ifn_id)
    df = pd.read_csv(tmp)
    if args.intermediate is None:
        tmp.unlink()
    cols = [c for c in vars_a_completar if c in df.columns]
    before = df[cols].isna().sum()
    for col in cols:
        df[col] = df[col].fillna(df.groupby(["parcela_id", "ifn_id"])[col].transform("first"))
    after = df[cols].isna().sum()
    changed = pd.DataFrame({"missing_before": before, "missing_after": after})
    print(changed[changed["missing_before"] != changed["missing_after"]].to_string())

    df.to_csv(args.output, index=False)
    print(f"Saved: {args.output} ({len(df):,} records)")


if __name__ == "__main__":
    main()
