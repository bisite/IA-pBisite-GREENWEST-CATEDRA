#!/usr/bin/env python3
"""Robustness check of the SHAP species ranking (Section 6.3).

Input: data/revision/species_shap_LightGBM_global_c4_test.csv, the per-species
SHAP table of the global LightGBM model for c4 on the test set, produced by
notebooks/evaluacion/exploracion/mejores_modelos_especies.ipynb (the table
behind the "species_LightGBM_global" figure). Columns:

    especie_id   NFI species code
    shap_abs     mean |SHAP| of the species one-hot column over active rows
    shap_signed  mean signed SHAP over the rows where the species is present
    n_activo     number of test rows of that species
    nombre       species name
    tipo         Conífera / Frondosa

The ranking (top 15 species increasing and decreasing carbon) is recomputed
using only the species with at least MIN_RECORDS test records, and both
rankings are printed side by side.

Usage:
    python scripts/revision/species_ranking_robustness.py [--min-records 30] [--top 15]
"""
import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SHAP_TABLE = REPO_ROOT / "data" / "revision" / "species_shap_LightGBM_global_c4_test.csv"


def ranking(df: pd.DataFrame, top: int):
    pos = df[df["shap_signed"] > 0].sort_values("shap_signed", ascending=False).head(top)
    neg = df[df["shap_signed"] < 0].sort_values("shap_signed").head(top)
    return pos, neg


def describe(label: str, df: pd.DataFrame, n_test_total: int, top: int):
    pos, neg = ranking(df, top)
    conif = lambda g: int((g["tipo"] == "Conífera").sum())
    print(f"\n=== {label} ===")
    print(f"species: {len(df)} | test records covered: {df['n_activo'].sum() / n_test_total:.1%}")
    print(f"conifers among the top {top} increasing carbon: {conif(pos)}")
    print(f"conifers among the top {top} decreasing carbon: {conif(neg)}")
    print(f"conifers among the {2 * top} species shown: {conif(pos) + conif(neg)}")
    print("increasing:", ", ".join(pos["nombre"]))
    print("decreasing:", ", ".join(neg["nombre"]))
    return pos, neg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-records", type=int, default=30)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    df = pd.read_csv(SHAP_TABLE)
    df = df[df["n_activo"] > 0]
    n_total = int(df["n_activo"].sum())

    all_pos, _ = describe("All species", df, n_total, args.top)
    rob_pos, _ = describe(f"Species with >= {args.min_records} test records",
                          df[df["n_activo"] >= args.min_records], n_total, args.top)

    dropped = sorted(set(all_pos["nombre"]) - set(rob_pos["nombre"]))
    print(f"\nSpecies leaving the 'increasing' list: {', '.join(dropped) or 'none'}")


if __name__ == "__main__":
    main()
