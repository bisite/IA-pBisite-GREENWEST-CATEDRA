#!/usr/bin/env python3
"""Export the train/test partition used by every published model.

The training scripts (catedra/training/scripts/entrenamiento_*.py) apply the
same filters and a GroupShuffleSplit by plot (``parcela_id``, test_size=0.2,
random_state=42). This script reproduces that partition without retraining
and writes one CSV per inventory/target combination to ``data/splits/``:

    row_index   index of the row in the training CSV (0-based, header excluded)
    parcela_id  plot identifier (the grouping variable)
    ifn_id      source inventory of the row
    split       "train" or "test"

Usage:
    python scripts/export_test_indices.py [--check-artifacts]

With --check-artifacts, the exported test indices are compared against the
``test_indices_real.joblib`` files saved next to the models (where present).
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_FILE = REPO_ROOT / "data" / "processed" / "ifn_entrenamiento_year_agrupado_periodo_completado_2.csv"
OUT_DIR = REPO_ROOT / "data" / "splits"
ARTIFACTS_DIR = REPO_ROOT / "models" / "artifacts"

# inventory folder -> substring that ifn_id must contain (None = all rows)
INVENTORIES = {"IFN2": "2", "IFN3": "3", "IFN2-3": None}
TARGETS = ["c4", "carbono_bruto4"]


def select_rows(df_all: pd.DataFrame, inv_filter, target: str) -> pd.DataFrame:
    """Same row filters as the training scripts."""
    df = df_all
    if inv_filter is not None:
        df = df[df["ifn_id"].astype(str).str.contains(inv_filter)]
    df = df[df[target].notna() & (df[target] != 0)]
    df = df[df["c"].isna() | (df["c"] < df[target])]
    df = df[~(df["fccarb"].notna() & (df["fccarb"] < 20))]
    return df


def split(df: pd.DataFrame) -> pd.Series:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, te_pos = next(gss.split(df, groups=df["parcela_id"]))
    labels = pd.Series("train", index=df.index)
    labels.iloc[te_pos] = "test"
    return labels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-artifacts", action="store_true")
    args = ap.parse_args()

    cols = ["ifn_id", "parcela_id", "c", "fccarb"] + TARGETS
    df_all = pd.read_csv(CSV_FILE, usecols=cols)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for inv, inv_filter in INVENTORIES.items():
        for target in TARGETS:
            df = select_rows(df_all, inv_filter, target)
            labels = split(df)
            out = pd.DataFrame({
                "row_index": df.index,
                "parcela_id": df["parcela_id"].values,
                "ifn_id": df["ifn_id"].values,
                "split": labels.values,
            })
            path = OUT_DIR / f"split_{inv}_{target}.csv"
            out.to_csv(path, index=False)
            n_test = (labels == "test").sum()
            msg = f"{inv:7s} {target:15s} rows={len(df):7d} train={len(df) - n_test:7d} test={n_test:6d} -> {path.relative_to(REPO_ROOT)}"

            if args.check_artifacts:
                ref = ARTIFACTS_DIR / inv / target / "base" / "test_indices_real.joblib"
                if ref.exists():
                    import joblib
                    ref_idx = pd.Index(joblib.load(ref))
                    test_idx = df.index[labels.values == "test"]
                    same = ref_idx.sort_values().equals(test_idx.sort_values())
                    msg += f" | matches {ref.name}: {same}"
                else:
                    msg += " | no saved indices to compare"
            print(msg)


if __name__ == "__main__":
    main()
