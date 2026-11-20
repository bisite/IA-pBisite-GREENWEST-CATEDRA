"""Correlation among the 44 selected predictors.

Spearman correlation (numeric + ordinal predictors), VIF (numeric + ordinal),
Cramér's V (nominal predictors). Same filtering and train split as
catedra/training/scripts/entrenamiento_ifn2_3_c4.py (global NFI2+NFI3, target c4).

Usage: python correlacion_44_variables.py  (run from scripts/revision/)
"""
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.model_selection import GroupShuffleSplit

RUTA = "../../data/processed/ifn_entrenamiento_year_agrupado_periodo_completado_2.csv"
OUT_FIG = "../../figures/revision"
OUT_TAB = "../../data/revision"
TARGET = "c4"

FEATURES = [
    'especie_id', 'tipo_especie', 'periodo_agrupado', 'radio', 'ocupa',
    'npies_1', 'npies_2', 'npies_5', 'npies_10', 'npies_15', 'npies_20',
    'npies_25', 'npies_30', 'npies_35', 'npies_40', 'npies_45', 'npies_50',
    'npies_55', 'npies_60', 'npies_65', 'npies_70',
    'estado_id', 'fccarb', 'disesp_id',
    'modcomb_id', 'nivel2_id', 'tratmasa_id',
    'rocosidad_id', 'orientacion_cat', 'elevacion', 'pendiente_cat',
    'martonneidx_id',
    'skt_mean_primavera', 'pr_sum_primavera', 'skt_std_verano', 'skt_std_primavera',
    'pr_sum_otoño', 'skt_mean_verano', 'pr_sum_verano', 'pr_sum_invierno',
    'gndvi_mean_verano', 'ndii_mean_primavera', 'gndvi_std_primavera', 'evi_mean_primavera'
]
# Nominal (unordered) categorical predictors -> Cramér's V instead of Spearman
NOMINAL = ['especie_id', 'tipo_especie', 'estado_id', 'tratmasa_id',
           'nivel2_id', 'modcomb_id', 'disesp_id']
# orientacion_cat is circular (N..NO): kept out of Spearman/VIF, reported with Cramér's V
CIRCULAR = ['orientacion_cat']

MAPA_PENDIENTE_INT = {'muy suave (<5°)': 0, 'suave (5-10°)': 1, 'moderada (10-15°)': 2,
                      'fuerte (15-20°)': 3, 'muy fuerte (20-30°)': 4,
                      'escarpada (30-50°)': 5, 'extrema (>50°)': 6}
MAPA_ORIENTACION_INT = {'N': 0, 'NE': 1, 'E': 2, 'SE': 3, 'S': 4, 'SO': 5, 'O': 6, 'NO': 7}


def cramers_v(a, b):
    ct = pd.crosstab(a, b)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return np.nan
    chi2 = chi2_contingency(ct, correction=False)[0]
    n = ct.values.sum()
    r, k = ct.shape
    # bias-corrected (Bergsma 2013)
    phi2 = max(0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    return np.sqrt(phi2 / max(1e-12, min(kc - 1, rc - 1)))


def main():
    os.makedirs(OUT_FIG, exist_ok=True)
    os.makedirs(OUT_TAB, exist_ok=True)
    usecols = list(dict.fromkeys(FEATURES + [TARGET, 'c', 'parcela_id']))
    df = pd.read_csv(RUTA, usecols=lambda c: c in usecols)
    df['pendiente_cat'] = df['pendiente_cat'].map(MAPA_PENDIENTE_INT)
    df['orientacion_cat'] = df['orientacion_cat'].map(MAPA_ORIENTACION_INT)
    # identical filters to the training script
    df = df[(df[TARGET].notna()) & (df[TARGET] != 0) & ((df['c'].isna()) | (df['c'] < df[TARGET]))]
    df = df[~(df['fccarb'].notna() & (df['fccarb'] < 20))]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, _ = next(gss.split(df, groups=df['parcela_id']))
    tr = df.iloc[tr_idx]
    print(f"train rows: {len(tr)}; features: {len(FEATURES)}")

    num = [c for c in FEATURES if c not in NOMINAL + CIRCULAR]
    X = tr[num].astype(float)
    X[[c for c in num if c.startswith('npies_')]] = X[[c for c in num if c.startswith('npies_')]].fillna(0)

    # --- Spearman
    rho = X.corr(method='spearman')
    rho.to_csv(f"{OUT_TAB}/spearman_44.csv")
    pairs = []
    for a, b in itertools.combinations(num, 2):
        pairs.append((a, b, rho.loc[a, b]))
    pairs = pd.DataFrame(pairs, columns=['var1', 'var2', 'rho']).assign(abs_rho=lambda d: d.rho.abs())
    pairs = pairs.sort_values('abs_rho', ascending=False)
    pairs.to_csv(f"{OUT_TAB}/spearman_pairs_44.csv", index=False)

    # --- VIF (on rank-transformed, standardised data; complete cases)
    Xc = X.dropna()
    R = np.corrcoef(Xc.rank().values, rowvar=False)
    vif = pd.Series(np.diag(np.linalg.pinv(R)), index=num, name='VIF').sort_values(ascending=False)
    vif.to_csv(f"{OUT_TAB}/vif_44.csv")

    # --- Cramér's V (nominal + circular)
    cat = NOMINAL + CIRCULAR
    cv = pd.DataFrame(np.eye(len(cat)), index=cat, columns=cat)
    for a, b in itertools.combinations(cat, 2):
        sub = tr[[a, b]].dropna()
        cv.loc[a, b] = cv.loc[b, a] = cramers_v(sub[a], sub[b])
    cv.to_csv(f"{OUT_TAB}/cramersv_nominal.csv")

    # --- figure
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(rho.values, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(num)), num, rotation=90, fontsize=7)
    ax.set_yticks(range(len(num)), num, fontsize=7)
    for i in range(len(num)):
        for j in range(len(num)):
            if i != j and abs(rho.values[i, j]) > 0.7:
                ax.text(j, i, f"{rho.values[i, j]:.2f}", ha='center', va='center', fontsize=5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman's ρ")
    ax.set_title("Spearman correlation among numeric and ordinal predictors (training set, NFI2+NFI3)")
    fig.tight_layout()
    fig.savefig(f"{OUT_FIG}/spearman_44_heatmap.pdf")
    fig.savefig(f"{OUT_FIG}/spearman_44_heatmap.png", dpi=200)

    # --- summary
    print(f"\nnumeric/ordinal predictors: {len(num)}; nominal/circular: {len(cat)}")
    print(f"max |rho| = {pairs.abs_rho.max():.3f}")
    for t in (0.9, 0.8, 0.7):
        print(f"pairs |rho| > {t}: {(pairs.abs_rho > t).sum()}")
    print("\nTop 15 pairs:\n", pairs.head(15).to_string(index=False))
    print("\nVIF (top 15):\n", vif.head(15).round(2).to_string())
    print(f"VIF > 10: {(vif > 10).sum()}  VIF > 5: {(vif > 5).sum()}")
    off = cv.where(~np.eye(len(cat), dtype=bool)).stack().sort_values(ascending=False)
    print("\nCramér's V top 8:\n", off.iloc[::2].head(8).round(3).to_string())


if __name__ == "__main__":
    main()
