# -*- coding: utf-8 -*-
"""
Statistics Charts Generator - For Scientific Publication
Uses pre-calculated CSVs. No titles. English labels. Large fonts.
Individual metric PDFs for single-model charts with horizontal dashed reference lines.
"""
import numpy as np
import pandas as pd
import os
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Publication-quality settings
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.family': 'serif'
})

# ==================== CONFIGURATION ====================
# ==================== CONFIGURATION ====================
from pathlib import Path
REPO_BASE = Path(__file__).resolve().parents[2]  # .../article-ecof
data_dir = str(REPO_BASE / "data/estadisticas_por_rango")
graficos_dir = str(REPO_BASE / "figures/estadisticas_por_rango")
os.makedirs(graficos_dir, exist_ok=True)

# Load data
print("=" * 100)
print("📊 STATISTICS CHARTS GENERATOR (PDF - PUBLICATION QUALITY)")
print("=" * 100)
print("\n📂 Loading data...")
results_df = pd.read_csv(os.path.join(data_dir, 'estadisticas_por_rango_todos.csv'))
print(f"   Records loaded: {len(results_df)}")

# Load original data for training distribution calculation
from sklearn.model_selection import GroupShuffleSplit
ruta_csv = str(REPO_BASE / "data/processed/ifn_entrenamiento_year_agrupado_periodo_completado_2.csv")
# Training-set distribution per bin, stored so the figures can be rebuilt without the training table
TRAIN_DIST_CSV = os.path.join(data_dir, 'training_distribution.csv')
df_original = pd.read_csv(ruta_csv) if os.path.exists(ruta_csv) else None
if df_original is not None:
    print(f"   Original data loaded: {len(df_original)} rows")

# Bins definition (must match generar_estadisticas_escalas.py)
# C4 Bins (Standard)
BINS_DEF_C4 = [(0, 20), (20, 50), (50, 100), (100, 150), (150, 200), (200, 300), 
               (300, 400), (400, 500), (500, 600), (600, 700), (700, 800), (800, None)]
BIN_LABELS_C4 = ['0-20', '20-50', '50-100', '100-150', '150-200', '200-300', 
                 '300-400', '400-500', '500-600', '600-700', '700-800', '800+']

# CB4 Bins (finer at low values: 0-20, 20-40, 40-70, 70-100, 100-150, 150-200)
BINS_DEF_CB4 = [(0, 20), (20, 40), (40, 70), (70, 100), (100, 150), (150, 200), (200, 300),
                (300, 400), (400, 500), (500, 600), (600, 700), (700, 800), (800, None)]
BIN_LABELS_CB4 = ['0-20', '20-40', '40-70', '70-100', '100-150', '150-200', '200-300',
                  '300-400', '400-500', '500-600', '600-700', '700-800', '800+']

def calcular_distribucion_training(target, bins_def):
    """Calculate percentage of training data in each bin, split by IFN."""
    df_work = df_original[(df_original[target].notna()) & (df_original[target] != 0)].copy()
    df_work = df_work[(df_work['c'].isna()) | (df_work['c'] < df_work[target])]
    df_work = df_work[~(df_work['fccarb'].notna() & (df_work['fccarb'] < 20))]
    
    # Identify IFN
    df_work['is_ifn3'] = df_work['dataset'].astype(str).str.contains('3') if 'dataset' in df_work.columns else df_work['ifn_id'].astype(str).str.contains('3')

    y = df_work[target]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, _ = next(gss.split(df_work, y, groups=df_work['parcela_id']))
    df_train = df_work.iloc[tr_idx]
    n_train = len(df_train)
    
    pct_total = []
    pct_ifn2 = []
    pct_ifn3 = []
    
    for (min_val, max_val) in bins_def:
        if max_val is None:
            mask = df_train[target] >= min_val
        else:
            mask = (df_train[target] >= min_val) & (df_train[target] < max_val)
        
        bin_data = df_train[mask]
        
        # Calculate percentages relative to TOTAL training set size
        p_tot = len(bin_data) / n_train * 100
        p_ifn3 = bin_data['is_ifn3'].sum() / n_train * 100
        p_ifn2 = (len(bin_data) - bin_data['is_ifn3'].sum()) / n_train * 100
        
        pct_total.append(p_tot)
        pct_ifn2.append(p_ifn2)
        pct_ifn3.append(p_ifn3)
    
    return pct_total, pct_ifn2, pct_ifn3, n_train

# Pre-calculate training distributions
print("   Calculating training distributions...")
if df_original is not None:
    train_dist_c4, train_dist_c4_ifn2, train_dist_c4_ifn3, n_train_c4 = calcular_distribucion_training('c4', BINS_DEF_C4)
    train_dist_cb4, train_dist_cb4_ifn2, train_dist_cb4_ifn3, n_train_cb4 = calcular_distribucion_training('carbono_bruto4', BINS_DEF_CB4)
    pd.DataFrame(
        [{'target': t, 'bin': lab, 'pct_total': a, 'pct_ifn2': b, 'pct_ifn3': c, 'n_train': n}
         for t, labs, (tot, i2, i3, n) in (
             ('c4', BIN_LABELS_C4, (train_dist_c4, train_dist_c4_ifn2, train_dist_c4_ifn3, n_train_c4)),
             ('carbono_bruto4', BIN_LABELS_CB4, (train_dist_cb4, train_dist_cb4_ifn2, train_dist_cb4_ifn3, n_train_cb4)))
         for lab, a, b, c in zip(labs, tot, i2, i3)]
    ).to_csv(TRAIN_DIST_CSV, index=False, float_format='%.17g')
else:
    print(f"   Training table not found; using {TRAIN_DIST_CSV}")
    _td = pd.read_csv(TRAIN_DIST_CSV)
    def _dist(t):
        d = _td[_td['target'] == t]
        return list(d['pct_total']), list(d['pct_ifn2']), list(d['pct_ifn3']), int(d['n_train'].iloc[0])
    train_dist_c4, train_dist_c4_ifn2, train_dist_c4_ifn3, n_train_c4 = _dist('c4')
    train_dist_cb4, train_dist_cb4_ifn2, train_dist_cb4_ifn3, n_train_cb4 = _dist('carbono_bruto4')
print(f"   C4 training: {n_train_c4:,} samples")
print(f"   Carbono_bruto4 training: {n_train_cb4:,} samples")

# Color configuration
colores_base = plt.cm.Set1(np.linspace(0, 1, 10))

# Different line styles and markers to distinguish overlapping lines
line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'X', 'P']

metricas_a_graficar = ['R2', 'RMSE', 'MAE', 'wMAPE', 'SMAPE', 'Bias']

# Labels with units depending on target
def get_metricas_labels(target):
    if target == 'c4':
        unit = 'tC/ha'
    else:  # carbono_bruto4
        unit = 'tC'
    return {
        'R2': 'R²',
        'RMSE': f'RMSE ({unit})',
        'MAE': f'MAE ({unit})',
        'wMAPE': 'wMAPE (%)',
        'SMAPE': 'SMAPE (%)',
        'Bias': f'Bias ({unit})'
    }

metricas_nombres = {
    'R2': 'R2_coefficient_determination',
    'RMSE': 'RMSE_root_mean_square_error',
    'MAE': 'MAE_mean_absolute_error',
    'wMAPE': 'wMAPE_weighted_mean_absolute_percentage_error',
    'SMAPE': 'SMAPE_symmetric_mean_absolute_percentage_error',
    'Bias': 'Bias'
}

for target_name in ['c4', 'carbono_bruto4']:
    print(f"\n{'='*80}")
    print(f"📈 Generating charts for {target_name.upper()}...")
    print("="*80)
    
    # Select appropriate bins
    rangos_bin = BIN_LABELS_C4 if target_name == 'c4' else BIN_LABELS_CB4
    
    # Define accumulator ranges (assuming standard steps but maybe adjusted names)
    # Keeping standard accumulators for simplicity or defining custom if needed.
    # The generating script might use different scales for accumulators too.
    # Assuming accumulators are consistent with scales used in generation.
    # The generation script uses:
    # C4: 20, 50, 100, 150, 200...
    # CB4: 20, 40, 70, 100, 150, 200...
    
    if target_name == 'c4':
        rangos_acum = ['0-20', '0-50', '0-100', '0-150', '0-200', '0-300', '0-400', '0-500', '0-600', '0-700', '0-800', 'Full']
        rangos_acum_csv = ['0-20', '0-50', '0-100', '0-150', '0-200', '0-300', '0-400', '0-500', '0-600', '0-700', '0-800', 'Completo']
    else:
        rangos_acum = ['0-20', '0-40', '0-70', '0-100', '0-150', '0-200', '0-300', '0-400', '0-500', '0-600', '0-700', '0-800', 'Full']
        rangos_acum_csv = ['0-20', '0-40', '0-70', '0-100', '0-150', '0-200', '0-300', '0-400', '0-500', '0-600', '0-700', '0-800', 'Completo']
    
    # Get labels with correct units for this target
    metricas_labels = get_metricas_labels(target_name)
    
    # Unit for X axis labels
    if target_name == 'c4':
        unit = 'tC/ha'
    else:
        unit = 'tC'
    x_label_range = f'Range of real values ({unit})'
    x_label_bin = f'Bin of real values ({unit})'
    
    target_graficos_dir = os.path.join(graficos_dir, target_name)
    os.makedirs(target_graficos_dir, exist_ok=True)
    
    df_target = results_df[results_df['Target'] == target_name]
    modelos_base = df_target[df_target['Tipo'] == 'base']['Modelo'].unique().tolist()
    modelos_stack = df_target[df_target['Tipo'] == 'stacking']['Modelo'].unique().tolist()
    
    df_acum = df_target[df_target['TipoRango'] == 'Acumulado']
    df_bin = df_target[df_target['TipoRango'] == 'Bin']
    
    x_pos = list(range(len(rangos_acum)))
    x_pos_bin = list(range(len(rangos_bin)))
    
    print(f"   Base models: {len(modelos_base)}")
    print(f"   Stacking models: {len(modelos_stack)}")
    
    # =====================================================
    # COMPARATIVE CHARTS - BASE MODELS (multi-model, no final values)
    # =====================================================
    print(f"\n   📊 Comparative charts - Base Models...")
    base_dir = os.path.join(target_graficos_dir, 'base_models')
    os.makedirs(base_dir, exist_ok=True)
    
    for metrica in metricas_a_graficar:
        # Cumulative range
        fig, ax = plt.subplots(figsize=(14, 8))
        for i, modelo in enumerate(modelos_base):
            df_modelo = df_acum[df_acum['Modelo'] == modelo]
            valores = []
            for rango in rangos_acum_csv:
                df_r = df_modelo[df_modelo['Rango'] == rango]
                if len(df_r) > 0 and not pd.isna(df_r[metrica].iloc[0]):
                    valores.append(df_r[metrica].iloc[0])
                else:
                    valores.append(np.nan)
            ax.plot(x_pos, valores, marker=markers[i % len(markers)], linestyle=line_styles[i % len(line_styles)], 
                    label=modelo, color=colores_base[i % len(colores_base)], linewidth=2.5, markersize=8, alpha=0.85)
        
        ax.set_xlabel(x_label_range, fontsize=16)
        ax.set_ylabel(metricas_labels[metrica], fontsize=16)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(rangos_acum, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        filename = f'{target_name}_base_models_{metricas_nombres[metrica]}_cumulative_range.pdf'
        plt.savefig(os.path.join(base_dir, filename), format='pdf', bbox_inches='tight')
        plt.close()
        
        # Separate bins - with training distribution histogram on top
        train_dist = train_dist_c4 if target_name == 'c4' else train_dist_cb4
        n_train = n_train_c4 if target_name == 'c4' else n_train_cb4
        
        fig, (ax_hist, ax) = plt.subplots(2, 1, figsize=(14, 10), 
                                           gridspec_kw={'height_ratios': [1, 3]}, sharex=True)
        
        # Top histogram - training data distribution
        ax_hist.bar(x_pos_bin, train_dist, color='#7f8c8d', edgecolor='black', alpha=0.7)
        for i, (x, pct) in enumerate(zip(x_pos_bin, train_dist)):
            if pct > 0.5:  # Only show label if > 0.5%
                ax_hist.text(x, pct + 0.5, f'{pct:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax_hist.set_ylabel('Training Data (%)', fontsize=14)
        ax_hist.set_ylim(0, max(train_dist) * 1.3)
        ax_hist.grid(True, alpha=0.3, axis='y')
        ax_hist.text(0.02, 0.95, f'N train = {n_train:,}', transform=ax_hist.transAxes, 
                     fontsize=11, verticalalignment='top', 
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Bottom plot - metrics lines
        for i, modelo in enumerate(modelos_base):
            df_modelo = df_bin[df_bin['Modelo'] == modelo]
            valores = []
            for rango in rangos_bin:
                df_r = df_modelo[df_modelo['Rango'] == rango]
                if len(df_r) > 0 and not pd.isna(df_r[metrica].iloc[0]):
                    valores.append(df_r[metrica].iloc[0])
                else:
                    valores.append(np.nan)
            ax.plot(x_pos_bin, valores, marker=markers[i % len(markers)], linestyle=line_styles[i % len(line_styles)],
                    label=modelo, color=colores_base[i % len(colores_base)], linewidth=2.5, markersize=8, alpha=0.85)
        
        ax.set_xlabel(x_label_bin, fontsize=16)
        ax.set_ylabel(metricas_labels[metrica], fontsize=16)
        ax.set_xticks(x_pos_bin)
        ax.set_xticklabels(rangos_bin, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'{target_name}_base_models_{metricas_nombres[metrica]}_separate_bins.pdf'
        plt.savefig(os.path.join(base_dir, filename), format='pdf', bbox_inches='tight')
        plt.close()
    
    # === SPECIAL COMBINED CHART: SMAPE + RMSE ===
    print(f"   📊 Generating combined SMAPE + RMSE chart...")
    train_dist = train_dist_c4 if target_name == 'c4' else train_dist_cb4
    train_dist_ifn2 = train_dist_c4_ifn2 if target_name == 'c4' else train_dist_cb4_ifn2
    train_dist_ifn3 = train_dist_c4_ifn3 if target_name == 'c4' else train_dist_cb4_ifn3
    n_train = n_train_c4 if target_name == 'c4' else n_train_cb4
    unit = 'tC/ha' if target_name == 'c4' else 'tC'
    
    # Limit bins to specific range for better visualization
    # 6 bins for both targets for consistency
    max_bin_idx = 5  # Index 5 -> '200-300' (Total 6 bins: 0-20 to 200-300)
        
    rangos_bin_limited = rangos_bin[:max_bin_idx + 1]
    x_pos_bin_limited = list(range(len(rangos_bin_limited)))
    train_dist_limited = train_dist[:max_bin_idx + 1]
    dist_ifn2_lim = train_dist_ifn2[:max_bin_idx + 1]
    dist_ifn3_lim = train_dist_ifn3[:max_bin_idx + 1]
    
    # Figure size for publication
    fig, (ax_hist, ax) = plt.subplots(2, 1, figsize=(8, 8), 
                                       gridspec_kw={'height_ratios': [1, 3]}, sharex=True)
    
    # Top histogram - STACKED BARS (IFN2/IFN3)
    ax_hist.bar(x_pos_bin_limited, dist_ifn2_lim, color='#e67e22', edgecolor='black', alpha=0.7, label='IFN2')
    ax_hist.bar(x_pos_bin_limited, dist_ifn3_lim, bottom=dist_ifn2_lim, color='#3498db', edgecolor='black', alpha=0.7, label='IFN3')
    
    # Add text labels for total percentage
    for i, (x, pct) in enumerate(zip(x_pos_bin_limited, train_dist_limited)):
        if pct > 0.5:
            ax_hist.text(x, pct + 1.0, f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
    ax_hist.set_ylabel('Training Data (%)', fontsize=12)
    ax_hist.set_ylim(0, max(train_dist_limited) * 1.45) # Más margen vertical
    ax_hist.grid(True, alpha=0.3, axis='y')
    
    # Leyenda IFN2/IFN3 arriba a la derecha, dentro
    ax_hist.legend(loc='upper right', fontsize=9, frameon=True, facecolor='white', framealpha=0.9)
    
    # N train text: arriba a la izquierda, en coordenadas de ejes
    ax_hist.text(0.02, 0.92, f'N train = {n_train:,}', transform=ax_hist.transAxes, 
                 fontsize=10, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Create second Y axis for RMSE
    ax2 = ax.twinx()
    
    # Filter out models with extreme values for better scale
    modelos_filtered = [m for m in modelos_base if m not in ['SVR', 'BayesianNN']]
    
    # Plot SMAPE on left axis (limited bins)
    for i, modelo in enumerate(modelos_filtered):
        df_modelo = df_bin[df_bin['Modelo'] == modelo]
        valores_smape = []
        for rango in rangos_bin_limited:
            df_r = df_modelo[df_modelo['Rango'] == rango]
            if len(df_r) > 0 and not pd.isna(df_r['SMAPE'].iloc[0]):
                valores_smape.append(df_r['SMAPE'].iloc[0])
            else:
                valores_smape.append(np.nan)
        ax.plot(x_pos_bin_limited, valores_smape, marker=markers[i % len(markers)], linestyle='-',
                label=f'{modelo}', color=colores_base[i % len(colores_base)], linewidth=2, markersize=6, alpha=0.85)
    
    # Plot RMSE on right axis (dashed lines, same colors, limited bins)
    for i, modelo in enumerate(modelos_filtered):
        df_modelo = df_bin[df_bin['Modelo'] == modelo]
        valores_rmse = []
        for rango in rangos_bin_limited:
            df_r = df_modelo[df_modelo['Rango'] == rango]
            if len(df_r) > 0 and not pd.isna(df_r['RMSE'].iloc[0]):
                valores_rmse.append(df_r['RMSE'].iloc[0])
            else:
                valores_rmse.append(np.nan)
        ax2.plot(x_pos_bin_limited, valores_rmse, marker=markers[i % len(markers)], linestyle='--',
                 color=colores_base[i % len(colores_base)], linewidth=2, markersize=6, alpha=0.6)
    
    ax.set_xlabel(x_label_bin, fontsize=14)
    ax.set_ylabel('SMAPE (%) — solid lines', fontsize=12, color='black')
    ax2.set_ylabel(f'RMSE ({unit}) — dashed lines', fontsize=12, color='gray')
    ax.set_xticks(x_pos_bin_limited)
    ax.set_xticklabels(rangos_bin_limited, rotation=45, ha='right', fontsize=11)
    
    # Leyenda de modelos DEBAJO del gráfico
    ax.legend(bbox_to_anchor=(0.5, -0.25), loc='upper center', fontsize=10, ncol=3, frameon=False)
    
    ax.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)  # Fixed RMSE scale 0-100
    ax2.tick_params(axis='y', labelcolor='gray')
    
    # Ajustar márgenes para que quepa la leyenda de abajo y no se corte arriba
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20, top=0.95)
    
    filename = f'{target_name}_base_models_SMAPE_RMSE_combined_bins.pdf'
    plt.savefig(os.path.join(base_dir, filename), format='pdf', bbox_inches='tight')
    plt.close()
    print(f"      ✅ Combined SMAPE+RMSE chart saved")
    
    print(f"      ✅ {len(metricas_a_graficar)*2 + 1} comparative PDF charts")
    
    # =====================================================
    # COMPARATIVE CHARTS - STACKING MODELS
    # =====================================================
    print(f"   📊 Comparative charts - Stacking Models...")
    stack_dir = os.path.join(target_graficos_dir, 'stacking_models')
    os.makedirs(stack_dir, exist_ok=True)
    
    for stack_num in range(1, 6):
        stack_modelos = [m for m in modelos_stack if m.startswith(f'Stack{stack_num}_')]
        if not stack_modelos:
            continue
        
        stack_subdir = os.path.join(stack_dir, f'Stack{stack_num}')
        os.makedirs(stack_subdir, exist_ok=True)
        
        for metrica in metricas_a_graficar:
            # Cumulative range
            fig, ax = plt.subplots(figsize=(14, 8))
            for i, modelo in enumerate(stack_modelos):
                df_modelo = df_acum[df_acum['Modelo'] == modelo]
                valores = []
                for rango in rangos_acum_csv:
                    df_r = df_modelo[df_modelo['Rango'] == rango]
                    if len(df_r) > 0 and not pd.isna(df_r[metrica].iloc[0]):
                        valores.append(df_r[metrica].iloc[0])
                    else:
                        valores.append(np.nan)
                label = modelo.replace(f'Stack{stack_num}_', '')
                ax.plot(x_pos, valores, marker=markers[i % len(markers)], linestyle=line_styles[i % len(line_styles)],
                       label=label, color=colores_base[i % len(colores_base)], linewidth=2.5, markersize=8, alpha=0.85)
            
            ax.set_xlabel(x_label_range, fontsize=16)
            ax.set_ylabel(metricas_labels[metrica], fontsize=16)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(rangos_acum, rotation=45, ha='right')
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            filename = f'{target_name}_Stack{stack_num}_{metricas_nombres[metrica]}_cumulative_range.pdf'
            plt.savefig(os.path.join(stack_subdir, filename), format='pdf', bbox_inches='tight')
            plt.close()
            
            # Separate bins - with training distribution histogram on top
            train_dist = train_dist_c4 if target_name == 'c4' else train_dist_cb4
            n_train = n_train_c4 if target_name == 'c4' else n_train_cb4
            
            fig, (ax_hist, ax) = plt.subplots(2, 1, figsize=(14, 10), 
                                               gridspec_kw={'height_ratios': [1, 3]}, sharex=True)
            
            # Top histogram - training data distribution
            ax_hist.bar(x_pos_bin, train_dist, color='#7f8c8d', edgecolor='black', alpha=0.7)
            for ii, (x, pct) in enumerate(zip(x_pos_bin, train_dist)):
                if pct > 0.5:
                    ax_hist.text(x, pct + 0.5, f'{pct:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax_hist.set_ylabel('Training Data (%)', fontsize=14)
            ax_hist.set_ylim(0, max(train_dist) * 1.3)
            ax_hist.grid(True, alpha=0.3, axis='y')
            ax_hist.text(0.02, 0.95, f'N train = {n_train:,}', transform=ax_hist.transAxes, 
                         fontsize=11, verticalalignment='top', 
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Bottom plot - metrics lines
            for i, modelo in enumerate(stack_modelos):
                df_modelo = df_bin[df_bin['Modelo'] == modelo]
                valores = []
                for rango in rangos_bin:
                    df_r = df_modelo[df_modelo['Rango'] == rango]
                    if len(df_r) > 0 and not pd.isna(df_r[metrica].iloc[0]):
                        valores.append(df_r[metrica].iloc[0])
                    else:
                        valores.append(np.nan)
                label = modelo.replace(f'Stack{stack_num}_', '')
                ax.plot(x_pos_bin, valores, marker=markers[i % len(markers)], linestyle=line_styles[i % len(line_styles)],
                       label=label, color=colores_base[i % len(colores_base)], linewidth=2.5, markersize=8, alpha=0.85)
            
            ax.set_xlabel(x_label_bin, fontsize=16)
            ax.set_ylabel(metricas_labels[metrica], fontsize=16)
            ax.set_xticks(x_pos_bin)
            ax.set_xticklabels(rangos_bin, rotation=45, ha='right')
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            filename = f'{target_name}_Stack{stack_num}_{metricas_nombres[metrica]}_separate_bins.pdf'
            plt.savefig(os.path.join(stack_subdir, filename), format='pdf', bbox_inches='tight')
            plt.close()
        
        print(f"      ✅ Stack{stack_num}: {len(metricas_a_graficar)*2} PDF charts")
    
    # =====================================================
    # INDIVIDUAL CHARTS - SINGLE MODEL (with dashed horizontal line for final value)
    # Each metric = one PDF file, in folder named by model
    # =====================================================
    print(f"   📊 Individual single-model charts (with reference lines)...")
    
    # Base models
    base_individual_dir = os.path.join(base_dir, 'individual_models')
    os.makedirs(base_individual_dir, exist_ok=True)
    
    for modelo in modelos_base:
        model_dir = os.path.join(base_individual_dir, modelo.replace("/", "_"))
        os.makedirs(model_dir, exist_ok=True)
        df_modelo = df_acum[df_acum['Modelo'] == modelo]
        
        for metrica in metricas_a_graficar:
            fig, ax = plt.subplots(figsize=(12, 7))
            valores = []
            for rango in rangos_acum_csv:
                df_r = df_modelo[df_modelo['Rango'] == rango]
                if len(df_r) > 0 and not pd.isna(df_r[metrica].iloc[0]):
                    valores.append(df_r[metrica].iloc[0])
                else:
                    valores.append(np.nan)
            
            color = '#2E7D32' if metrica == 'R2' else '#1565C0'
            ax.bar(x_pos, valores, color=color, alpha=0.7, edgecolor='black', linewidth=1.5)
            ax.plot(x_pos, valores, 'o-', color='#D32F2F', markersize=8, linewidth=2)
            
            # Horizontal dashed line at final value
            final_value = valores[-1] if not np.isnan(valores[-1]) else None
            if final_value is not None:
                ax.axhline(y=final_value, color='#D32F2F', linestyle='--', linewidth=2, alpha=0.8)
                # Text outside the plot area, to the left of Y axis
                ax.annotate(f'{final_value:.2f}', xy=(-1.2, final_value), 
                           fontsize=13, color='#D32F2F', va='center', ha='right', fontweight='bold',
                           annotation_clip=False)
            
            ax.set_xlabel(x_label_range, fontsize=16)
            ax.set_ylabel(metricas_labels[metrica], fontsize=16)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(rangos_acum, rotation=45, ha='right', fontsize=13)
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            filename = f'{target_name}_{modelo.replace("/", "_")}_{metricas_nombres[metrica]}_cumulative.pdf'
            plt.savefig(os.path.join(model_dir, filename), format='pdf', bbox_inches='tight')
            plt.close()
    
    print(f"      ✅ {len(modelos_base) * len(metricas_a_graficar)} individual PDF charts for base models")
    
    # Stacking models
    stack_individual_dir = os.path.join(stack_dir, 'individual_models')
    os.makedirs(stack_individual_dir, exist_ok=True)
    
    for modelo in modelos_stack:
        model_dir = os.path.join(stack_individual_dir, modelo.replace("/", "_"))
        os.makedirs(model_dir, exist_ok=True)
        df_modelo = df_acum[df_acum['Modelo'] == modelo]
        
        for metrica in metricas_a_graficar:
            fig, ax = plt.subplots(figsize=(12, 7))
            valores = []
            for rango in rangos_acum_csv:
                df_r = df_modelo[df_modelo['Rango'] == rango]
                if len(df_r) > 0 and not pd.isna(df_r[metrica].iloc[0]):
                    valores.append(df_r[metrica].iloc[0])
                else:
                    valores.append(np.nan)
            
            color = '#6A1B9A' if metrica == 'R2' else '#E65100'
            ax.bar(x_pos, valores, color=color, alpha=0.7, edgecolor='black', linewidth=1.5)
            ax.plot(x_pos, valores, 'o-', color='#D32F2F', markersize=8, linewidth=2)
            
            # Horizontal dashed line at final value
            final_value = valores[-1] if not np.isnan(valores[-1]) else None
            if final_value is not None:
                ax.axhline(y=final_value, color='#D32F2F', linestyle='--', linewidth=2, alpha=0.8)
                # Text outside the plot area, to the left of Y axis
                ax.annotate(f'{final_value:.2f}', xy=(-1.2, final_value), 
                           fontsize=13, color='#D32F2F', va='center', ha='right', fontweight='bold',
                           annotation_clip=False)
            
            ax.set_xlabel(x_label_range, fontsize=16)
            ax.set_ylabel(metricas_labels[metrica], fontsize=16)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(rangos_acum, rotation=45, ha='right', fontsize=13)
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            filename = f'{target_name}_{modelo.replace("/", "_")}_{metricas_nombres[metrica]}_cumulative.pdf'
            plt.savefig(os.path.join(model_dir, filename), format='pdf', bbox_inches='tight')
            plt.close()
    
    print(f"      ✅ {len(modelos_stack) * len(metricas_a_graficar)} individual PDF charts for stacking models")
    
    # =====================================================
    # HEATMAPS
    # =====================================================
    print(f"   📊 Generating heatmaps...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    heatmap_data = []
    for modelo in modelos_base:
        row = []
        df_modelo = df_acum[df_acum['Modelo'] == modelo]
        for rango in rangos_acum_csv:
            df_r = df_modelo[df_modelo['Rango'] == rango]
            if len(df_r) > 0 and not pd.isna(df_r['R2'].iloc[0]):
                row.append(df_r['R2'].iloc[0])
            else:
                row.append(np.nan)
        heatmap_data.append(row)
    
    heatmap_array = np.array(heatmap_data)
    im = ax.imshow(heatmap_array, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(rangos_acum)))
    ax.set_xticklabels(rangos_acum, rotation=45, ha='right', fontsize=13)
    ax.set_yticks(range(len(modelos_base)))
    ax.set_yticklabels(modelos_base, fontsize=13)
    ax.set_xlabel(x_label_range, fontsize=16)
    ax.set_ylabel('Model', fontsize=16)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('R²', fontsize=14)
    for i in range(len(modelos_base)):
        for j in range(len(rangos_acum)):
            val = heatmap_array[i, j]
            if not np.isnan(val):
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=11)
    plt.tight_layout()
    filename = f'{target_name}_heatmap_R2_base_models_cumulative_range.pdf'
    plt.savefig(os.path.join(base_dir, filename), format='pdf', bbox_inches='tight')
    plt.close()
    print(f"      ✅ Heatmap PDF for base models")
    
    for stack_num in range(1, 6):
        stack_modelos = [m for m in modelos_stack if m.startswith(f'Stack{stack_num}_')]
        if not stack_modelos:
            continue
        
        # Heatmap cumulative range
        fig, ax = plt.subplots(figsize=(14, 6))
        heatmap_data = []
        for modelo in stack_modelos:
            row = []
            df_modelo = df_acum[df_acum['Modelo'] == modelo]
            for rango in rangos_acum_csv:
                df_r = df_modelo[df_modelo['Rango'] == rango]
                if len(df_r) > 0 and not pd.isna(df_r['R2'].iloc[0]):
                    row.append(df_r['R2'].iloc[0])
                else:
                    row.append(np.nan)
            heatmap_data.append(row)
        
        heatmap_array = np.array(heatmap_data)
        im = ax.imshow(heatmap_array, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
        ax.set_xticks(range(len(rangos_acum)))
        ax.set_xticklabels(rangos_acum, rotation=45, ha='right', fontsize=13)
        ax.set_yticks(range(len(stack_modelos)))
        ax.set_yticklabels([m.replace(f'Stack{stack_num}_', '') for m in stack_modelos], fontsize=13)
        ax.set_xlabel(x_label_range, fontsize=16)
        ax.set_ylabel('Meta-model', fontsize=16)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('R²', fontsize=14)
        for i in range(len(stack_modelos)):
            for j in range(len(rangos_acum)):
                val = heatmap_array[i, j]
                if not np.isnan(val):
                    color = 'white' if abs(val) > 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=11)
        plt.tight_layout()
        filename = f'{target_name}_heatmap_R2_Stack{stack_num}_cumulative_range.pdf'
        plt.savefig(os.path.join(stack_dir, f'Stack{stack_num}', filename), format='pdf', bbox_inches='tight')
        plt.close()
    
    print(f"      ✅ Heatmaps PDF for stacking models (cumulative range)")
    
    # =====================================================
    # HEATMAPS - SEPARATE BINS (NON-CUMULATIVE)
    # =====================================================
    print(f"   📊 Generating heatmaps (separate bins - non-cumulative)...")
    
    # Heatmap for base models - separate bins
    fig, ax = plt.subplots(figsize=(14, 8))
    heatmap_data = []
    for modelo in modelos_base:
        row = []
        df_modelo = df_bin[df_bin['Modelo'] == modelo]
        for rango in rangos_bin:
            df_r = df_modelo[df_modelo['Rango'] == rango]
            if len(df_r) > 0 and not pd.isna(df_r['R2'].iloc[0]):
                row.append(df_r['R2'].iloc[0])
            else:
                row.append(np.nan)
        heatmap_data.append(row)
    
    heatmap_array = np.array(heatmap_data)
    im = ax.imshow(heatmap_array, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(rangos_bin)))
    ax.set_xticklabels(rangos_bin, rotation=45, ha='right', fontsize=13)
    ax.set_yticks(range(len(modelos_base)))
    ax.set_yticklabels(modelos_base, fontsize=13)
    ax.set_xlabel(x_label_bin, fontsize=16)
    ax.set_ylabel('Model', fontsize=16)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('R²', fontsize=14)
    for i in range(len(modelos_base)):
        for j in range(len(rangos_bin)):
            val = heatmap_array[i, j]
            if not np.isnan(val):
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=11)
    plt.tight_layout()
    filename = f'{target_name}_heatmap_R2_base_models_separate_bins.pdf'
    plt.savefig(os.path.join(base_dir, filename), format='pdf', bbox_inches='tight')
    plt.close()
    print(f"      ✅ Heatmap PDF for base models (separate bins)")
    
    # Heatmaps for stacking models - separate bins
    for stack_num in range(1, 6):
        stack_modelos = [m for m in modelos_stack if m.startswith(f'Stack{stack_num}_')]
        if not stack_modelos:
            continue
        
        fig, ax = plt.subplots(figsize=(14, 6))
        heatmap_data = []
        for modelo in stack_modelos:
            row = []
            df_modelo = df_bin[df_bin['Modelo'] == modelo]
            for rango in rangos_bin:
                df_r = df_modelo[df_modelo['Rango'] == rango]
                if len(df_r) > 0 and not pd.isna(df_r['R2'].iloc[0]):
                    row.append(df_r['R2'].iloc[0])
                else:
                    row.append(np.nan)
            heatmap_data.append(row)
        
        heatmap_array = np.array(heatmap_data)
        im = ax.imshow(heatmap_array, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
        ax.set_xticks(range(len(rangos_bin)))
        ax.set_xticklabels(rangos_bin, rotation=45, ha='right', fontsize=13)
        ax.set_yticks(range(len(stack_modelos)))
        ax.set_yticklabels([m.replace(f'Stack{stack_num}_', '') for m in stack_modelos], fontsize=13)
        ax.set_xlabel(x_label_bin, fontsize=16)
        ax.set_ylabel('Meta-model', fontsize=16)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('R²', fontsize=14)
        for i in range(len(stack_modelos)):
            for j in range(len(rangos_bin)):
                val = heatmap_array[i, j]
                if not np.isnan(val):
                    color = 'white' if abs(val) > 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=11)
        plt.tight_layout()
        filename = f'{target_name}_heatmap_R2_Stack{stack_num}_separate_bins.pdf'
        plt.savefig(os.path.join(stack_dir, f'Stack{stack_num}', filename), format='pdf', bbox_inches='tight')
        plt.close()
    
    print(f"      ✅ Heatmaps PDF for stacking models (separate bins - R2)")
    
    # =====================================================
    # HEATMAPS - SEPARATE BINS - RMSE (more interpretable for narrow ranges)
    # =====================================================
    print(f"   📊 Generating RMSE heatmaps (separate bins)...")
    
    # Heatmap RMSE for base models - separate bins (limited range)
    fig, ax = plt.subplots(figsize=(14, 8))
    heatmap_data = []
    for modelo in modelos_base:
        row = []
        df_modelo = df_bin[df_bin['Modelo'] == modelo]
        for rango in rangos_bin_limited:
            df_r = df_modelo[df_modelo['Rango'] == rango]
            if len(df_r) > 0 and not pd.isna(df_r['RMSE'].iloc[0]):
                row.append(df_r['RMSE'].iloc[0])
            else:
                row.append(np.nan)
        heatmap_data.append(row)
    
    heatmap_array = np.array(heatmap_data)
    # Use reversed colormap (lower RMSE = better = green)
    im = ax.imshow(heatmap_array, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(len(rangos_bin_limited)))
    ax.set_xticklabels(rangos_bin_limited, rotation=45, ha='right', fontsize=13)
    ax.set_yticks(range(len(modelos_base)))
    ax.set_yticklabels(modelos_base, fontsize=13)
    ax.set_xlabel(x_label_bin, fontsize=16)
    ax.set_ylabel('Model', fontsize=16)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'RMSE ({unit})', fontsize=14)
    for i in range(len(modelos_base)):
        for j in range(len(rangos_bin_limited)):
            val = heatmap_array[i, j]
            if not np.isnan(val):
                # Determine text color based on background intensity
                norm_val = (val - np.nanmin(heatmap_array)) / (np.nanmax(heatmap_array) - np.nanmin(heatmap_array) + 1e-10)
                color = 'white' if norm_val > 0.6 or norm_val < 0.4 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', color=color, fontsize=10)
    plt.tight_layout()
    filename = f'{target_name}_heatmap_RMSE_base_models_separate_bins.pdf'
    plt.savefig(os.path.join(base_dir, filename), format='pdf', bbox_inches='tight')
    plt.close()
    print(f"      ✅ Heatmap RMSE PDF for base models (separate bins - limited)")
    
    # Heatmaps RMSE for stacking models - separate bins
    for stack_num in range(1, 6):
        stack_modelos = [m for m in modelos_stack if m.startswith(f'Stack{stack_num}_')]
        if not stack_modelos:
            continue
        
        fig, ax = plt.subplots(figsize=(14, 6))
        heatmap_data = []
        for modelo in stack_modelos:
            row = []
            df_modelo = df_bin[df_bin['Modelo'] == modelo]
            for rango in rangos_bin_limited:
                df_r = df_modelo[df_modelo['Rango'] == rango]
                if len(df_r) > 0 and not pd.isna(df_r['RMSE'].iloc[0]):
                    row.append(df_r['RMSE'].iloc[0])
                else:
                    row.append(np.nan)
            heatmap_data.append(row)
        
        heatmap_array = np.array(heatmap_data)
        im = ax.imshow(heatmap_array, cmap='RdYlGn_r', aspect='auto')
        ax.set_xticks(range(len(rangos_bin_limited)))
        ax.set_xticklabels(rangos_bin_limited, rotation=45, ha='right', fontsize=13)
        ax.set_yticks(range(len(stack_modelos)))
        ax.set_yticklabels([m.replace(f'Stack{stack_num}_', '') for m in stack_modelos], fontsize=13)
        ax.set_xlabel(x_label_bin, fontsize=16)
        ax.set_ylabel('Meta-model', fontsize=16)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f'RMSE ({unit})', fontsize=14)
        for i in range(len(stack_modelos)):
            for j in range(len(rangos_bin_limited)):
                val = heatmap_array[i, j]
                if not np.isnan(val):
                    norm_val = (val - np.nanmin(heatmap_array)) / (np.nanmax(heatmap_array) - np.nanmin(heatmap_array) + 1e-10)
                    color = 'white' if norm_val > 0.6 or norm_val < 0.4 else 'black'
                    ax.text(j, i, f'{val:.1f}', ha='center', va='center', color=color, fontsize=10)
        plt.tight_layout()
        filename = f'{target_name}_heatmap_RMSE_Stack{stack_num}_separate_bins.pdf'
        plt.savefig(os.path.join(stack_dir, f'Stack{stack_num}', filename), format='pdf', bbox_inches='tight')
        plt.close()
    
    print(f"      ✅ Heatmaps RMSE PDF for stacking models (separate bins - limited)")
    
    # =====================================================
    # HEATMAPS - SEPARATE BINS - SMAPE (percentage error, very interpretable)
    # =====================================================
    print(f"   📊 Generating SMAPE heatmaps (separate bins)...")
    
    # Heatmap SMAPE for base models - separate bins (limited range)
    fig, ax = plt.subplots(figsize=(14, 8))
    heatmap_data = []
    for modelo in modelos_base:
        row = []
        df_modelo = df_bin[df_bin['Modelo'] == modelo]
        for rango in rangos_bin_limited:
            df_r = df_modelo[df_modelo['Rango'] == rango]
            if len(df_r) > 0 and not pd.isna(df_r['SMAPE'].iloc[0]):
                row.append(df_r['SMAPE'].iloc[0])
            else:
                row.append(np.nan)
        heatmap_data.append(row)
    
    heatmap_array = np.array(heatmap_data)
    # Use reversed colormap (lower SMAPE = better = green), cap at 100%
    im = ax.imshow(heatmap_array, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=100)
    ax.set_xticks(range(len(rangos_bin_limited)))
    ax.set_xticklabels(rangos_bin_limited, rotation=45, ha='right', fontsize=13)
    ax.set_yticks(range(len(modelos_base)))
    ax.set_yticklabels(modelos_base, fontsize=13)
    ax.set_xlabel(x_label_bin, fontsize=16)
    ax.set_ylabel('Model', fontsize=16)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('SMAPE (%)', fontsize=14)
    for i in range(len(modelos_base)):
        for j in range(len(rangos_bin_limited)):
            val = heatmap_array[i, j]
            if not np.isnan(val):
                color = 'white' if val > 60 or val < 40 else 'black'
                ax.text(j, i, f'{val:.0f}', ha='center', va='center', color=color, fontsize=10)
    plt.tight_layout()
    filename = f'{target_name}_heatmap_SMAPE_base_models_separate_bins.pdf'
    plt.savefig(os.path.join(base_dir, filename), format='pdf', bbox_inches='tight')
    plt.close()
    print(f"      ✅ Heatmap SMAPE PDF for base models (separate bins - limited)")
    
    # Heatmaps SMAPE for stacking models - separate bins
    for stack_num in range(1, 6):
        stack_modelos = [m for m in modelos_stack if m.startswith(f'Stack{stack_num}_')]
        if not stack_modelos:
            continue
        
        fig, ax = plt.subplots(figsize=(14, 6))
        heatmap_data = []
        for modelo in stack_modelos:
            row = []
            df_modelo = df_bin[df_bin['Modelo'] == modelo]
            for rango in rangos_bin_limited:
                df_r = df_modelo[df_modelo['Rango'] == rango]
                if len(df_r) > 0 and not pd.isna(df_r['SMAPE'].iloc[0]):
                    row.append(df_r['SMAPE'].iloc[0])
                else:
                    row.append(np.nan)
            heatmap_data.append(row)
        
        heatmap_array = np.array(heatmap_data)
        im = ax.imshow(heatmap_array, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=100)
        ax.set_xticks(range(len(rangos_bin_limited)))
        ax.set_xticklabels(rangos_bin_limited, rotation=45, ha='right', fontsize=13)
        ax.set_yticks(range(len(stack_modelos)))
        ax.set_yticklabels([m.replace(f'Stack{stack_num}_', '') for m in stack_modelos], fontsize=13)
        ax.set_xlabel(x_label_bin, fontsize=16)
        ax.set_ylabel('Meta-model', fontsize=16)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('SMAPE (%)', fontsize=14)
        for i in range(len(stack_modelos)):
            for j in range(len(rangos_bin_limited)):
                val = heatmap_array[i, j]
                if not np.isnan(val):
                    color = 'white' if val > 60 or val < 40 else 'black'
                    ax.text(j, i, f'{val:.0f}', ha='center', va='center', color=color, fontsize=10)
        plt.tight_layout()
        filename = f'{target_name}_heatmap_SMAPE_Stack{stack_num}_separate_bins.pdf'
        plt.savefig(os.path.join(stack_dir, f'Stack{stack_num}', filename), format='pdf', bbox_inches='tight')
        plt.close()
    
    print(f"      ✅ Heatmaps SMAPE PDF for stacking models (separate bins - limited)")
    
    # =====================================================
    # PERCENTILE ANALYSIS CHART
    # =====================================================
    print(f"   📊 Generating Combined SMAPE+RMSE Chart (Percentiles)...")
    
    pct_dir = os.path.join(target_graficos_dir, 'percentiles')
    os.makedirs(pct_dir, exist_ok=True)
    
    PCT_LABELS_CSV = os.path.join(data_dir, 'percentile_labels.csv')
    if df_original is not None:
        # 1. Recalculate percentiles cuts to know real values (same logic as stats script)
        df_work_p = df_original[(df_original[target_name].notna()) & (df_original[target_name] != 0)].copy()
        df_work_p = df_work_p[(df_work_p['c'].isna()) | (df_work_p['c'] < df_work_p[target_name])]
        df_work_p = df_work_p[~(df_work_p['fccarb'].notna() & (df_work_p['fccarb'] < 20))]
    
        y_ref = df_work_p[target_name].dropna().values
        qs = np.linspace(0, 1, 11)
        cuts = np.quantile(y_ref, qs)
        cuts[0] = -np.inf
        cuts[-1] = np.inf
    
        bins_pct = [(cuts[i], cuts[i+1]) for i in range(len(cuts)-1)]
    
        # 2. Get training distribution on these bins
        d_tot, d_ifn2, d_ifn3, n_tr = calcular_distribucion_training(target_name, bins_pct)
    
        # 3. Create labels
        x_labels_pct = []
        rangos_pct_csv = [] 
        for i, q in enumerate(qs[:-1]):
            label_csv = f'P{int(q*100):02d}-P{int((q+0.1)*100):02d}' # Matches CSV format 'P00-P10'
            rangos_pct_csv.append(label_csv)
        
            vmin = cuts[i] if cuts[i] != -np.inf else 0
            vmax = cuts[i+1] if cuts[i+1] != np.inf else 999
        
            # Format real value range
            if i == 0: vmin = y_ref.min()
            if i == len(qs) - 2: vmax = y_ref.max() # Last bin
        
            # Format real value range
            def fmt_val(v):
                if abs(v) < 0.1: return f"{v:.4f}"
                if abs(v) < 10: return f"{v:.2f}"
                return f"{v:.1f}"

            fmt_min = fmt_val(vmin)
            fmt_max = fmt_val(vmax)
        
            if cuts[i+1] == np.inf:
                lbl = f"{label_csv}\n(>{fmt_min})"
            else:
                lbl = f"{label_csv}\n({fmt_min}-{fmt_max})"
            x_labels_pct.append(lbl)
        
        _lbl = pd.DataFrame({'target': target_name, 'rango': rangos_pct_csv, 'label': x_labels_pct})
        _old = pd.read_csv(PCT_LABELS_CSV) if os.path.exists(PCT_LABELS_CSV) else None
        _old = _old[_old['target'] != target_name] if _old is not None else None
        pd.concat([_old, _lbl]).to_csv(PCT_LABELS_CSV, index=False)
    else:
        _lbl = pd.read_csv(PCT_LABELS_CSV)
        _lbl = _lbl[_lbl['target'] == target_name]
        rangos_pct_csv = list(_lbl['rango'])
        x_labels_pct = list(_lbl['label'])
        
    x_pos_pct = list(range(len(x_labels_pct)))
    
    # 4. Filter results for percentiles
    df_pct_res = results_df[(results_df['Target'] == target_name) & (results_df['TipoRango'] == 'Percentil')]
    
    # Filter base models 
    models_to_plot = [m for m in modelos_base if m not in ['SVR', 'BayesianNN']]
    
    # 5. Plot Combined Chart
    
    if len(df_pct_res) > 0:
        # Single plot, compact size for publication
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Lines SMAPE/RMSE
        ax2 = ax.twinx()
        
        # Plot SMAPE (Left Axis)
        for i, modelo in enumerate(models_to_plot):
            df_modelo = df_pct_res[df_pct_res['Modelo'] == modelo]
            valores_smape = []
            for rango in rangos_pct_csv:
                df_r = df_modelo[df_modelo['Rango'] == rango]
                if len(df_r) > 0 and not pd.isna(df_r['SMAPE'].iloc[0]):
                    valores_smape.append(df_r['SMAPE'].iloc[0])
                else:
                    valores_smape.append(np.nan)
            ax.plot(x_pos_pct, valores_smape, marker=markers[i % len(markers)], linestyle='-',
                    label=f'{modelo}', color=colores_base[i % len(colores_base)], linewidth=2, markersize=8, alpha=0.85)

        # Plot RMSE (Right Axis)
        for i, modelo in enumerate(models_to_plot):
            df_modelo = df_pct_res[df_pct_res['Modelo'] == modelo]
            valores_rmse = []
            for rango in rangos_pct_csv:
                df_r = df_modelo[df_modelo['Rango'] == rango]
                if len(df_r) > 0 and not pd.isna(df_r['RMSE'].iloc[0]):
                    valores_rmse.append(df_r['RMSE'].iloc[0])
                else:
                    valores_rmse.append(np.nan)
            ax2.plot(x_pos_pct, valores_rmse, marker=markers[i % len(markers)], linestyle='--',
                     color=colores_base[i % len(colores_base)], linewidth=2, markersize=8, alpha=0.5)

        # Labels & Style
        unit = 'tC/ha' if target_name == 'c4' else 'tC'
        ax.set_xlabel(f'Percentile Range (Real values of {target_name} in {unit})', fontsize=12, fontweight='bold')
        ax.set_ylabel('SMAPE (%) — solid lines', fontsize=12, color='black')
        ax2.set_ylabel(f'RMSE ({unit}) — dashed lines', fontsize=12, color='gray')
        
        ax.set_xticks(x_pos_pct)
        ax.set_xticklabels(x_labels_pct, rotation=45, ha='right', fontsize=10)
        
        # Legend INSIDE plot - Adjusted position
        # Move slightly to the right to avoid high SMAPE values on the left
        if target_name == 'carbono_bruto4':
             ax.legend(bbox_to_anchor=(0.60, 0.98), loc='upper center', fontsize=9, ncol=3, frameon=True, facecolor='white', framealpha=0.9)
        else:
             ax.legend(loc='upper center', fontsize=9, ncol=3, frameon=True, facecolor='white', framealpha=0.9)
             
        ax.grid(True, alpha=0.3)
        ax2.tick_params(axis='y', labelcolor='gray')
        
        plt.tight_layout()
        
        filename = f'{target_name}_base_models_SMAPE_RMSE_percentiles.pdf'
        plt.savefig(os.path.join(pct_dir, filename), format='pdf', bbox_inches='tight')
        plt.close()
        
        print(f"      ✅ Generated {filename}")
    else:
        print("      ⚠️ No percentile data found in CSV")

    # =====================================================
    # PERCENTILE ANALYSIS CHART (RMSE vs %RMSE)
    # =====================================================
    print(f"   📊 Generating Combined RMSE + %RMSE Chart (Percentiles)...")
    
    if len(df_pct_res) > 0:
        # Calculate %RMSE safely
        df_pct_calc = df_pct_res.copy()
        df_pct_calc['PCT_RMSE'] = df_pct_calc.apply(
            lambda row: (row['RMSE'] / row['MAE'] * row['wMAPE']) if (row['MAE'] > 0 and not pd.isna(row['wMAPE'])) else np.nan, 
            axis=1
        )
        
        # Single plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax2 = ax.twinx()
        
        # Plot %RMSE (Left Axis, Solid)
        for i, modelo in enumerate(models_to_plot):
            df_modelo = df_pct_calc[df_pct_calc['Modelo'] == modelo]
            valores_prmse = []
            for rango in rangos_pct_csv:
                df_r = df_modelo[df_modelo['Rango'] == rango]
                val = df_r['PCT_RMSE'].iloc[0] if len(df_r) > 0 else np.nan
                valores_prmse.append(val)
            ax.plot(x_pos_pct, valores_prmse, marker=markers[i % len(markers)], linestyle='-',
                    label=f'{modelo}', color=colores_base[i % len(colores_base)], linewidth=2, markersize=8, alpha=0.85)

        # Plot RMSE (Right Axis, Dashed)
        for i, modelo in enumerate(models_to_plot):
            df_modelo = df_pct_calc[df_pct_calc['Modelo'] == modelo]
            valores_rmse = []
            for rango in rangos_pct_csv:
                df_r = df_modelo[df_modelo['Rango'] == rango]
                val = df_r['RMSE'].iloc[0] if len(df_r) > 0 else np.nan
                valores_rmse.append(val)
            ax2.plot(x_pos_pct, valores_rmse, marker=markers[i % len(markers)], linestyle='--',
                     color=colores_base[i % len(colores_base)], linewidth=2, markersize=8, alpha=0.5)

        # Labels & Style
        unit = 'tC/ha' if target_name == 'c4' else 'tC'
        ax.set_xlabel(f'Percentile Range (Real values of {target_name} in {unit})', fontsize=12, fontweight='bold')
        ax.set_ylabel('%RMSE — solid lines', fontsize=12, color='black')
        ax2.set_ylabel(f'RMSE ({unit}) — dashed lines', fontsize=12, color='gray')
        
        ax.set_xticks(x_pos_pct)
        ax.set_xticklabels(x_labels_pct, rotation=45, ha='right', fontsize=10)
        
        # Legend (Reuse same positioning logic)
        if target_name == 'carbono_bruto4':
             ax.legend(bbox_to_anchor=(0.60, 0.98), loc='upper center', fontsize=9, ncol=3, frameon=True, facecolor='white', framealpha=0.9)
        else:
             ax.legend(loc='upper center', fontsize=9, ncol=3, frameon=True, facecolor='white', framealpha=0.9)
             
        ax.grid(True, alpha=0.3)
        ax2.tick_params(axis='y', labelcolor='gray')
        
        plt.tight_layout()
        
        filename = f'{target_name}_base_models_RMSE_PCT_RMSE_percentiles.pdf'
        plt.savefig(os.path.join(pct_dir, filename), format='pdf', bbox_inches='tight')
        plt.close()
        
        print(f"      ✅ Generated {filename}")

print("\n" + "=" * 100)
print("✅ PDF CHART GENERATION COMPLETED (PUBLICATION QUALITY)")
print(f"📊 Charts saved in: {graficos_dir}")
print("=" * 100)
