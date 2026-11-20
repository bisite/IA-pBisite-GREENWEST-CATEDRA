# -*- coding: utf-8 -*-
"""
REGENERAR TABLAS ORDENADAS DESDE PREDICCIONES GUARDADAS
=======================================================
Combina modelos Locales vs Globales en la misma tabla, ordenados por R2.
Usa los archivos .parquet generados previamente para no recalcular inferencia.
"""

import pandas as pd
import numpy as np
import os
import glob
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Configuración
from pathlib import Path
# Folder written by generar_tablas_y_predicciones.py (run that script first)
OUTPUT_DIR = str(Path(__file__).resolve().parents[2] / "tables" / "supplementary")
print(f"Working directory: {OUTPUT_DIR}")

TARGETS = ['c4', 'carbono_bruto4']

def calc_wmape(y_true, y_pred):
    sum_abs_y = np.sum(np.abs(y_true))
    return 100.0 * np.sum(np.abs(y_true - y_pred)) / sum_abs_y if sum_abs_y != 0 else np.nan

def calc_smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    return 100.0 * np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) if np.any(mask) else 0.0

def calc_bias(y_true, y_pred): return np.mean(y_pred - y_true)

def get_metrics_from_df(df, target_col='y_true', suffix=""):
    metrics = []
    # Columnas de modelos son todas excepto meta-data
    meta_cols = ['y_true', 'parcela_id', 'ifn_id', 'periodo_agrupado']
    models = [c for c in df.columns if c not in meta_cols]
    
    y_true = df[target_col].values
    
    for m in models:
        y_pred = df[m].values
        # Limpiar nombre
        clean_name = m.replace('_PyTorch', '')
        
        # Nueva limpieza: Evitar duplicar sufijos si ya venían en el parquet
        clean_name = clean_name.replace("$_{\\text{global}}$", "").replace("$ {\\text{global}}$", "").strip()
        
        if suffix:
            clean_name += suffix
        
        # Corregir formato RandomForest
        if "random_forest" in clean_name.lower(): 
            clean_name = clean_name.replace("Random_Forest", "RandomForest").replace("Random Forest", "RandomForest")
            
        ms = {
            'Modelo': clean_name,
            'R2': r2_score(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'wMAPE': calc_wmape(y_true, y_pred),
            'SMAPE': calc_smape(y_true, y_pred),
            'Bias': calc_bias(y_true, y_pred)
        }
        metrics.append(ms)
    return metrics

def generate_latex(results, caption, label, table_type='table'):
    if not results: return f"% Sin resultados {label}"
    df = pd.DataFrame(results).sort_values('R2', ascending=False)
    best_idx = df.index[0]
    
    # Formato LaTeX correcto para subíndice global
    # Reemplazar "text{global}" malformado si existiera, aunque lo metemos bien aqui
    
    rows = []
    if table_type == 'longtable':
        rows = ["\\begin{longtable}{lrrrrrr}", 
                f"    \\caption{{{caption}}}", 
                f"    \\label{{{label}}} \\\\", 
                "    \\toprule",
                "    Modelo & $R^2_{\\text{test}}$ & RMSE$_{\\text{test}}$ & MAE$_{\\text{test}}$ & wMAPE & SMAPE & Bias \\\\",
                "    \\midrule", "    \\endfirsthead", "    \\toprule",
                "    Modelo & $R^2_{\\text{test}}$ & RMSE$_{\\text{test}}$ & MAE$_{\\text{test}}$ & wMAPE & SMAPE & Bias \\\\",
                "    \\midrule", "    \\endhead", "    \\midrule",
                "    \\multicolumn{7}{r}{{Continúa en la siguiente página...}} \\\\",
                "    \\endfoot", "    \\bottomrule", "    \\endlastfoot"]
    else:
        rows = ["\\begin{table}[H]", "    \\centering", 
                f"    \\caption{{{caption}}}", 
                f"    \\label{{{label}}}", 
                "    \\begin{tabular}{lrrrrrr}", "        \\toprule", 
                "        Modelo & $R^2_{\\text{test}}$ & RMSE$_{\\text{test}}$ & MAE$_{\\text{test}}$ & wMAPE & SMAPE & Bias \\\\", 
                "        \\midrule"]

    for idx, row in df.iterrows():
        mname = row['Modelo'].replace('_', ' ').replace('$ {', '$_{')  # keep the LaTeX subscript
        # Arreglar formato subíndice si es necesario
        vals = [row['R2'], row['RMSE'], row['MAE'], row['wMAPE'], row['SMAPE'], row['Bias']]
        
        row_str = ""
        if idx == best_idx:
            row_str = f"    \\textbf{{{mname}}} & " + " & ".join([f"\\textbf{{{v:.3f}}}" for v in vals]) + " \\\\"
        else:
            row_str = f"    {mname} & " + " & ".join([f"{v:.3f}" for v in vals]) + " \\\\"
        
        # indentación longtable vs table
        if table_type == 'table': row_str = "    " + row_str
        rows.append(row_str)
            
    if table_type == 'longtable':
        rows.append("\\end{longtable}\n\n")
    else:
        rows.append("        \\bottomrule")
        rows.append("    \\end{tabular}")
        rows.append("\\end{table}\n\n")

    return "\n".join(rows)

def process_target(tgt):
    print(f"Procesando {tgt}...")
    
    # Cargar Parquets
    # 1. Locales
    f_ifn2 = f"{OUTPUT_DIR}/preds_{tgt}_IFN2_Locales.parquet"
    f_ifn3 = f"{OUTPUT_DIR}/preds_{tgt}_IFN3_Locales.parquet"
    # 2. Globales Full (para filtrar)
    f_global = f"{OUTPUT_DIR}/preds_{tgt}_Global_Full.parquet"
    
    if not os.path.exists(f_ifn2) or not os.path.exists(f_global):
        print(f"Faltan archivos para {tgt}")
        return

    df_ifn2 = pd.read_parquet(f_ifn2)
    df_ifn3 = pd.read_parquet(f_ifn3)
    df_global = pd.read_parquet(f_global)
    
    # --- ESCENARIO IFN2 ---
    # a) Métricas modelos locales
    m_ifn2_local = get_metrics_from_df(df_ifn2)
    
    # b) Métricas modelos globales (subset IFN2)
    # Filtrar df_global por ifn_id que contenga '2'
    # df_global tiene columna 'ifn_id' guardada.
    mask2 = df_global['ifn_id'].astype(str).str.contains('2')
    df_global_sub2 = df_global[mask2]
    m_ifn2_global = get_metrics_from_df(df_global_sub2, suffix="$_{\\text{global}}$")
    
    # Unir
    # Necesitamos separar Base de Stack para las tablas
    all_m2 = m_ifn2_local + m_ifn2_global
    base_m2 = [m for m in all_m2 if 'Stack' not in m['Modelo']]
    stack_m2 = [m for m in all_m2 if 'Stack' in m['Modelo']]
    
    # Generar LaTeX
    tex_base2 = generate_latex(base_m2, f"Comparativa Modelos Base en IFN2 ({tgt})", f"tab:base_comp_ifn2_{tgt}", 'table')
    tex_stack2 = generate_latex(stack_m2, f"Comparativa Modelos Stacking en IFN2 ({tgt})", f"tab:stack_comp_ifn2_{tgt}", 'longtable')
    
    # Guardar
    with open(f"{OUTPUT_DIR}/tab_base_comparativa_ifn2_{tgt}.tex", 'w') as f: f.write(tex_base2)
    with open(f"{OUTPUT_DIR}/tab_stack_comparativa_ifn2_{tgt}.tex", 'w') as f: f.write(tex_stack2)

    # --- ESCENARIO IFN3 ---
    # a) Métricas modelos locales
    m_ifn3_local = get_metrics_from_df(df_ifn3)
    
    # b) Métricas modelos globales (subset IFN3)
    mask3 = df_global['ifn_id'].astype(str).str.contains('3')
    df_global_sub3 = df_global[mask3]
    m_ifn3_global = get_metrics_from_df(df_global_sub3, suffix="$_{\\text{global}}$")
    
    # Unir
    all_m3 = m_ifn3_local + m_ifn3_global
    base_m3 = [m for m in all_m3 if 'Stack' not in m['Modelo']]
    stack_m3 = [m for m in all_m3 if 'Stack' in m['Modelo']]
    
    # Generar LaTeX
    tex_base3 = generate_latex(base_m3, f"Comparativa Modelos Base en IFN3 ({tgt})", f"tab:base_comp_ifn3_{tgt}", 'table')
    tex_stack3 = generate_latex(stack_m3, f"Comparativa Modelos Stacking en IFN3 ({tgt})", f"tab:stack_comp_ifn3_{tgt}", 'longtable')
    
    # Guardar
    with open(f"{OUTPUT_DIR}/tab_base_comparativa_ifn3_{tgt}.tex", 'w') as f: f.write(tex_base3)
    with open(f"{OUTPUT_DIR}/tab_stack_comparativa_ifn3_{tgt}.tex", 'w') as f: f.write(tex_stack3)

def main():
    for t in TARGETS:
        process_target(t)
    print("✅ Tablas comparativas generadas y ordenadas.")

if __name__ == "__main__":
    main()
