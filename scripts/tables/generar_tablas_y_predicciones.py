# -*- coding: utf-8 -*-
"""
GENERAR TABLAS Y GUARDAR PREDICCIONES
=====================================
1. Genera tablas LaTeX (Base y Stacking).
2. Guarda TODAS las predicciones y targets en archivos .parquet para análisis futuro rápido.
"""

import joblib
import pandas as pd
import numpy as np
import os
import glob
import sys
import gc
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore")

# ==================== CONFIGURACIÓN ====================
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "models" / "artifacts"
CSV_FILE = str(REPO_ROOT / "data" / "processed" / "ifn_entrenamiento_year_agrupado_periodo_completado_2.csv")
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
# Tables (.tex), test-set predictions (.parquet) and the run log
OUT_DIR = str(REPO_ROOT / "tables" / "supplementary")
LOG_FILE = f"{OUT_DIR}/run_log_{TIMESTAMP}.txt"
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES_C4 = [
    'especie_id', 'tipo_especie', 'periodo_agrupado', 'radio', 'ocupa', 'npies_1', 'npies_2', 'npies_5',
    'npies_10', 'npies_15', 'npies_20', 'npies_25', 'npies_30', 'npies_35', 'npies_40', 'npies_45',
    'npies_50', 'npies_55', 'npies_60', 'npies_65', 'npies_70', 'estado_id', 'fccarb', 'disesp_id',
    'modcomb_id', 'nivel2_id', 'tratmasa_id', 'rocosidad_id', 'orientacion_cat', 'elevacion', 'pendiente_cat',
    'martonneidx_id', 'skt_mean_primavera', 'pr_sum_primavera', 'skt_std_verano', 'skt_std_primavera',
    'pr_sum_otoño', 'skt_mean_verano', 'pr_sum_verano', 'pr_sum_invierno', 'gndvi_mean_verano',
    'ndii_mean_primavera', 'gndvi_std_primavera', 'evi_mean_primavera'
]
FEATURES_CB = [
    'especie_id', 'tipo_especie', 'periodo_agrupado', 'radio', 'ocupa', 'npies_1', 'npies_2', 'npies_5',
    'npies_10', 'npies_15', 'npies_20', 'npies_25', 'npies_30', 'npies_35', 'npies_40', 'npies_45',
    'npies_50', 'npies_55', 'npies_60', 'npies_65', 'npies_70', 'estado_id', 'fccarb', 'disesp_id',
    'modcomb_id', 'nivel2_id', 'tratmasa_id', 'rocosidad_id', 'orientacion_cat', 'elevacion', 'pendiente_cat',
    'martonneidx_id', 'gndvi_mean_verano', 'ndii_mean_primavera', 'gndvi_std_primavera', 'evi_mean_primavera'
]

MAPA_PENDIENTE = {'muy suave (<5°)': 0, 'suave (5-10°)': 1, 'moderada (10-15°)': 2, 'fuerte (15-20°)': 3, 'muy fuerte (20-30°)': 4, 'escarpada (30-50°)': 5, 'extrema (>50°)': 6}
MAPA_ORIENTACION = {'N': 0, 'NE': 1, 'E': 2, 'SE': 3, 'S': 4, 'SO': 5, 'O': 6, 'NO': 7}

TARGET_CONFIGS = [
    {
        'target': 'c4',
        'desc': 'tC/ha',
        'features': FEATURES_C4,
        'path_global_base': str(ARTIFACTS_DIR / 'IFN2-3' / 'c4' / 'base'),
        'path_global_stacking': str(ARTIFACTS_DIR / 'IFN2-3' / 'c4' / 'stackings'),
        'path_ifn2_base': str(ARTIFACTS_DIR / 'IFN2' / 'c4' / 'base'),
        'path_ifn2_stacking': str(ARTIFACTS_DIR / 'IFN2' / 'c4' / 'stackings'),
        'path_ifn3_base': str(ARTIFACTS_DIR / 'IFN3' / 'c4' / 'base'),
        'path_ifn3_stacking': str(ARTIFACTS_DIR / 'IFN3' / 'c4' / 'stackings')
    },
    {
        'target': 'carbono_bruto4',
        'desc': 'tC',
        'features': FEATURES_CB,
        'path_global_base': str(ARTIFACTS_DIR / 'IFN2-3' / 'carbono_bruto4' / 'base'),
        'path_global_stacking': str(ARTIFACTS_DIR / 'IFN2-3' / 'carbono_bruto4' / 'stackings'),
        'path_ifn2_base': str(ARTIFACTS_DIR / 'IFN2' / 'carbono_bruto4' / 'base'),
        'path_ifn2_stacking': str(ARTIFACTS_DIR / 'IFN2' / 'carbono_bruto4' / 'stackings'),
        'path_ifn3_base': str(ARTIFACTS_DIR / 'IFN3' / 'carbono_bruto4' / 'base'),
        'path_ifn3_stacking': str(ARTIFACTS_DIR / 'IFN3' / 'carbono_bruto4' / 'stackings')
    }
]

# Clases Dummy
def cast_to_str(X): return pd.DataFrame(X).astype(str).values
def make_ohe(): 
    try: return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except: return OneHotEncoder(handle_unknown='ignore', sparse=False)

class PyTorchMLPDropout(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes=(100,), activation='relu', dropout_rate=0.2, use_batchnorm=True):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(input_dim, 1))
    def forward(self, x): return self.model(x)
PyTorchMLP = PyTorchMLPDropout

class MLPRegressorTorch(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        for k, v in kwargs.items(): setattr(self, k, v)
        self.model_ = None
    def predict(self, X):
        if hasattr(X, 'values'): X = X.values
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_.to(device).eval()
        with torch.no_grad():
            return self.model_(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy().flatten()

# Logger
class DualLogger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(LOG_FILE, "a", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()
sys.stdout = DualLogger()

# Métricas
def calc_wmape(y_true, y_pred):
    sum_abs_y = np.sum(np.abs(y_true))
    return 100.0 * np.sum(np.abs(y_true - y_pred)) / sum_abs_y if sum_abs_y != 0 else np.nan

def calc_smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    return 100.0 * np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) if np.any(mask) else 0.0

def calc_bias(y_true, y_pred): return np.mean(y_pred - y_true)

def calc_metrics(y_true, y_pred, model_name):
    return {
        'Modelo': model_name,
        'R2': r2_score(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'wMAPE': calc_wmape(y_true, y_pred),
        'SMAPE': calc_smape(y_true, y_pred),
        'Bias': calc_bias(y_true, y_pred)
    }

def preparar_datos(df_raw, target, features, inventario_filter=None):
    if inventario_filter == 'IFN2':
        df_base = df_raw[df_raw['ifn_id'].astype(str).str.contains('2')].copy()
    elif inventario_filter == 'IFN3':
        df_base = df_raw[df_raw['ifn_id'].astype(str).str.contains('3')].copy()
    else:
        df_base = df_raw.copy()
    
    df_base['pendiente_cat'] = df_base['pendiente_cat'].map(MAPA_PENDIENTE)
    df_base['orientacion_cat'] = df_base['orientacion_cat'].map(MAPA_ORIENTACION)
    df = df_base[(df_base[target].notna()) & (df_base[target] != 0) & ((df_base['c'].isna()) | (df_base['c'] < df_base[target]))].copy()
    df = df[~(df['fccarb'].notna() & (df['fccarb'] < 20))].copy()
    df['ifn_id_str'] = df['ifn_id'].astype(str)
    
    present_features = [f for f in features if f in df.columns]
    X = df[present_features].astype('float32')
    y = df[target].values
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_idx = next(gss.split(X, y, groups=df['parcela_id']))
    
    return {'X_test': X.iloc[test_idx], 'y_test': y[test_idx], 'df_test': df.iloc[test_idx], 'test_idx': test_idx}

STACK_CONFIGS = {
    1: ['LightGBM', 'Random_Forest'],
    2: ['LightGBM', 'XGBoost', 'GBDT'],
    3: ['CatBoost', 'Random_Forest', 'GBDT'],
    4: ['CatBoost', 'LightGBM', 'Random_Forest', 'GBDT'],
    5: ['CatBoost', 'LightGBM', 'XGBoost', 'Random_Forest', 'GBDT', 'BaggedDT'],
}

def evaluar_stack(path, sid, cache, X_test):
    needed = STACK_CONFIGS[sid]
    preds = [cache[m].predict(X_test) for m in needed if m in cache]
    if len(preds) != len(needed): return None, "Faltan bases"
    meta = joblib.load(path)
    return meta.predict(np.column_stack(preds)), None

def process_base(path, X, y, metrics, predictions_dict, suffix=""):
    files = glob.glob(f"{path}/*.joblib")
    files = [f for f in files if 'Stack' not in f and 'test' not in f and 'context' not in f]
    cache = {}
    print(f"   -> Cargando modelos base de {os.path.basename(os.path.normpath(path))}...")
    for p in files:
        try:
            name = os.path.basename(p).replace('.joblib','').replace('_PyTorch','')
            model = joblib.load(p)
            cache[name] = model
            dname = f"{name}{suffix}"
            if "random_forest" in dname.lower(): dname = dname.replace("Random_Forest","RandomForest").replace("Random Forest","RandomForest")
            
            y_pred = model.predict(X)
            metrics.append(calc_metrics(y, y_pred, dname))
            predictions_dict[dname] = y_pred # GUARDAR PREDICCIÓN
            
        except Exception as e: print(f"Err Base {os.path.basename(p)}: {e}")
    return cache

def process_stack(path, cache, X, y, metrics, predictions_dict, suffix=""):
    files = glob.glob(f"{path}/*.joblib")
    print(f"   -> Evaluando stackings de {os.path.basename(os.path.normpath(path))}...")
    for p in files:
        if 'Stack' not in p: continue
        try:
            fname = os.path.basename(p)
            parts = fname.split('_')
            if len(parts)<2: continue
            sid = int(parts[0].replace('Stack',''))
            mname = fname.replace(f"{parts[0]}_", "").replace(".joblib", "")
            dname = f"Stack{sid} {mname}{suffix}"
            if "random_forest" in dname.lower(): dname = dname.replace("Random_Forest","RandomForest")
            
            pred, err = evaluar_stack(p, sid, cache, X)
            if not err: 
                metrics.append(calc_metrics(y, pred, dname))
                predictions_dict[dname] = pred # GUARDAR PREDICCIÓN
            
            gc.collect()
        except Exception as e: print(f"Err Stack {os.path.basename(p)}: {e}")

# ==================== GENERACIÓN TABLAS ====================
def latex_std_table(results, caption, label):
    if not results: return f"% Sin resultados para {label}"
    df = pd.DataFrame(results).sort_values('R2', ascending=False)
    best_idx = df.index[0]
    
    rows = ["\\begin{table}[H]", "    \\centering", f"    \\caption{{{caption}}}", f"    \\label{{{label}}}", 
            "    \\begin{tabular}{lrrrrrr}", "        \\toprule", 
            "        Modelo & $R^2_{\\text{test}}$ & RMSE$_{\\text{test}}$ & MAE$_{\\text{test}}$ & wMAPE & SMAPE & Bias \\\\", "        \\midrule"]
    
    for idx, row in df.iterrows():
        mname = row['Modelo'].replace('_', ' ')
        vals = [row['R2'], row['RMSE'], row['MAE'], row['wMAPE'], row['SMAPE'], row['Bias']]
        if idx == best_idx:
            rows.append(f"        \\textbf{{{mname}}} & " + " & ".join([f"\\textbf{{{v:.3f}}}" for v in vals]) + " \\\\")
        else:
            rows.append(f"        {mname} & " + " & ".join([f"{v:.3f}" for v in vals]) + " \\\\")
            
    rows.append("        \\bottomrule")
    rows.append("    \\end{tabular}")
    rows.append("\\end{table}\n\n")
    return "\n".join(rows)

def latex_long_table(results, caption, label):
    if not results: return f"% Sin resultados para {label}"
    df = pd.DataFrame(results).sort_values('R2', ascending=False)
    best_idx = df.index[0]
    
    rows = ["\\begin{longtable}{lrrrrrr}", f"    \\caption{{{caption}}}", f"    \\label{{{label}}} \\\\", "    \\toprule",
            "    Modelo & $R^2_{\\text{test}}$ & RMSE$_{\\text{test}}$ & MAE$_{\\text{test}}$ & wMAPE & SMAPE & Bias \\\\",
            "    \\midrule", "    \\endfirsthead", "    \\toprule",
            "    Modelo & $R^2_{\\text{test}}$ & RMSE$_{\\text{test}}$ & MAE$_{\\text{test}}$ & wMAPE & SMAPE & Bias \\\\",
            "    \\midrule", "    \\endhead", "    \\midrule",
            "    \\multicolumn{7}{r}{{Continúa en la siguiente página...}} \\\\",
            "    \\endfoot", "    \\bottomrule", "    \\endlastfoot"]
    
    for idx, row in df.iterrows():
        mname = row['Modelo'].replace('_', ' ')
        vals = [row['R2'], row['RMSE'], row['MAE'], row['wMAPE'], row['SMAPE'], row['Bias']]
        if idx == best_idx:
            rows.append(f"    \\textbf{{{mname}}} & " + " & ".join([f"\\textbf{{{v:.3f}}}" for v in vals]) + " \\\\")
        else:
            rows.append(f"    {mname} & " + " & ".join([f"{v:.3f}" for v in vals]) + " \\\\")
            
    rows.append("\\end{longtable}\n\n")
    return "\n".join(rows)

def save_partial(content, filename):
    with open(filename, "w", encoding='utf-8') as f:
        f.write(content)
    print(f"   -> Tabla guardada: {filename}")

def save_predictions_parquet(predictions_dict, y_true, df_test, scenario_name, target):
    """Guarda un DF con: id_parcela, ifn, y_true, y_preds..."""
    print(f"   💾 Guardando predicciones para {scenario_name}...")
    df_preds = pd.DataFrame(predictions_dict)
    df_preds['y_true'] = y_true
    
    # Añadir identificadores útiles del DF original
    if 'parcela_id' in df_test.columns: df_preds['parcela_id'] = df_test['parcela_id'].values
    if 'ifn_id' in df_test.columns: df_preds['ifn_id'] = df_test['ifn_id'].values
    if 'periodo_agrupado' in df_test.columns: df_preds['periodo_agrupado'] = df_test['periodo_agrupado'].values
    
    fname = f"{OUT_DIR}/preds_{target}_{scenario_name}.parquet"
    df_preds.to_parquet(fname)
    print(f"      -> {fname}")

def main():
    print("🚀 INICIO (Métricas + Tablas + Predicciones)...")
    df_raw = pd.read_csv(CSV_FILE)
    
    for c in TARGET_CONFIGS:
        tgt = c['target']
        print(f"\nTarget: {tgt}...")
        
        # Splits
        dg = preparar_datos(df_raw, tgt, c['features'], None)
        d2 = preparar_datos(df_raw, tgt, c['features'], 'IFN2')
        d3 = preparar_datos(df_raw, tgt, c['features'], 'IFN3')
        
        # Global Subsets
        tg = dg['df_test']
        xg, yg = dg['X_test'], dg['y_test']
        m2 = tg['ifn_id_str'].str.contains('2').values
        m3 = tg['ifn_id_str'].str.contains('3').values
        xg2, yg2, tg2 = xg.iloc[m2], yg[m2], tg.iloc[m2]
        xg3, yg3, tg3 = xg.iloc[m3], yg[m3], tg.iloc[m3]
        
        gc.collect()

        # --- 1. IFN2 (Test Set IFN2 Only) ---
        print("  > Escenario 1: IFN2-only vs Global (subset IFN2)...")
        res_b2, res_s2 = [], []
        pred_dict_ifn2 = {} # Diccionario acumulativo para este test set!
        
        # Locales
        c2 = process_base(c['path_ifn2_base'], d2['X_test'], d2['y_test'], res_b2, pred_dict_ifn2)
        process_stack(c['path_ifn2_stacking'], c2, d2['X_test'], d2['y_test'], res_s2, pred_dict_ifn2)
        del c2; gc.collect()
        
        # Los modelos IFN2-only se evalúan en su propio test (d2['X_test']) y
        # los globales en el subconjunto IFN2 del test global (xg2): son
        # splits distintos, por lo que las predicciones se guardan por separado.
        
        save_predictions_parquet(pred_dict_ifn2, d2['y_test'], d2['df_test'], "IFN2_Locales", tgt)
        save_partial(latex_std_table(res_b2, f"Modelos Base IFN2-only ({tgt})", f"tab:base_ifn2_{tgt}"), f"{OUT_DIR}/tab_base_ifn2_{tgt}.tex")
        save_partial(latex_long_table(res_s2, f"Modelos Stacking IFN2-only ({tgt})", f"tab:stack_ifn2_{tgt}"), f"{OUT_DIR}/tab_stack_ifn2_{tgt}.tex")
        
        # --- 2. IFN3 (Test Set IFN3 Only) ---
        print("  > Escenario 2: IFN3-only...")
        res_b3, res_s3 = [], []
        pred_dict_ifn3 = {}
        c3 = process_base(c['path_ifn3_base'], d3['X_test'], d3['y_test'], res_b3, pred_dict_ifn3)
        process_stack(c['path_ifn3_stacking'], c3, d3['X_test'], d3['y_test'], res_s3, pred_dict_ifn3)
        del c3; gc.collect()
        
        save_predictions_parquet(pred_dict_ifn3, d3['y_test'], d3['df_test'], "IFN3_Locales", tgt)
        save_partial(latex_std_table(res_b3, f"Modelos Base IFN3-only ({tgt})", f"tab:base_ifn3_{tgt}"), f"{OUT_DIR}/tab_base_ifn3_{tgt}.tex")
        save_partial(latex_long_table(res_s3, f"Modelos Stacking IFN3-only ({tgt})", f"tab:stack_ifn3_{tgt}"), f"{OUT_DIR}/tab_stack_ifn3_{tgt}.tex")
        
        # --- 3. Global Full: evalúa el modelo global sobre todo el test global ---
        # (para comparar con IFN2-local se usaría el subconjunto xg2).
        print("  > Escenario 3: Global Completo...")
        res_global_full = [] # Para tabla global
        res_global_ifn2 = [] # Para añadir a tabla comparativa IFN2 (Opcional, manual en latex)
        pred_dict_global = {}
        
        cg = process_base(c['path_global_base'], xg, yg, res_global_full, pred_dict_global, "$_{\\text{global}}$")
        process_stack(c['path_global_stacking'], cg, xg, yg, res_global_full, pred_dict_global, "$_{\\text{global}}$") # base y stacking se acumulan juntos en res_global_full
        
        save_predictions_parquet(pred_dict_global, yg, tg, "Global_Full", tgt)
        del cg; gc.collect()
        
        # Tablas separadas para modelos base y stacking
        res_gb = [r for r in res_global_full if 'Stack' not in r['Modelo']]
        res_gs = [r for r in res_global_full if 'Stack' in r['Modelo']]
        
        save_partial(latex_std_table(res_gb, f"Modelos Base Globales ({tgt})", f"tab:base_global_{tgt}"), f"{OUT_DIR}/tab_base_global_{tgt}.tex")
        save_partial(latex_long_table(res_gs, f"Modelos Stacking Globales ({tgt})", f"tab:stack_global_{tgt}"), f"{OUT_DIR}/tab_stack_global_{tgt}.tex")

    print(f"\n✅ Hecho. Salida en: {OUT_DIR}")

if __name__ == "__main__":
    main()
