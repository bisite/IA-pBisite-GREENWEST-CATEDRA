# GreenWest: forest carbon stock estimation in Spain

Code, data partitions, test-set predictions and results of the article

> Gabriel-Vegas, J., Urbón-Jiménez, M. A., Armenteros-Cosme, P., Gil-González, A.-B., de Luis-Reboredo, A., Pérez-Lancho, B.
> *Data-driven estimation of forest carbon stocks in Spain: a machine learning approach using National Forest Inventory data.*
> Ecological Informatics (submitted).

The models predict the carbon stock recorded in the fourth Spanish National Forest Inventory (NFI4) from the conditions measured in an earlier inventory (NFI2 or NFI3), combining NFI dendrometric variables, ERA5-Land climate statistics and Landsat spectral indices.

| | |
|---|---|
| **Targets** | `c4`, carbon density (tC/ha), and `carbono_bruto4`, gross carbon per plot, species and diameter class (tC) |
| **Training sets** | NFI2 only, NFI3 only, and NFI2+NFI3 (global) |
| **Base models** | LightGBM, XGBoost, CatBoost, GBDT, Random Forest, Bagged DT, SVR, MLP and Bayesian NN, plus AdaBoost and KNN for the NFI2-only and NFI3-only sets |
| **Stacking** | 5 base-model configurations × 6 meta-models (linear regression, ridge, gradient boosting, random forest, SVR, MLP) |
| **Evaluation** | Hold-out test set of 20% of the plots (grouped by plot, `random_state=42`); hyper-parameters selected with `GridSearchCV` and `GroupKFold(5)` on the training set |

## Workflow

```
NFI2/NFI3/NFI4 + ERA5-Land + Landsat
        │  catedra/pipeline/            (1 → 5)
        ▼
training table  ──►  catedra/training/scripts/   base models → stacking
                                │
                                ▼
                     scripts/tables/              test-set predictions and metrics
                                │
                                ▼
                     scripts/revision/, notebooks/   figures, SHAP and further analyses
```

## Repository layout

| Path | Contents |
|---|---|
| `catedra/pipeline/` | Data pipeline in execution order: NFI extraction (1), feature engineering (2), database loading (3.0), carbon computation (3.1), Martonne aridity index (3.2), Earth Engine ERA5-Land and Landsat variables (3.3, download modules in `modules/gee_finder_and_downloader/`), export of the training table (4), grouped period and completion of plot-level variables (4.1) and exploratory analysis (5) |
| `catedra/training/scripts/` | Training of the base models (`entrenamiento_<inventory>_<target>.py`) and of the stacking ensembles (`stacking_<inventory>_<target>.py`), with the hyper-parameter grids |
| `scripts/export_test_indices.py`, `data/splits/` | Train/test partition of every inventory and target |
| `scripts/tables/`, `tables/supplementary/` | Test-set predictions of every model (`preds_*.parquet`) and the model comparison tables |
| `scripts/revision/` | Figures and additional analyses (see the table below) |
| `notebooks/` | Descriptive statistics, model evaluation and SHAP analyses |
| `data/revision/`, `data/estadisticas_por_rango/`, `models/metrics/` | Intermediate results behind the figures and tables |
| `data/species_catalog.csv` | NFI species codes, names and conifer/broadleaf type |
| `models/scenarios/` | Stack5-MLP meta-models used for the carbon scenarios |
| `figures/` | Figures |

## Where each result comes from

| Result | Code | Output |
|---|---|---|
| Size of the final training data, number of species, R² and RMSE% overall and for horizons of up to 6 years | `scripts/revision/dataset_summary.py` | `data/revision/dataset_summary.csv` |
| Period between inventories, target distributions, descriptive statistics, target histograms | `notebooks/EDA/estadisticos_descriptivos.ipynb` | `figures/EDA/` |
| Top-3 models; model comparison tables (Supplementary Tables S1–S12) | `scripts/tables/generar_tablas_y_predicciones.py`, `scripts/tables/regenerar_tablas_comparativas.py` | `tables/supplementary/` |
| Error by period and observed vs. predicted values (Fig. 5.1, Supplementary Fig. S3) | `scripts/revision/generar_figuras_publicacion.py --models all` | `figures/fig62_publicacion/` |
| Bias test (Piñeiro et al., 2008) | `scripts/revision/generar_figuras_publicacion.py` (test in `regression_scatter.py`) | `figures/fig62_publicacion/bias_test.csv` |
| Behaviour at the extremes: plots above 300 tC/ha, below 10 tC/ha, negative predictions | `scripts/revision/bias_extreme_values.py` | `data/revision/bias_extreme_values.csv` |
| Absolute error by inventory and target (global vs. local, base vs. stacking) | `scripts/revision/generar_figura_consolidada.py` | `figures/consolidado/` |
| Sample-size-controlled comparison (Supplementary Table S13) | `scripts/revision/subsampling_experiment.py`, `scripts/revision/add_baseline_to_comparison.py` | `models/metrics/subsampling/` |
| Carbon scenarios | `scripts/revision/generar_sensibilidad_escenarios.py` | `figures/sensibilidad_escenarios/` |
| SMAPE and RMSE by decile | `scripts/revision/generar_estadisticas_escalas.py`, `scripts/revision/generar_graficos_estadisticas.py` | `data/estadisticas_por_rango/`, `figures/estadisticas_por_rango/` |
| Mean SHAP rank of the variables; SHAP summary plots | `notebooks/evaluacion/exploracion/mejores_modelos_shap.ipynb` | `figures/mejores_modelos_shap/` |
| SHAP values of the period between inventories | `scripts/revision/shap_periodo_analysis.py` | `figures/shap_periodo/` |
| SHAP by species and robustness of the species ranking | `notebooks/evaluacion/exploracion/mejores_modelos_especies.ipynb`, `scripts/revision/species_ranking_robustness.py` | `data/revision/species_shap_LightGBM_global_c4_test.csv` |
| Contribution of stand density (models retrained without `npies`) | `scripts/revision/npies_sensitivity_experiment.py` | `models/metrics/npies_sensitivity/` |
| Correlation, VIF and Cramér's V of the selected predictors | `scripts/revision/correlacion_44_variables.py` | `data/revision/`, `figures/revision/` |

## Quick start

The test-set predictions are included, so the comparison tables and the analyses built on them run without the training data or the models:

```bash
pip install -r requirements-lock.txt
python scripts/tables/regenerar_tablas_comparativas.py      # Supplementary comparison tables
python scripts/revision/bias_extreme_values.py              # extreme-value analysis
python scripts/revision/species_ranking_robustness.py       # species ranking robustness
python scripts/revision/generar_graficos_estadisticas.py    # SMAPE/RMSE by decile figures
```

## Data and models

- **Integrated database** (NFI, climate and spectral variables, with schema and variable dictionary): GREDOS repository, University of Salamanca, doi:[10.71636/0fvh-ad55](https://doi.org/10.71636/0fvh-ad55).
- **Training table** (`data/processed/ifn_entrenamiento_year_agrupado_periodo_completado_2.csv`, 259,853 records): produced from that database by `catedra/pipeline/4.Extraer_BBDD.ipynb` followed by `catedra/pipeline/4.1.Periodo_agrupado_y_completado.py`.
- **Trained models**: produced by the scripts in `catedra/training/scripts/` (base models first, then stacking) in `models/artifacts/<IFN2|IFN3|IFN2-3>/<c4|carbono_bruto4>/<base|stackings>/`. They are not included because of their size (about 42 GB).
- **Train/test partition** (`data/splits/`) and **test-set predictions** of every model (`tables/supplementary/preds_*.parquet`): included.

The pipeline notebooks read the database connection from the `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD` and `DB_NAME` environment variables (`catedra/pipeline/db.py`), and the Earth Engine project from `GEE_PROJECT`.

## Environment

Python 3.10 with the package versions in `requirements-lock.txt`. Random Forest, KNN and SVR use RAPIDS cuML 24.02 on GPU (installed with conda, see the header of `requirements-lock.txt`); without cuML, the training scripts fall back to the scikit-learn estimators. The data pipeline uses the packages listed in `catedra/pipeline/requirements.txt`.

The `scripts/revision/run_*_slurm.sh` files are SLURM launchers for the analyses that need a GPU; submit them from the repository root.

## Citation and licence

If you use this code, please cite the article (see `CITATION.cff`). The code is released under the MIT licence (`LICENSE`); data derived from the GREDOS dataset are distributed under its CC BY-NC-SA 4.0 licence.
