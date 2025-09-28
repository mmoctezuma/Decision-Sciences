# CO₂ Emissions Forecasting and Classification

This repository contains a complete pipeline to download, process, and model World Bank data with the goal of predicting and classifying CO₂ emissions behavior and running scenario analyses.

Includes:
- ETL scripts for data download and preprocessing.
- Econometric (Fixed Effects) and ML (XGBoost) models.
- Binary classification of 10-year CO₂ changes.
- EV adoption scenario (EV50) analysis.
- Organized folders with results, metrics, predictions, and reports.

---

## Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for environment and dependency management (Python 3.10+).

```bash
# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate   # Linux / Mac
.venv\Scripts\activate      # Windows

# Install dependencies from pyproject.toml + uv.lock
uv sync
```

Alternative (pip):
```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\\Scripts\\activate on Windows
pip install -e .
```

---

## Repository Structure

```
.
├── ETL/
│   ├── download_from_WDI.py              # Downloads World Bank indicators → data/raw_data
│   ├── etl_preprocess_and_summary.py     # Preprocesses and generates summaries → data/processed
│   └── summary_report.md                 # ETL notes and data overview
│
├── models/
│   ├── co2_models.py                     # XGBoost & Fixed Effects models, GDP shock scenarios
│   ├── classify_10years.py               # Predict CO₂ + compare past vs future decade
│   ├── co2_classifier.py                 # Classifier (Logit + XGB) on target binary column
│   ├── fe_log_log_model.py               # Alternative FE log–log pipeline (+ per-country option)
│   ├── ev50_scenario.py                  # EV50 scenario (50% EV adoption) impact analysis
│   ├── positive_and_shap_tables.py       # Utilities: positive-class table, regional share, SHAP table
│   ├── REPORT_CO2_MODELS.md              # Extended model report (FE/XGB)
│   └── REPORT_CO2_CLASSIFIER.md          # Classifier methodology and insights
│
├── data/
│   ├── raw_data/                         # Raw World Bank downloads
│   ├── processed/                        # Processed datasets ready for modeling
│   └── ev_adoption_v2.csv                # EV adoption & fleet input (for EV50 scenario)
│
├── model_results/
│   ├── metrics_fe.csv                    # FE model metrics
│   ├── metrics_xgb.csv                   # XGB model metrics
│   ├── predictions_fe.csv                # FE baseline predictions (if --predict)
│   ├── predictions_xgb.csv               # XGB baseline predictions (if --predict)
│   ├── scenario_xgb_10pct.csv            # XGB +10% GDP scenario (if run without --predict)
│   ├── compare_10years.csv               # Past vs future decade comparison
│   ├── fe/                               # Alternative FE pipeline outputs
│   │   ├── model_summary.txt
│   │   ├── metrics.csv
│   │   ├── predictions_full.csv
│   │   ├── elasticity_table.csv
│   │   └── scenario_gdp_plus10pct.csv
│   └── classifier/                       # Classifier results (metrics, models)
│
└── README.md
```

---

## Usage

### 1. Data download

```bash
python ETL/download_from_WDI.py \
  --output data/raw_data/wb_co2_and_indicators.csv \
  --start 1970 --end 2022
```

### 2. ETL

### Tables and SHAP summaries (optional)

Produce positive-class and regional share tables from the 10‑year comparison; optionally compute SHAP global importance (requires xgboost + shap):

```bash
# From decade comparison
python models/positive_and_shap_tables.py \
  --compare_csv model_results/compare_10years.csv \
  --outdir model_results/classifier \
  --positive_table --regional_share

# With SHAP global importance (requires test features)
python models/positive_and_shap_tables.py \
  --compare_csv model_results/compare_10years.csv \
  --outdir model_results/classifier \
  --add_shap \
  --xgb_model_path model_results/classifier/xgb_classifier_model.json \
  --x_test_csv path/to/X_test.csv
```

You can run the ETL script in different modes depending on your needs:

#### 🔹 Case 1: Only generate a summary of the raw data  
```bash
python ETL/etl_preprocess_and_summary.py \
  --input data/raw_data/wb_co2_and_indicators.csv \
  --workdir data/processed \
  --stage summarize
```

#### 🔹 Case 2: Preprocess data with winsorization and recommendations  
```bash
python ETL/etl_preprocess_and_summary.py \
  --input data/raw_data/wb_co2_and_indicators.csv \
  --workdir data/processed \
  --stage preprocess \
  --policy_csv data/processed/summary_first/recommendations.csv \
  --start 1960 --end 2022 \
  --winsor --winsor_limits 0.01 0.01
```

#### 🔹 Case 3: Full pipeline with a focus year  
```bash
python ETL/etl_preprocess_and_summary.py \
  --input data/raw_data/wb_co2_and_indicators.csv \
  --workdir data/processed \
  --start 1990 --end 2023 --focus_year 2023 \
  --winsor --winsor_limits 0.01 0.01 \
  --stage all
```

### 3. Econometric & ML models

```bash
# Fixed Effects & XGBoost
python models/co2_models.py \
  --input_file data/processed/wide_clean.csv \
  --output_dir model_results \
  --model_type fe --split_year 2020 --predict

python models/co2_models.py \
  --input_file data/processed/wide_clean.csv \
  --output_dir model_results \
  --model_type xgb --split_year 2020 --shock_pct 10
```

Alternative FE pipeline (log–log), with additional artifacts and optional per‑country run:

```bash
# All countries, with time effects and +10% GDP elasticity table
python models/fe_log_log_model.py \
  --input_file data/processed/wide_clean.csv \
  --output_dir model_results/fe \
  --time_effects --shock_pct 10

# Single country example (Mexico), using last available year as base
python models/fe_log_log_model.py \
  --input_file data/processed/wide_clean.csv \
  --output_dir model_results/fe \
  --country MEX --time_effects --shock_pct 10
```

### 4. EV50 scenario (50% EV adoption)

Estimate CO₂ reductions if 50% of fleet shifts to EVs (uses electricity mix, fleet and adoption data).

```bash
python models/ev50_scenario.py \
  --wide_csv data/processed/wide_clean.csv \
  --ev_csv data/ev_adoption_v2.csv \
  --outdir model_results/ev50 \
  --target_share 0.50 --ice_g_per_km 180 --ev_kwh_per_km 0.18 \
  --km_per_vehicle 12000 --transport_cap_pct 0.20
```

Outputs in `model_results/ev50/`:
- `ev50_results.csv`, `ev50_results_capped.csv`, `ev50_sensitivity_capped.csv`, `ev50_top_abs.csv`, `ev50_top_pct.csv`

### 5. 10-year classification

```bash
python -m models.classify_10years \
  --input_file data/processed/wide_clean.csv \
  --output_file_compare model_results/compare_10years.csv \
  --output_file_df data/processed/panel_target.csv \
  --target EN.GHG.CO2.MT.CE.AR5 \
  --gdp NY.GDP.MKTP.CD \
  --controls SP.POP.TOTL SP.URB.TOTL.IN.ZS EG.USE.PCAP.KG.OE
```

### 6. Final classifier (Logit + XGB)

```bash
python models/co2_classifier.py \
  --input_file data/processed/panel_target.csv \
  --output_dir model_results/classifier \
  --features EN.GHG.CO2.MT.CE.AR5 NY.GDP.PCAP.CD SP.POP.TOTL EG.USE.PCAP.KG.OE EG.FEC.RNEW.ZS
```

---

## Expected results

- Metrics: `model_results/metrics_fe.csv`, `model_results/metrics_xgb.csv` (and `model_results/fe/metrics.csv` in the alt FE pipeline).
- Predictions (if `--predict`): `model_results/predictions_fe.csv`, `model_results/predictions_xgb.csv` (or `model_results/fe/predictions_full.csv`).
- XGB scenario: `model_results/scenario_xgb_10pct.csv`.
- 10-year comparison: `model_results/compare_10years.csv`.
- Classifier artifacts: `model_results/classifier/metrics_classifier.csv`, `logit_model.pkl`, `xgb_classifier_model.json`.
- EV50 scenario outputs: `model_results/ev50/ev50_results*.csv`, `ev50_top_abs.csv`, `ev50_top_pct.csv`.

---

## Credits

Developed as a project on modeling and analyzing CO₂ emissions using World Bank data.
