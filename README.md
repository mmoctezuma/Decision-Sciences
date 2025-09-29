# CO₂ Emissions Forecasting and Classification

This repository contains a complete pipeline to download, process, and model World Bank data with the goal of predicting and classifying CO₂ emissions behavior and running scenario analyses.

Includes:
- ETL scripts for data download and preprocessing.
- Econometric (Fixed Effects) and ML (XGBoost) models.
- Binary classification of 10-year CO₂ changes.
- EV adoption scenario (EV50) analysis.
 - Renewables investment scenario (5-year) analysis.
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
│   ├── renew_invest_scenario.py          # Renewables investment 5-year scenario + prioritization
│   ├── positive_and_shap_tables.py       # Utilities: positive-class table, regional share, SHAP table
│   ├── REPORT_CO2_MODELS.md              # Extended model report (FE/XGB)
│   └── REPORT_CO2_CLASSIFIER.md          # Classifier methodology and insights
│   └── REPORT_RENEW_SCENARIO.md          # Renewables scenario methodology and usage
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

You can run the ETL script in different modes depending on your needs:

#### Case 1: Only generate a summary of the raw data  
```bash
python -m ETL.etl_preprocess_and_summary \
  --input data/raw_data/wb_co2_and_indicators.csv \
  --workdir data/processed \
  --stage summarize
```

#### Case 2: Preprocess data with winsorization and recommendations  
```bash
python -m ETL.etl_preprocess_and_summary \
  --input data/raw_data/wb_co2_and_indicators.csv \
  --workdir data/processed \
  --stage preprocess \
  --policy_csv data/processed/summary_first/recommendations.csv \
  --start 1960 --end 2022 \
  --winsor --winsor_limits 0.01 0.01
```

#### Case 3: Full pipeline with a focus year  
```bash
python -m ETL.etl_preprocess_and_summary \
  --input data/raw_data/wb_co2_and_indicators.csv \
  --workdir data/processed \
  --start 1990 --end 2023 --focus_year 2023 \
  --winsor --winsor_limits 0.01 0.01 \
  --stage all
```

### 3. Econometric & ML models

```bash
# Fixed Effects & XGBoost
python -m models.co2_models \
  --input_file data/processed/wide_clean.csv \
  --output_dir model_results \
  --model_type fe --split_year 2020 --predict

python -m models.co2_models \
  --input_file data/processed/wide_clean.csv \
  --output_dir model_results \
  --model_type xgb --split_year 2020 --predict
```

For the +10% GDP scenario

```bash
# Fixed Effects & XGBoost
python -m models.co2_models \
  --input_file data/processed/wide_clean.csv \
  --output_dir model_results \
  --model_type fe --split_year 2020 --shock_pct 10

python -m models.co2_models \
  --input_file data/processed/wide_clean.csv \
  --output_dir model_results \
  --model_type xgb --split_year 2020 --shock_pct 10
  
# Single country example (Mexico)
python -m models.co2_models \
  --input_file data/processed/wide_clean.csv \
  --output_dir model_results \
  --model_type fe --split_year 2020 \
  --country MEX --shock_pct 10

python -m models.co2_models \
  --input_file data/processed/wide_clean.csv \
  --output_dir model_results \
  --model_type xgb --split_year 2020 \
  --country MEX --shock_pct 10
```

Alternative FE pipeline (log–log), with additional artifacts and optional per‑country run:

```bash
# All countries, with time effects and +10% GDP elasticity table
python -m models.fe_log_log_model \
  --input_file data/processed/wide_clean.csv \
  --output_dir model_results/fe \
  --time_effects --shock_pct 10

# Single country example (Mexico), using last available year as base
python -m models.fe_log_log_model \
  --input_file data/processed/wide_clean.csv \
  --output_dir model_results/fe \
  --country MEX --time_effects --shock_pct 10
```

### 4. EV50 scenario (50% EV adoption)

Estimate CO₂ reductions if 50% of fleet shifts to EVs (uses electricity mix, fleet and adoption data).

```bash
python -m models.ev50_scenario \
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

### Tables and SHAP (optional)

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

### 6. Final classifier (Logit + XGB)

```bash
python -m models.co2_classifier \
  --input_file data/processed/panel_target.csv \
  --output_dir model_results/classifier \
  --features EN.GHG.CO2.MT.CE.AR5 NY.GDP.PCAP.CD SP.POP.TOTL EG.USE.PCAP.KG.OE EG.FEC.RNEW.ZS
```

### 7. Renewables investment (5-year scenario)

Simulate a planned increase in the electricity mix share for renewables or hydro over 5 years, compare against a baseline projection, and produce country-level prioritization.

```bash
python -m models.renew_invest_scenario \
  --input_file data/processed/wide_clean.csv \
  --output_dir model_results/renew \
  --years 5 \
  --renew_pp_total 10.0 \
  --invest_target EG.ELC.RNEW.ZS \
  --controls SP.POP.TOTL SP.URB.TOTL.IN.ZS EG.FEC.RNEW.ZS \
            EG.ELC.COAL.ZS EG.ELC.NGAS.ZS EG.ELC.PETR.ZS \
            EG.ELC.NUCL.ZS EG.ELC.HYRO.ZS EG.ELC.RNEW.ZS
```

Outputs in `model_results/renew/`:
- `renew_5y_country_results.csv` — Baseline vs scenario CO₂, reduction%, probability.
- `renew_5y_priorities.csv` — Country-level tech prioritization (hydro vs non-hydro renewables).
- `renew_5y_global_summary.csv` — Global summary and scenario parameters.

---

## Models & Results (generated artifacts)

The following artifacts are produced in `model_results/` by the ETL, modeling, classification, and scenario scripts. Paths are relative to the repository root.

Top-level outputs
- `model_results/metrics_fe.csv` — Fixed Effects model metrics.
- `model_results/predictions_fe.csv` — FE baseline predictions (when run with `--predict`).
- `model_results/metrics_xgb.csv` — XGBoost model metrics.
- `model_results/predictions_xgb.csv` — XGB baseline predictions (when run with `--predict`).
- `model_results/scenario_xgb_10pct.csv` — XGB +10% GDP shock scenario results.
- `model_results/compare_10years.csv` — Decade-to-decade comparison used for classification.

Alternative FE pipeline (`models/fe_log_log_model.py`)
- `model_results/fe/metrics.csv` — Metrics for the alternative FE log–log pipeline.
- `model_results/fe/predictions_full.csv` — Full FE predictions across the panel.
- `model_results/fe/elasticity_table.csv` — Elasticities (incl. +10% GDP table when requested).
- `model_results/fe/scenario_gdp_plus10pct.csv` — FE +10% GDP scenario results.
- `model_results/fe/model_summary.txt` — Text summary of the FE model fit.

Classifier artifacts (`models/co2_classifier.py`)
- `model_results/classifier/metrics_classifier.csv` — Metrics for Logit and XGB classifiers.
- `model_results/classifier/logit_model.pkl` — Trained Logistic Regression model.
- `model_results/classifier/xgb_classifier_model.json` — Trained XGBoost classifier.
- `model_results/classifier/scaler.pkl` — Feature scaler used by the Logit pipeline.

EV50 scenario (`models/ev50_scenario.py`)
- `model_results/ev50/ev50_results.csv` — Estimated CO₂ impact by country.
- `model_results/ev50/ev50_results_capped.csv` — Same as above with transport cap applied.
- `model_results/ev50/ev50_sensitivity_capped.csv` — Sensitivity analysis under cap.
- `model_results/ev50/ev50_top_abs.csv` — Top absolute emission reductions.
- `model_results/ev50/ev50_top_pct.csv` — Top percentage emission reductions.

Renewables investment scenario (`models/renew_invest_scenario.py`)
- `model_results/renew/renew_5y_global_summary.csv` — Global 5‑year summary results.
- `model_results/renew/renew_5y_country_results.csv` — Per‑country results for the 5‑year horizon.
- `model_results/renew/renew_5y_priorities.csv` — Suggested priority list under the scenario.

---

## Credits

Developed as a project on modeling and analyzing CO₂ emissions using World Bank data.
