# CO₂ Emissions Forecasting and Classification

This repository contains a complete pipeline to download, process, and model World Bank data with the goal of **predicting and classifying CO₂ emissions behavior** under different scenarios.

It includes:
- **ETL scripts** for data download and preprocessing.
- Econometric and machine learning models (XGBoost).
- Binary classification of 10-year CO₂ changes.
- Organized folders with results and metrics.

---

## Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for environment and dependency management.

```bash
# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate   # Linux / Mac
.venv\Scripts\activate      # Windows

# Install dependencies from pyproject.toml + uv.lock
uv sync
```

---

## Repository Structure

```
.
├── ETL/
│   ├── download_from_WDI.py              # Downloads World Bank indicators → data/raw_data
│   └── etl_preprocess_and_summary.py     # Preprocesses and generates summaries → data/processed
│
├── models/
│   ├── co2_models.py                     # XGBoost & Fixed Effects models, GDP shock scenarios
│   ├── classify_10years.py               # Predict CO₂ + compare past vs future decade
│   └── co2_classifier.py                 # Classifier (Logit + XGB) on target binary column
│
├── data/
│   ├── raw_data/                         # Raw World Bank downloads
│   └── processed/                        # Processed datasets ready for modeling
│
├── model_results/
│   ├── metricas_fe.json                  # FE model metrics
│   ├── metricas_xgb.json                 # XGB model metrics
│   ├── predictions_fe.csv                # FE model predictions (last decades)
│   ├── predictions_xgb.csv               # XGB model predictions (last decades)
│   ├── compare_10years.csv               # Past vs future decade comparison
│   ├── fe/                               # Additional FE results
│   └── classifier/                       # Classifier results (metrics, pickle models)
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

### 4. 10-year classification

```bash
python -m models.classify_10years \
  --input_file data/processed/wide_clean.csv \
  --output_file_compare model_results/compare_10years.csv \
  --output_file_df data/processed/panel_target.csv \
  --target EN.GHG.CO2.MT.CE.AR5 \
  --gdp NY.GDP.MKTP.CD \
  --controls SP.POP.TOTL SP.URB.TOTL.IN.ZS EG.USE.PCAP.KG.OE
```

### 5. Final classifier (Logit + XGB)

```bash
python models/co2_classifier.py \
  --input_file data/processed/panel_target.csv \
  --output_dir model_results/classifier \
  --features EN.GHG.CO2.MT.CE.AR5 NY.GDP.PCAP.CD SP.POP.TOTL EG.USE.PCAP.KG.OE EG.FEC.RNEW.ZS
```

---

## Expected results

- **Metrics (FE & XGB):** stored in `model_results/metricas_fe.json` and `metricas_xgb.json`.
- **Predictions:** stored in `predictions_fe.csv` and `predictions_xgb.csv`.
- **10-year comparison:** `compare_10years.csv`.
- **Classifiers:** available in `model_results/classifier/` with metrics and serialized models.

---

## Credits

Developed as part of a test project on modeling and analyzing CO₂ emissions using World Bank data.
