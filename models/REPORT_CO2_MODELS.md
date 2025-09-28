# CO₂–GDP Modeling and Scenario Analysis

**Task.** Develop a predictive model to forecast CO₂ emissions from a comprehensive indicator set, and answer:  
**“If a country increases its GDP by 10%, what is the expected % change in CO₂ emissions, holding other factors constant?”**

This repository implements a dual pipeline:
- **Panel Fixed Effects (FE):** estimates an elasticity β from a log–log model. The % change in CO₂ for a k% GDP change is approximately **β × k%**.
- **XGBoost (XGB):** learns a flexible non-linear mapping in log space and answers the question via **counterfactual simulation**: base vs. +10% shock to ln(GDP).

## Data & Features (log-spec)
- Target: `ln_CO2 = ln(EN.GHG.CO2.MT.CE.AR5)`
- Main driver: `ln_GDP = ln(NY.GDP.MKTP.CD)`
- Example controls (log-transformed): population, urbanization, land-use CO₂, renewables share, etc.
- Train/test split via `--split_year` for out-of-sample evaluation.

## Modeling Choices

**Why FE + XGB (short version).** We combine a **Fixed Effects (FE)** panel model with **XGBoost (XGB)** to balance **economic interpretability** and **predictive flexibility**. FE delivers a clean, policy‑friendly **elasticity of CO₂ with respect to GDP** (log–log), while XGB captures **non‑linearities and interactions** across many indicators and enables **country/year‑specific counterfactuals** for the +10% GDP scenario.

### Why Fixed Effects (FE)?
- **Panel structure fit.** Country–year data naturally calls for FE, which controls **time‑invariant country heterogeneity** (institutions, geography) and can include **time effects** for global shocks.
- **Interpretable elasticity.** In a log–log specification, the coefficient on `ln_GDP` is the **CO₂–GDP elasticity (β)**. A +10% GDP shock implies **%ΔCO₂ ≈ β × 10%** (ceteris paribus).
- **Baseline clarity.** FE is transparent and robust relative to pooled OLS; it avoids stronger exogeneity assumptions required by random effects.

_Alternatives considered:_ Random Effects/GLS (efficiency hinges on strict exogeneity); GAM/regularized GLM (more flexible but less direct for an elasticity); dynamic panel (Arellano–Bond) adds inertia but complicates communication.

### Why XGBoost?
- **Non‑linearity & interactions.** XGB captures thresholds, saturation, and interactions between GDP, energy mix, urbanization, etc., that a linear FE model cannot.
- **Predictive performance.** Strong tabular baseline with early stopping and good out‑of‑sample behavior.
- **Scenario engine.** In log‑space we perturb `ln_GDP` by `ln(1.10)` and compare **CO₂_base vs CO₂_shock** per country and year; ideal for the policy question.

_Alternatives considered:_ Random Forest (robust yet typically behind XGB here); CatBoost (great for high‑cardinality categoricals; not central in this feature set); neural nets/LSTM (higher cost, lower explainability for policy use).

### Why both, not just one?
- **Conceptual and empirical cross‑check.** FE offers a causal‑style, easily interpretable elasticity; XGB validates and reveals **heterogeneous responses**. Agreement in direction strengthens confidence.
- **Communication + performance.** FE for a clear narrative and elasticity; XGB for **granular rankings by country** under the +10% GDP shock.

### Extensions (optional roadmap)
- FE with **Driscoll–Kraay** errors (dependence across countries).
- **Dynamic panel** for persistence in emissions.
- **Monotonic constraints** in XGB (if we want structural priors).
- **SHAP/PDP** for global and local explanations of non‑linear drivers.

## How to Reproduce
```bash
# FE with baseline predictions and metrics
python co2_models.py --input_file <wide_clean.csv> --output_dir <out>   --model_type fe --split_year 2015 --predict

# XGB with +10% GDP scenario and metrics
python co2_models.py --input_file <wide_clean.csv> --output_dir <out>   --model_type xgb --split_year 2015 --shock_pct 10
```

## Model Performance
**Fixed Effects (FE)**

|   rmse_train |   mae_train |   r2_train |   n_obs_train |   rmse_test |   mae_test |   r2_test |   n_obs_test |
|-------------:|------------:|-----------:|--------------:|------------:|-----------:|----------:|-------------:|
|      17.6069 |     17.5221 |     -27.77 |          6065 |     17.7264 |    17.6343 |  -27.7116 |          392 |
**XGBoost (XGB)**

|   rmse_train |   mae_train |   r2_train |   n_obs_train |   rmse_test |   mae_test |   r2_test |   n_obs_test |
|-------------:|------------:|-----------:|--------------:|------------:|-----------:|----------:|-------------:|
|     0.302969 |    0.223147 |   0.991481 |          6065 |    0.778787 |   0.483978 |  0.944582 |          392 |
## Scenario Analysis (+10% GDP)
**Scenario: +10% GDP shock (XGB)**

- Year evaluated: **2022**
- Countries covered: **192**
- Mean expected % change in CO₂: **12.84%**
- Median expected % change in CO₂: **3.35%**
**Top 10 countries by expected % increase (latest year)**

| iso3c   | Country               |   exp_change_% |
|:--------|:----------------------|---------------:|
| GUM     | Guam                  |       692.514  |
| CYM     | Cayman Islands        |       291.642  |
| MHL     | Marshall Islands      |       182.944  |
| CPV     | Cabo Verde            |       109.349  |
| GMB     | Gambia, The           |        52.0373 |
| AGO     | Angola                |        49.5292 |
| TZA     | Tanzania              |        41.2727 |
| PAK     | Pakistan              |        40.8832 |
| BRB     | Barbados              |        39.6644 |
| FSM     | Micronesia, Fed. Sts. |        38.2558 |
**Bottom 10 countries (lowest or negative change)**

| iso3c   | Country                  |   exp_change_% |
|:--------|:-------------------------|---------------:|
| GRL     | Greenland                |      -23.9925  |
| MNP     | Northern Mariana Islands |      -20.7084  |
| SGP     | Singapore                |      -18.3155  |
| ISR     | Israel                   |      -15.0153  |
| IRL     | Ireland                  |      -14.7197  |
| NOR     | Norway                   |      -12.381   |
| VUT     | Vanuatu                  |       -8.36096 |
| UGA     | Uganda                   |       -6.13    |
| CZE     | Czechia                  |       -4.47417 |
| GTM     | Guatemala                |       -4.45119 |
**Average expected % change by year (XGB scenario)**

|   year |   avg_exp_change_% |
|-------:|-------------------:|
|   1990 |            4.91897 |
|   1991 |            2.89663 |
|   1992 |            4.00119 |
|   1993 |            4.14999 |
|   1994 |            3.64198 |
|   1995 |            5.85331 |
|   1996 |            5.25217 |
|   1997 |            3.93391 |
|   1998 |            4.30212 |
|   1999 |            4.74127 |
|   2000 |            2.73006 |
|   2001 |            5.2964  |
|   2002 |            9.93714 |
|   2003 |           16.5753  |
|   2004 |            8.43637 |
|   2005 |            4.4971  |
|   2006 |            5.52178 |
|   2007 |            7.13225 |
|   2008 |            6.40897 |
|   2009 |            5.65293 |
|   2010 |            5.70467 |
|   2011 |            6.61212 |
|   2012 |            6.33371 |
|   2013 |            5.8242  |
|   2014 |            6.33663 |
|   2015 |            7.15669 |
|   2016 |           11.2492  |
|   2017 |           14.3336  |
|   2018 |           28.0496  |
|   2019 |           31.0505  |
|   2020 |           27.0986  |
|   2021 |           11.6572  |
|   2022 |           12.8397  |
## Baseline Predictions
- FE baseline predictions available for 1990–2022 (`predictions_fe.csv`).
- XGB baseline predictions available for 1990–2022 (`predictions_xgb.csv`).


## Interpretation Guide
- **Use FE when you want an interpretable elasticity.** Multiply β by the GDP shock in % to get the expected % change in CO₂.
- **Use XGB when you want flexible, possibly non-linear responses** and country/year-specific effects. The `scenario_xgb_10pct.csv` gives a per-country answer.

## Files Produced (this run)
- `metrics_fe.csv`, `metrics_xgb.csv` – train/test metrics.
- `predictions_fe.csv`, `predictions_xgb.csv` – baseline predictions of CO₂.
- `scenario_xgb_10pct.csv` – base vs. scenario and `% expected_change_pct`.

> **Answer to the central question:**  
> Given a **+10% GDP** increase, consult:
> - **FE:** `%ΔCO₂ ≈ β × 10%` (β is the ln_GDP coefficient from the FE regression).
> - **XGB:** the column `expected_change_pct` in `scenario_xgb_10pct.csv` (country-by-country).

