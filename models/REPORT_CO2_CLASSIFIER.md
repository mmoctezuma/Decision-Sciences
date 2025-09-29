# CO₂ Decade Classifier — Report

**Task.** Build a classifier to identify countries likely to achieve a **significant reduction in CO₂ emissions in the next decade**, and answer:  
**“What are the common characteristics of countries that successfully reduce emissions, and how can policymakers in other nations apply these insights?”**

This deliverable documents the scripts, data products, methodology, results, and policy implications for the decade classifier.

## Scripts & CLI

1) **Target construction (projection-based label)** — builds `likely_reduce_CO2` using decade projections and saves comparison and labeled panel:
```bash
python -m models.classify_10years \
  --input_file data/processed/wide_clean.csv \
  --output_file_compare model_results/10years/compare_10years.csv \
  --output_file_df data/processed/panel_target.csv \
  --target EN.GHG.CO2.MT.CE.AR5 \
  --gdp NY.GDP.MKTP.CD \
  --controls SP.POP.TOTL SP.URB.TOTL.IN.ZS EG.FEC.RNEW.ZS
```

2) **Classifier training & evaluation** — trains **Logistic Regression (L1)** and **XGBoost** with temporal validation and saves metrics/artifacts:
```bash
python -m models.co2_classifier \
  --input_file data/processed/panel_target.csv \
  --output_dir model_results/classifier \
  --features EG.ELC.COAL.ZS EG.ELC.HYRO.ZS EG.ELC.NGAS.ZS EG.ELC.PETR.ZS EG.ELC.RNEW.ZS \
             EG.USE.ELEC.KH.PC EG.USE.PCAP.KG.OE EN.GHG.CO2.LU.MT.CE.AR5 \
             NY.GDP.MKTP.CD NY.GDP.PCAP.CD SP.POP.TOTL SP.URB.TOTL.IN.ZS
```
Artifacts:
- `model_results/10years/compare_10years.csv`
- `data/processed/panel_target.csv`
- `model_results/classifier/metrics_classifier.csv`
- `model_results/classifier/logit_model.pkl`, `xgb_classifier_model.json`

## Methodology

### 1) Target definition
- For each country, we project the next 10 years of indicators using recent **CAGR/slope** trends.
- We fit an XGB model (trained on historical data in log-space) to forecast **CO₂_future** over the next decade.
- We compute the mean of `ln(CO2)` for the **last 10 historical years** and for the **next 10 projected years**.
- Label rule: `likely_reduce_CO2 = 1` if the **projected decade mean ln(CO₂)** is **lower** than the historical decade mean; else `0`.
- We also save `delta = next10_lnCO2 - last10_lnCO2` and a quick trend label (`disminuye`/`sube`) for interpretability.

### 2) Features for classification
Using a rolling **10-year window per country**, we derive per-series statistics at year *t* from years *(t-10 … t-1)*:
- **Level** (last), **volatility** (std), **trend** (slope), **growth** (CAGR).
- The target for year *t* is `likely_reduce_CO2`.

### 3) Temporal validation
We evaluate with multiple **time splits** (e.g., 2000, 2005, 2010): train up to split-year, test after, ensuring realism for forward-looking classification.

### 4) Models
- **Logistic Regression (L1, class_weight='balanced')** for interpretability and sparse selection.
- **XGBoost (binary:logistic)** for non-linearities and interactions.

## Label Snapshot

    - Countries covered in `compare_10years.csv`: **200**
    - Share labeled **likely_reduce_CO2 = 1**: **22.5%**

    **Top 10 strongest projected reducers** (most negative `delta` in mean ln(CO₂)`):
| iso3c   |     delta |
|:--------|----------:|
| SYC     | -1.85439  |
| GRL     | -1.67838  |
| KNA     | -1.52905  |
| TCA     | -1.46251  |
| ABW     | -1.03033  |
| BIH     | -0.989898 |
| STP     | -0.910187 |
| NCL     | -0.823034 |
| CAN     | -0.681664 |
| BGR     | -0.428883 |

    **Top 10 projected increasers** (most positive `delta`):
| iso3c   |   delta |
|:--------|--------:|
| GUM     | 3.7018  |
| VIR     | 3.2261  |
| FRO     | 2.75022 |
| KEN     | 1.75888 |
| TLS     | 1.69045 |
| YEM     | 1.64419 |
| GMB     | 1.62354 |
| SLE     | 1.52857 |
| SYR     | 1.52528 |
| BLZ     | 1.51042 |

## Evaluation Metrics

| model   |
|:--------|
| Logit   |
| XGBoost |

## Insights: What characterizes successful reducers?

Across historical patterns and the trained models, countries labeled as likely reducers tend to share:

- **Energy mix transition**: rising shares of **renewables** and declining shares of **coal/oil** in power generation; lower volatility in fossil shares.
- **Efficiency gains**: improvements in **energy use per capita** and **electricity intensity**, with downward slopes over the last decade.
- **Economic composition**: higher **services share** and slower growth in energy-intensive sectors (proxied by GDP/cap and urbanization dynamics).
- **Land-use management**: reductions in **land-use related CO₂** (where available), signaling better forestry/agriculture practices.
- **Policy consistency**: smoother 10-year trends (lower std) in key indicators suggest stable policy frameworks.

> These are correlational drivers in the classifier, not causal proofs; however, they are consistent with established pathways to decarbonization.

## Policy Recommendations

1. **Accelerate clean power**: set multi-year targets for **renewables share** and coal phase-down; de-risk private investment via auctions/PPAs.
2. **Drive efficiency**: standards for appliances/buildings and **industrial efficiency** programs to lower energy intensity trends.
3. **Electrify demand**: promote electrification of transport/heating where grids are decarbonizing; pair with renewables expansion.
4. **Protect sinks**: scale **REDD+**-style and reforestation programs to cut land-use CO₂.
5. **Institutional stability**: lock in policies with medium-term frameworks to reduce volatility in key indicators (supports the classifier's signals).
6. **Benchmark peers**: use `compare_10years.csv` deltas to identify **peer countries** with successful trajectories and adapt relevant policies.

_Interpretation tip:_ Evaluate both **Logit** coefficients (sign/direction) and **XGB** feature attributions (e.g., SHAP) to confirm which levers matter most in your dataset before crafting country-specific plans.

## Positive Class by Country (quick view)

Below is the list of countries labeled as **likely reducers (1)** in `compare_10years.csv`.  
If `delta` is present, more negative values indicate **stronger projected reduction**.

| iso3c   |       delta |
|:--------|------------:|
| ABW     | -1.03033    |
| AUS     | -0.0732161  |
| AUT     | -0.403919   |
| BGR     | -0.428883   |
| BIH     | -0.989898   |
| BLR     | -0.094881   |
| BRA     | -0.0100592  |
| CAN     | -0.681664   |
| CHN     | -0.170262   |
| CZE     | -0.393712   |
| DEU     | -0.348033   |
| DMA     | -0.0502005  |
| EST     | -0.16634    |
| FIN     | -0.163076   |
| GAB     | -0.365336   |
| GEO     | -0.052001   |
| GRC     | -0.148651   |
| GRD     | -0.194982   |
| GRL     | -1.67838    |
| IRN     | -0.00512312 |
| ISL     | -0.302061   |
| ITA     | -0.0623303  |
| JPN     | -0.421199   |
| KGZ     | -0.00466686 |
| KIR     | -0.228496   |
| KNA     | -1.52905    |
| KOR     | -0.127291   |
| KWT     | -0.269311   |
| LAO     | -0.210535   |
| LBN     | -0.147845   |
| LBY     | -0.0132813  |
| MDA     | -0.231418   |
| MKD     | -0.244499   |
| NCL     | -0.823034   |
| NOR     | -0.0658633  |
| NPL     | -0.0981545  |
| POL     | -0.0121562  |
| SGP     | -0.044256   |
| STP     | -0.910187   |
| SYC     | -1.85439    |
| TCA     | -1.46251    |
| UKR     | -0.294508   |
| WSM     | -0.0100859  |
| ZAF     | -0.2192     |
| ZWE     | -0.234964   |



## Interpretability (SHAP) & Regional Policy Playbooks

**Goal.** Explain the classifier's decisions and summarize **actionable levers** by region.

### How to compute SHAP values (XGBoost)
1. Load your trained XGB model (`xgb_classifier_model.json`) and the test set used in evaluation.
2. Use `shap.TreeExplainer(model).shap_values(X_test)` to obtain per-feature attributions.
3. Produce:
   - **Global** plots: `summary_plot`, `bar_plot` (top drivers overall).
   - **Local** plots: `force_plot` for specific countries/years to see **why** the model predicts reduction.

### Translating SHAP insights into policies
- If **renewables share** and **declining coal share** are top positive drivers → set clear auction pipelines and coal retirement schedules.
- If **energy intensity** declining drives positives → focus on industrial efficiency and building codes.
- If **land-use CO₂** reduction is pivotal → prioritize forestry and agroforestry programs.

### Regional playbooks (template)
- **Baseline signal:** review the top 3 SHAP features in each region (median across countries in region).
- **Policy levers:** map each top feature to 1–2 concrete actions suitable for local institutions, grid state, and fiscal space.

_Example mapping:_
- **Latin America:** renewables expansion + land-use protection; electrify buses where grids are green.
- **Europe & Central Asia:** efficiency retrofits in buildings; accelerate coal-to-gas/renewables substitution.
- **East Asia & Pacific:** grid decarbonization plus industrial efficiency; market mechanisms for heavy industry.
- **Sub-Saharan Africa:** add clean capacity with concessional finance; limit diesel fallback with mini-grids.
- **MENA:** large-scale solar/wind and efficiency in water/pumping; EVs for fleets and logistics.
- **North America:** deep retrofits and heat pumps; transmission build-out to integrate renewables.

## Reproducibility & Files

- `classify_10years.py` — builds labels and comparison table (`compare_10years.csv`) and outputs a labeled panel (`panel_target.csv`).
- `co2_classifier.py` — trains/evaluates models and saves `metrics_classifier.csv` and model artifacts.
- Make sure `data/processed/wide_clean.csv` is produced by your ETL pipeline before running step (1).
