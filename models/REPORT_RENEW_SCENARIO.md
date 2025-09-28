# Renewables Investment Scenario (5-year)

Goal
- Estimate 5-year CO₂ impact from increasing the electricity mix share of renewables or hydro by a planned amount (in percentage points), and suggest a per‑country technology priority (hydro vs. non‑hydro renewables).

Inputs
- `data/processed/wide_clean.csv`: panel with target and drivers.
- Required columns: `EN.GHG.CO2.MT.CE.AR5` (CO₂), `NY.GDP.MKTP.CD` (GDP), optional controls (population, urban share, energy use, electricity mix shares).

Method
- Baseline projection: extend each country’s series 5 years via CAGR from historical values.
- Model: fit XGBoost on log-transformed target/features from `models/co2_models.py` (`build_features`, `fit_xgb`).
- Scenario: linearly ramp the chosen electricity mix share (`EG.ELC.RNEW.ZS` or `EG.ELC.HYRO.ZS`) by `renew_pp_total` over 5 years; reduce fossil shares proportionally to keep the mix bounded and renormalized.
- Prediction: compare baseline vs. scenario mean CO₂ over 5 future years per country.
- Probability: map reduction percent to a probability via a logistic calibration centered at 0%.
- Prioritization: sensitivity with +5pp to hydro and +5pp to non‑hydro renewables; pick the tech yielding larger CO₂ reduction.

Key Parameters
- `--years` (int, default 5): horizon for projection.
- `--renew_pp_total` (float, default 10.0): total increase (pp) in the chosen mix share over the horizon.
- `--invest_target` (`EG.ELC.RNEW.ZS` | `EG.ELC.HYRO.ZS`): technology focus for the main scenario.
- `--controls` (list): optional controls; script auto‑filters based on coverage.

Usage
```
python models/renew_invest_scenario.py \
  --input_file data/processed/wide_clean.csv \
  --output_dir model_results/renew \
  --years 5 \
  --renew_pp_total 10.0 \
  --invest_target EG.ELC.RNEW.ZS \
  --controls SP.POP.TOTL SP.URB.TOTL.IN.ZS EG.FEC.RNEW.ZS \
            EG.ELC.COAL.ZS EG.ELC.NGAS.ZS EG.ELC.PETR.ZS \
            EG.ELC.NUCL.ZS EG.ELC.HYRO.ZS EG.ELC.RNEW.ZS
```

Outputs
- `model_results/renew/renew_5y_country_results.csv`: `iso3c`, `Country`, `region`, `grid_g_per_kwh`, `CO2_base_5y`, `CO2_scen_5y`, `reduction_pct_5y`, `prob_reduce_5y`.
- `model_results/renew/renew_5y_priorities.csv`: `iso3c`, `Country`, `CO2_base_5y`, deltas for +5pp hydro/non‑hydro, `priority_tech`.
- `model_results/renew/renew_5y_global_summary.csv`: high‑level summary and scenario parameters.

Assumptions & Notes
- Mix shares are sanitized: NaN→0, percent→fraction, clipped to [0,1], renormalized if sum>1.
- Fossil shares (coal, gas, oil) are reduced proportionally to accommodate the target increase.
- Baseline CAGR projection requires ≥2 valid years per country; otherwise country is skipped in projection phase.
- Probability is heuristic; adjust `k` in `calibrated_probability` for different steepness.

Limitations
- XGBoost feature effects depend on available historical coverage and controls; interpret scenario results directionally.
- Grid intensity is a coarse mix‑weighted average; upstream and non‑combustion effects are not captured.

Version
- Script: `models/renew_invest_scenario.py`
- Outputs folder: `model_results/renew/`
