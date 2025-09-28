# World Bank CO₂ & Socio-Economic Indicators – Extended Summary Report

## 1. Dataset Overview
- **Source:** World Development Indicators (World Bank, 1990–2022).
- **Scope:** Around 200 countries, covering 30+ socio-economic, environmental, and energy indicators.
- **Structure:**
  - Identifiers: `iso3c`, `Country`, `year`.
  - Indicators: CO₂ emissions, GDP, population, energy use, urbanization, education, R&D, investment, etc.
- **Coverage:** Missing values are frequent, especially in early decades, as shown in `country_coverage_by_series.csv`.

## 2. Preprocessing Steps
- **Reshape:** Converted raw WDI format (`YR####` columns) into panel data (`year` column).
- **Interpolation (no extrapolation):**
  - Levels with growth trend (e.g., GDP, population, CO₂ total) → log-linear interpolation.
  - Per-capita indicators (e.g., GDP per capita, CO₂ per capita) → linear interpolation.
  - Rates 0–100 (e.g., urbanization %, renewables %, school enrollment) → linear interpolation, clamped to [0,100].
  - Other ratios/indices (e.g., energy intensity, R&D % GDP) → linear interpolation.
- **Winsorization:** Applied only to per-capita and share-rate indicators (1% tails).
- **Gaps:** Internal gaps per country/series were quantified (`country_series_gaps.csv`).
- **Series policy:** Each variable classified by recommended transformation and bounds (`recommendations.csv`).

## 3. Key Statistics
Selected indicators from `stats_preview.csv` and `summary_statistics.csv`:

| Indicator | Median | P01 | P99 | Missing % |
|-----------|--------|-----|-----|-----------|
| NY.GDP.MKTP.CD | 13899217679.92 | 77347018.63 | 4601224878604.01 | 5.1% |
| EN.GHG.CO2.MT.CE.AR5 | 8.26 | 0.00 | 2550.18 | 6.5% |
| EN.GHG.CO2.PC.CE.AR5 | 2.16 | 0.00 | 33.01 | 6.5% |
| SP.POP.TOTL | 5386675.50 | 15211.39 | 327050431.64 | 0.0% |
| SP.URB.TOTL.IN.ZS | 57.12 | 13.12 | 100.00 | 0.9% |

## 4. Correlations (Pearson, 1990–2022)
- **GDP vs. CO2 total**: 0.80
- **GDP per capita vs. CO2 per capita**: 0.31
- **Urbanization vs. energy per capita**: 0.56
- **Renewables share vs. CO2 per capita**: -0.35
- **Secondary education vs. GDP per capita**: 0.50

## 5. Notable Patterns & Anomalies
- **Top emitters 2020:** China, USA, India, Russia, Japan, Germany, Iran, South Korea, Canada, Saudi Arabia.
- **Reductions 2000–2020:** European countries (UK, Germany) show clear declines in CO₂ per capita.
- **Fast increases 2000–2020:** Emerging economies (China, Vietnam, Middle East) show sharp rises.
- **Anomalies:** Small island nations report volatile per-capita CO₂ due to small denominators; early GDP values in some African economies reflect sparse reporting.

## 6. Caveats & Limitations
- Sparse coverage in education, R&D, and energy mix indicators before 1980s.
- Interpolation does not extrapolate at series edges (values remain NaN).
- Winsorization mitigates outliers but anomalies remain (e.g., oil-rich microstates).
- Correlations show associations only, not causation.

## 7. Next Steps
- Integrate policy data (e.g., Paris Agreement, subsidies).
- Estimate panel regressions and ML models to quantify drivers of emissions.
- Explore heterogeneity by income group and region.
- Use dataset for scenario analysis (GDP shocks, renewable expansion, urbanization).
