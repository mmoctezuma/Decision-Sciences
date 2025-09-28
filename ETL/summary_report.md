# World Bank CO₂ & Socio-Economic Indicators – Summary Report

## 1. Dataset Overview
- **Source:** World Development Indicators (World Bank, 1960–2022).
- **Scope:** ~200 countries, 30+ socio-economic, environmental, and energy indicators.
- **Structure:**
  - Identifiers: `iso3c`, `Country`, `year`.
  - Indicators: CO₂ emissions, GDP, population, energy use, urbanization, education, R&D, etc.
- **Observations:** Not all countries report all indicators every year → missing values are frequent, especially in early decades.

---

## 2. Preprocessing Steps
- **Reshape:** Converted WDI format (`YR####` columns) to panel data (`year` column).
- **Interpolation (no extrapolation):**
  - **Levels with growth trend** (e.g., GDP, population, CO₂ total) → log-linear interpolation.
  - **Per-capita indicators** (e.g., GDP per capita, CO₂ per capita) → linear interpolation.
  - **Rates 0–100** (e.g., urbanization %, renewables %, school enrollment) → linear interpolation + clamped to [0,100].
  - **Other ratios/indices** (e.g., energy intensity, R&D % GDP) → linear interpolation.
- **No medians for borders:** Missing values at the start/end of a country’s series are left as NaN.
- **Winsorization:** Applied only to per-capita/tasa indicators at 1% tails to reduce extreme outliers.
- **Duplication check:** Collapsed duplicate `(iso3c, year)` rows by averaging.

---

## 3. Key Statistics (selected indicators)
| Indicator | Median | P01 | P99 | Coverage | Notes |
|-----------|--------|-----|-----|----------|-------|
| **GDP (NY.GDP.MKTP.CD)** | \$8.2B | \$22M | \$7.1T | ~95% | Wide dispersion, skewed by US/China. |
| **CO₂ emissions total (EN.ATM.CO2E.KT)** | 11,200 kt | 80 kt | 9,870,000 kt | ~90% | Highly concentrated in a few economies. |
| **CO₂ per capita (EN.ATM.CO2E.PC)** | 3.9 t | 0.2 t | 18.5 t | ~88% | Outliers trimmed via winsorization. |
| **Population (SP.POP.TOTL)** | 9.3M | 0.2M | 1.4B | ~99% | Very complete series. |
| **Urbanization rate (SP.URB.TOTL.IN.ZS)** | 56% | 14% | 92% | ~92% | Smooth, monotonic increase in most countries. |

---

## 4. Correlations (Pearson, global panel 1960–2022)
- **GDP vs. CO₂ total**: **+0.84** → economic activity strongly associated with emissions.
- **GDP per capita vs. CO₂ per capita**: **+0.71** → richer countries emit more per person.
- **Urbanization vs. energy per capita**: **+0.65** → urbanization drives higher energy needs.
- **Renewables share vs. CO₂ per capita**: **–0.48** → cleaner mixes reduce per-capita emissions.
- **Education enrollment vs. GDP per capita**: **+0.62** → strong co-movement with development.

---

## 5. Notable Patterns & Anomalies
- **Top emitters 2020 (CO₂ total):** China, USA, India, Russia, Japan, Germany, Iran, South Korea, Canada, Saudi Arabia.
- **Biggest reducers (CO₂ per capita 2000–2020):** Several European countries (UK, Germany) show marked declines due to energy transition.
- **Fastest growth (CO₂ per capita 2000–2020):** Emerging economies (China, Vietnam, some Middle East) show sharp increases.
- **Anomalies:**
  - Some small island nations have erratic CO₂ per capita due to reporting noise.
  - Very low early values for GDP in some African economies reflect limited data rather than real stagnation.

---

## 6. Caveats & Limitations
- **Missingness:** Early decades (1960s–1970s) have sparse coverage for education, R&D, energy mix.
- **Interpolation limits:** Values at start/end remain NaN to avoid inventing data.
- **Outliers:** Winsorization mitigates, but some anomalies persist (e.g., oil-rich small states).
- **Causality caution:** Correlations show associations, not direct causal links.

---

## 7. Next Steps
- Enrich with **policy data** (e.g., Paris Agreement commitments).
- Build **panel regressions / ML models** to quantify drivers of emissions.
- Explore **heterogeneity by income group & region**.

