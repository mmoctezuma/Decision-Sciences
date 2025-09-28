# Strategic Recommendation Report — 5-Year Renewable Investment Scenario

**Objective.** Estimate the likelihood that a strong investment in renewables leads to a **CO₂ reduction within 5 years**, and rank **investment priorities** (hydro vs non-hydro) by country.

**Method (high level).** We train an XGBoost model on ln-CO₂ using ln-GDP and energy controls; project each country 5 years ahead (CAGR); create a scenario with an additional *Δ share* in the chosen renewable technology; compare **baseline vs scenario**; convert the % reduction into a **probability of success**; and test marginal sensitivity (+5 pp) to prioritize **hydro vs non-hydro**.

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
---

## Global snapshot (5-year horizon)
- Countries evaluated: **181**
- Avg. reduction across countries: **4.87%**
- Share with probability of success > 0.7: **0.13**
- Scenario parameters: **Δ10.0 pp in EG.ELC.RNEW.ZS over 5 yrs**

> *Interpretation.* A negative % means lower CO₂ under the investment scenario. The probability maps that % to a success likelihood; values >0.7 indicate high confidence.

---

## Ranked list A — Highest likelihood of reduction (Top 15)
(Countries sorted by **probability**, then by **% reduction**)

| ISO3 | Country | Region | Prob. (%) | ΔCO₂% (5y) | Grid gCO₂/kWh |
|---|---|---|---:|---:|---:|
| PRI | Puerto Rico (US) | nan | 95.1 | -37.18 | 441 |
| BLZ | Belize | nan | 94.9 | -36.52 | 68 |
| GUM | Guam | nan | 94.1 | -34.51 | 663 |
| LSO | Lesotho | nan | 93.6 | -33.60 | 52 |
| PNG | Papua New Guinea | nan | 89.7 | -26.99 | 284 |
| BRA | Brazil | nan | 88.4 | -25.34 | 66 |
| VIR | Virgin Islands (U.S.) | nan | 87.4 | -24.19 | 681 |
| ZMB | Zambia | nan | 86.5 | -23.27 | 89 |
| FRO | Faroe Islands | nan | 86.4 | -23.10 | 369 |
| LUX | Luxembourg | nan | 84.7 | -21.34 | 46 |
| BRB | Barbados | nan | 82.4 | -19.32 | 644 |
| VCT | St. Vincent and the Grenadines | nan | 82.3 | -19.23 | 541 |
| LCA | St. Lucia | nan | 80.6 | -17.76 | 681 |
| PAK | Pakistan | nan | 76.6 | -14.81 | 339 |
| HKG | Hong Kong SAR, China | nan | 75.7 | -14.23 | 677 |


---

## Ranked list B — Largest absolute CO₂ cuts (Mt, Top 15)
(Countries with **biggest absolute drop** regardless of probability)

| ISO3 | Country | Region | ΔCO₂ (Mt) | ΔCO₂% (5y) | Prob. (%) |
|---|---|---:|---:|---:|---:|
| BRA | Brazil | nan | 112.53 | -25.34 | 88.4 |
| IND | India | nan | 68.99 | -1.74 | 53.5 |
| PAK | Pakistan | nan | 68.41 | -14.81 | 76.6 |
| KOR | Korea, Rep. | nan | 48.42 | -7.72 | 65.0 |
| FRA | France | nan | 38.91 | -7.04 | 63.7 |
| COL | Colombia | nan | 30.04 | -13.14 | 74.1 |
| GBR | United Kingdom | nan | 28.85 | -8.11 | 65.7 |
| ZAF | South Africa | nan | 15.29 | -3.01 | 56.0 |
| DZA | Algeria | nan | 15.17 | -5.66 | 61.1 |
| MAR | Morocco | nan | 14.14 | -6.59 | 62.9 |
| TUR | Turkiye | nan | 13.41 | -3.26 | 56.5 |
| MEX | Mexico | nan | 12.07 | -1.50 | 53.0 |
| PRI | Puerto Rico (US) | nan | 11.77 | -37.18 | 95.1 |
| THA | Thailand | nan | 9.36 | -3.13 | 56.2 |
| JPN | Japan | nan | 8.43 | -1.01 | 52.0 |


---

## Investment prioritization — Hydro vs Non-Hydro
We compute marginal sensitivity (+5 pp) for **hydro** and **non-hydro renewables** and recommend the option with the **larger 5-year CO₂ drop** (more negative Δ).

**Top 20 countries by marginal gain**

| ISO3 | Country | Priority | Δ% (Hydro +5pp) | Δ% (Non-Hydro +5pp) | Grid gCO₂/kWh |
|---|---|---|---:|---:|---:|
| PRI | Puerto Rico (US) | hydro | -44.86 | -37.91 | 441 |
| BLZ | Belize | non_hydro_renewables | -36.39 | -36.68 | 68 |
| GUM | Guam | hydro | -36.33 | -34.21 | 663 |
| BRB | Barbados | hydro | -32.91 | -21.91 | 644 |
| VIR | Virgin Islands (U.S.) | non_hydro_renewables | -24.94 | -26.71 | 681 |
| PNG | Papua New Guinea | non_hydro_renewables | -22.85 | -26.05 | 284 |
| LCA | St. Lucia | hydro | -24.59 | -16.58 | 681 |
| LSO | Lesotho | non_hydro_renewables | 3.40 | -22.04 | 52 |
| FRO | Faroe Islands | non_hydro_renewables | -20.51 | -20.93 | 369 |
| TON | Tonga | hydro | -20.59 | 27.38 | 569 |
| VCT | St. Vincent and the Grenadines | non_hydro_renewables | -10.39 | -18.49 | 541 |
| LUX | Luxembourg | non_hydro_renewables | -13.03 | -17.16 | 46 |
| MMR | Myanmar | hydro | -15.76 | -10.16 | 415 |
| BRA | Brazil | hydro | -14.44 | -8.58 | 66 |
| HKG | Hong Kong SAR, China | non_hydro_renewables | -7.27 | -13.78 | 677 |
| SWZ | Eswatini | non_hydro_renewables | -8.85 | -13.62 | 111 |
| VUT | Vanuatu | hydro | -13.33 | -11.30 | 448 |
| TLS | Timor-Leste | hydro | -12.99 | -0.35 | 697 |
| DMA | Dominica | non_hydro_renewables | -7.18 | -11.58 | 491 |
| DZA | Algeria | hydro | -11.36 | -4.29 | 329 |


---

## Strategic recommendations

1. **Sequence investments by probability and impact.** Start with countries in *Ranked list A* (probability > 70%) that also appear in *Ranked list B* (large absolute cuts). This maximizes early wins and measurable Mt-CO₂ reductions.
2. **Pick technology by country sensitivity.** Use the **Priority** column above:
   - `non_hydro_renewables` → favor **solar/wind** build-out and grid integration (storage, flexibility).
   - `hydro` → prioritize **hydro upgrades/new capacity** where hydrology and social safeguards are favorable.
3. **Target “dirty grids” first.** Where **grid gCO₂/kWh** is high (e.g., >500 gCO₂/kWh), non-hydro often yields faster cuts by crowding out **coal**. Couple with explicit **coal retirement schedules**.
4. **Lock in certainty with policy design.** Technology-neutral auctions with carbon-intensity penalties, firm PPAs, and accelerated permitting will increase the realized Δ share over the 5-year window.
5. **De-risk the pipeline.** Use blended finance and grid CAPEX (transmission + storage) in countries where the **best_delta** is large but probability is middling (0.5–0.7).

### Expected outcomes (portfolio level)
- **CO₂ trajectory:** Negative ΔCO₂% in the countries above; aggregate average near the global snapshot figure.
- **Risk profile:** Highest success rates in countries with (i) improving renewable share, (ii) high fossil baseline, and (iii) stable GDP growth.
- **Time-to-impact:** First 24–36 months for construction & grid works; visible CO₂ effects by year 3–5 as renewable dispatch scales.

---

### Notes & limitations
- This is a **reduced-form, system-level** model; it does not endogenize demand-side efficiency, hydro seasonality, or sub-national siting.
- Results depend on **data coverage** and on the assumed **Δ share ramp** across five years.
- For executive decisions, complement with **project-level LCOE**, grid constraints, and social/environmental risk screenings.
