# -*- coding: utf-8 -*-

"""
Pipeline: Panel FE regression (log–log) → forecast + interpretación del +k% PIB (c.p.)

- Responde: “Si el PIB sube k%, ¿en cuánto % cambian las emisiones de CO2?”
- Soporta filtrar por país con --country (iso3c). Si no se pasa, usa todos los países.

Requisitos:
  pip install linearmodels pandas numpy statsmodels scikit-learn

Entradas (desde tu ETL):
  - <workdir>/wide_clean.csv  con columnas: iso3c, Country, year, EN.ATM.CO2E.KT, NY.GDP.MKTP.CD, ...

Salidas (en <output_dir>/fe/):
  - model_summary[_{ISO3C}].txt
  - metrics[_{ISO3C}].csv
  - predictions_full[_{ISO3C}].csv
  - elasticity_table[_{ISO3C}].csv
  - scenario_gdp_plus{K}pct[_{ISO3C}].csv
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm

_USING_LM = True
try:
    from linearmodels.panel import PanelOLS
except Exception:
    _USING_LM = False


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_panel(workdir: str) -> pd.DataFrame:
    df = pd.read_csv(workdir)
    df = df.dropna(subset=["iso3c", "year"])
    df["year"] = df["year"].astype(int)
    return df


def build_features(
    df: pd.DataFrame,
    target: str = "EN.GHG.CO2.MT.CE.AR5",
    gdp: str = "NY.GDP.MKTP.CD",
    controls: list | None = None,
) -> tuple[pd.DataFrame, list]:
    """Crea ln_CO2, ln_GDP y ln_controles; filtra filas con NaN en esas columnas."""
    if controls is None:
        controls = ["SP.POP.TOTL", "SP.URB.TOTL.IN.ZS", "EN.GHG.CO2.LU.MT.CE.AR5"]

    keep = ["iso3c", "Country", "year", target, gdp] + [c for c in controls if c in df.columns]
    X = df.loc[:, [c for c in keep if c in df.columns]].copy()

    def _safe_log(s: pd.Series, eps=1e-9):
        s = pd.to_numeric(s, errors="coerce")
        min_pos = s[s > 0].min()
        floor = float(min_pos) if pd.notna(min_pos) else eps
        return np.log(s.clip(lower=floor))

    X["ln_CO2"] = _safe_log(X[target])
    X["ln_GDP"] = _safe_log(X[gdp])

    ln_controls = []
    for c in controls:
        if c in X.columns:
            X[f"ln_{c}"] = _safe_log(X[c])
            ln_controls.append(f"ln_{c}")

    X = X.dropna(subset=["ln_CO2", "ln_GDP"] + ln_controls)
    return X, ln_controls


def fit_panel_fe(
    X: pd.DataFrame,
    ln_controls: list,
    use_time_effects: bool,
    cluster_entity: bool,
    workdir: str | None,
    single_country: bool,
    iso3c_code: str | None,
):
    """
    - Si hay múltiples países: FE de país + (opcional) FE de año.
    - Si hay un solo país: usa solo FE de año (o dummies de año).
    Devuelve (resultados, beta, fitted_df).
    """
    fe_dir = os.path.join(workdir, "fe") if workdir else None
    if fe_dir:
        _ensure_dir(fe_dir)

    suffix = f"_{iso3c_code}" if iso3c_code else ""

    if _USING_LM:
        Xp = X.set_index(["iso3c", "year"]).sort_index()
        rhs = "ln_GDP"
        if ln_controls:
            rhs += " + " + " + ".join(ln_controls)

        if single_country:
            rhs_fe = rhs + ((" + TimeEffects") if use_time_effects else "")
        else:
            rhs_fe = rhs + " + EntityEffects" + ((" + TimeEffects") if use_time_effects else "")

        mod = PanelOLS.from_formula(f"ln_CO2 ~ {rhs_fe}", data=Xp)
        cov_kw = {"cluster_entity": True} if (cluster_entity and not single_country) else {}
        res = mod.fit(cov_type="clustered" if cov_kw else "unadjusted", **cov_kw)
        beta = res.params["ln_GDP"]

        if fe_dir:
            with open(os.path.join(fe_dir, f"model_summary{suffix}.txt"), "w", encoding="utf-8") as f:
                f.write(str(res.summary))

        fitted = res.fitted_values.reset_index().rename(columns={"fitted_values": "ln_CO2_hat"})
        fitted["CO2_hat"] = np.exp(fitted["ln_CO2_hat"])
        return res, float(beta), fitted

    d = X.copy().set_index(["iso3c", "year"]).sort_index()
    Z_list = [d[["ln_GDP"] + ln_controls]]
    if not single_country:
        ent_dum = pd.get_dummies(d.index.get_level_values(0), prefix="c", drop_first=True)
        Z_list.append(ent_dum)
    if use_time_effects:
        time_dum = pd.get_dummies(d.index.get_level_values(1), prefix="t", drop_first=True)
        Z_list.append(time_dum)
    Z = pd.concat(Z_list, axis=1)
    Z = sm.add_constant(Z)

    if not single_country:
        model = sm.OLS(d["ln_CO2"], Z).fit(
            cov_type="cluster", cov_kwds={"groups": d.index.get_level_values(0)}
        )
    else:
        model = sm.OLS(d["ln_CO2"], Z).fit(cov_type="HC1")

    beta = model.params["ln_GDP"]

    if fe_dir:
        with open(os.path.join(fe_dir, f"model_summary{suffix}.txt"), "w", encoding="utf-8") as f:
            f.write(str(model.summary()))

    ln_hat = model.predict(Z)
    fitted = d.reset_index()[["iso3c", "year"]].copy()
    fitted["ln_CO2_hat"] = ln_hat.values
    fitted["CO2_hat"] = np.exp(fitted["ln_CO2_hat"])
    return model, float(beta), fitted


def build_elasticity_table(beta: float, shock_pct: float = 10.0) -> pd.DataFrame:
    shock = shock_pct / 100.0
    factor = (1.0 + shock) ** beta
    approx_pct = beta * shock_pct
    exact_pct = (factor - 1.0) * 100.0
    return pd.DataFrame({
        "beta_lnGDP_to_lnCO2": [beta],
        "shock_gdp_pct": [shock_pct],
        "expected_change_co2_pct_linear": [approx_pct],
        "expected_change_co2_pct_exact": [exact_pct],
        "level_multiplier": [factor],
    })


def scenario_gdp_plus_k(
    df: pd.DataFrame,
    target_col: str,
    beta: float,
    shock_pct: float,
    base_year: int | None = None,
    country: str | None = None,
) -> pd.DataFrame:
    """Escenario: CO₂_new = CO₂_base * (1+shock)^beta, por país o país específico."""
    shock = shock_pct / 100.0
    mult = (1.0 + shock) ** beta

    d = df if country is None else df[df["iso3c"] == country]
    if base_year is None:
        base = (d.dropna(subset=[target_col])
                  .sort_values(["iso3c", "year"])
                  .groupby("iso3c")
                  .tail(1)[["iso3c", "Country", "year", target_col]]
                  .rename(columns={target_col: "CO2_base", "year": "base_year"}))
    else:
        base = (d[d["year"] == base_year][["iso3c", "Country", "year", target_col]]
                  .rename(columns={target_col: "CO2_base", "year": "base_year"}))
        base = base.dropna(subset=["CO2_base"])

    base["CO2_scenario"] = base["CO2_base"] * mult
    base["expected_change_pct"] = (base["CO2_scenario"] / base["CO2_base"] - 1.0) * 100.0
    base["shock_gdp_pct"] = shock_pct
    base["beta_lnGDP_to_lnCO2"] = beta
    return base.sort_values("CO2_base", ascending=False)


def compute_metrics(df: pd.DataFrame) -> dict:
    """Calcula métricas básicas para CO₂ log y nivel."""
    metrics = {}
    if "ln_CO2" in df and "ln_CO2_hat" in df:
        y_true, y_pred = df["ln_CO2"], df["ln_CO2_hat"]
        metrics["rmse_log"] = mean_squared_error(y_true, y_pred) ** 0.5
        metrics["mae_log"] = mean_absolute_error(y_true, y_pred)
        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        metrics["r2_log"] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    if "CO2_true" in df and "CO2_hat" in df:
        y_true, y_pred = df["CO2_true"], df["CO2_hat"]
        metrics["rmse_level"] = mean_squared_error(y_true, y_pred) ** 0.5
        metrics["mae_level"] = mean_absolute_error(y_true, y_pred)
        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        metrics["r2_level"] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    metrics["n_obs"] = len(df)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Panel FE regression → forecast + interpretación del +k% PIB (c.p.)")
    parser.add_argument("--input_file", required=True, default='data/processed/wide_clean.csv')
    parser.add_argument("--output_dir", required=True, help="Directorio de trabajo.")
    parser.add_argument("--target", default="EN.GHG.CO2.MT.CE.AR5", help="Variable dependiente (CO2 total).")
    parser.add_argument("--gdp", default="NY.GDP.MKTP.CD", help="Columna de PIB total.")
    parser.add_argument("--controls", nargs="*", default=["SP.POP.TOTL", "SP.URB.TOTL.IN.ZS", "EG.USE.PCAP.KG.OE"],
                    help="Controles (default población, urbanización, energía pc).")
    parser.add_argument("--time_effects", action="store_true", help="Incluir efectos fijos de año.")
    parser.add_argument("--shock_pct", type=float, default=10.0, help="Shock de PIB en % (default 10).")
    parser.add_argument("--base_year", type=int, default=None, help="Año base (si None, usa último por país).")
    parser.add_argument("--country", type=str, default=None,
                    help="Código iso3c del país (ej. MEX, USA). Si no se da, usa todos.")
    args = parser.parse_args()

    out_dir = args.output_dir
    _ensure_dir(os.path.join(out_dir, 'fe'))

    df = load_panel(args.input_file)
    if args.country:
        df = df[df["iso3c"] == args.country]
        if df.empty:
            raise ValueError(f"No hay datos para {args.country}")
        print(f"📌 Entrenando solo para {args.country}")

    X, ln_controls = build_features(df, target=args.target, gdp=args.gdp, controls=args.controls)

    single_country = X["iso3c"].nunique() == 1
    iso3c_code = X["iso3c"].iloc[0] if single_country else None
    suffix = f"_{iso3c_code}" if iso3c_code else ""

    res, beta, fitted = fit_panel_fe(
        X,
        ln_controls=ln_controls,
        use_time_effects=args.time_effects,
        cluster_entity=True,
        workdir=args.output_dir,
        single_country=single_country,
        iso3c_code=iso3c_code,
    )
    
    merged = X.merge(fitted, on=["iso3c", "year"], how="inner")
    merged = merged.rename(columns={args.target: "CO2_true"})
    merged["residual_ln"] = merged["ln_CO2"] - merged["ln_CO2_hat"]
    merged.to_csv(os.path.join(out_dir, f"predictions_full{suffix}.csv"), index=False)

    metrics = compute_metrics(merged)
    pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, f"metrics{suffix}.csv"), index=False)

    elastic = build_elasticity_table(beta, shock_pct=args.shock_pct)
    elastic.to_csv(os.path.join(out_dir, f"elasticity_table{suffix}.csv"), index=False)
    
    scen = scenario_gdp_plus_k(
        df=df,
        target_col=args.target,
        beta=beta,
        shock_pct=args.shock_pct,
        base_year=args.base_year,
        country=iso3c_code if single_country else None
    )
    scen.to_csv(os.path.join(out_dir, f"scenario_gdp_plus{int(args.shock_pct)}pct{suffix}.csv"), index=False)

    print(f"Modelo FE listo. β(ln GDP → ln CO2) = {beta:.3f}")
    print(f"➡ Un {args.shock_pct:.1f}% más de PIB ⇒ CO₂ ≈ {elastic['expected_change_co2_pct_exact'].iat[0]:.2f}% (exacto)")
    if single_country:
        print(f"País: {iso3c_code}")


if __name__ == "__main__":
    main()

