# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from models.co2_models import load_panel, build_features, fit_xgb


# =========================
#  FUNCIONES AUXILIARES
# =========================
def compute_cagr(series):
    valid = series.dropna()
    if len(valid) < 2:
        return None
    valid = valid.sort_index()
    start, end = valid.iloc[0], valid.iloc[-1]
    years = len(valid) - 1
    if start > 0 and end > 0 and years > 0:
        return (end/start)**(1/years) - 1 
    else:
        return np.nan


def project_series(series, years=10):
    g = compute_cagr(series)
    if g is None or not np.isfinite(g):
        return pd.Series([np.nan]*years,
                         index=range(series.index[-1]+1, series.index[-1]+years+1))
    projections = [series.iloc[-1] * ((1+g)**i) for i in range(1, years+1)]
    return pd.Series(projections,
                     index=range(series.index[-1]+1, series.index[-1]+years+1))


def extend_country(df_country, years=10):
    df_country = df_country.sort_values("year")
    results = []
    for col in df_country.columns:
        if col in ["iso3c", "Country", "year", "region"]:
            continue
        series = df_country.set_index("year")[col].dropna()
        if len(series) > 1:
            proj = project_series(series, years)
            proj.name = col
            results.append(proj)
    if not results:
        return None
    df_proj = pd.concat(results, axis=1).reset_index().rename(columns={"index": "year"})
    df_proj["iso3c"] = df_country["iso3c"].iloc[0]
    df_proj["Country"] = df_country["Country"].iloc[0]
    if "region" in df_country.columns:
        df_proj["region"] = df_country["region"].iloc[0]
    return df_proj


def extended_data(df, years=10):
    extended_rows = []
    for iso, group in df.groupby("iso3c"):
        res = extend_country(group, years=years)
        if res is not None and not res.empty:
            extended_rows.append(res)
    if not extended_rows:
        return pd.DataFrame()
    return pd.concat(extended_rows, ignore_index=True)

# =========================
#  MAIN
# =========================
def main(input_file, output_file_compare, output_file_df, target, gdp, controls, years):
    df = load_panel(input_file)
    X, ln_controls = build_features(df, target=target, gdp=gdp, controls=controls)
    bst, features = fit_xgb(X, controls=controls)
    
    future_df = extended_data(df, years=10)
    
    future_X, _ = build_features(future_df, target=target, gdp=gdp, controls=controls)
    dmat_future = xgb.DMatrix(future_X[features])
    y_future = bst.predict(dmat_future)
    future_X["CO2_pred"] = np.exp(y_future)
    
    last10 = (
        X[X["year"] >= (X["year"].max() - 9)]
        .groupby("iso3c")["ln_CO2"]
        .mean()
    )
    next10 = (
        future_X[future_X["year"] > X["year"].max()]
        .groupby("iso3c")["CO2_pred"]
        .apply(lambda s: np.log(s).mean())
    )

    compare = pd.DataFrame({
        "last10_lnCO2": last10,
        "next10_lnCO2": next10
    })
    compare.reset_index(inplace=True)
    compare["delta"] = compare["next10_lnCO2"] - compare["last10_lnCO2"]
    compare["trend"] = compare["delta"].apply(lambda x: "disminuye" if x < 0 else "sube")
    compare["likely_reduce_CO2"] = compare["delta"].apply(lambda x: 1 if x < 0 else 0)
    
    compare.to_csv(output_file_compare, index=False)
    print(f"Resultados guardados en {output_file_compare}")
    
    iso_likely = dict(zip(compare["iso3c"], compare["likely_reduce_CO2"]))
    df['likely_reduce_CO2'] = df['iso3c'].map(iso_likely)
    df.to_csv(output_file_df, index=False)


# =========================
#  ENTRYPOINT CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clasifica si los países subirán o bajarán CO2 en 10 años")
    parser.add_argument("--input_file", default="data/processed/wide_clean.csv", help="Archivo de entrada (panel wide)")
    parser.add_argument("--output_file_compare", default="model_results/10years/compare_10years.csv", help="Archivo CSV de salida para comparacion")
    parser.add_argument("--output_file_df", default="data/processed/panel_target.csv", help="Archivo CSV de salida para df con target")
    parser.add_argument("--target", default="EN.GHG.CO2.MT.CE.AR5", help="Variable target CO2")
    parser.add_argument("--gdp", default="NY.GDP.MKTP.CD", help="Variable PIB")
    parser.add_argument("--controls", nargs="+", default=['SP.POP.TOTL', 'SP.URB.TOTL.IN.ZS', 'EG.FEC.RNEW.ZS'], help="Variables de control")
    parser.add_argument("--years", default=10, help="Años a futuro")
    args = parser.parse_args()

    main(args.input_file, args.output_file_compare, args.output_file_df, args.target, args.gdp, args.controls, args.years)

