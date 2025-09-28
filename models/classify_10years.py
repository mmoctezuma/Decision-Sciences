# -*- coding: utf-8 -*-
import argparse
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from models.helpers import load_panel, build_features, extended_data
from models.co2_models import fit_xgb

logger = logging.getLogger(__name__)


"""Projection helpers moved to models.helpers"""

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
    logger.info("Resultados guardados en %s", output_file_compare)
    
    iso_likely = dict(zip(compare["iso3c"], compare["likely_reduce_CO2"]))
    df['likely_reduce_CO2'] = df['iso3c'].map(iso_likely)
    df.to_csv(output_file_df, index=False)


# =========================
#  ENTRYPOINT CLI
# =========================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
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
