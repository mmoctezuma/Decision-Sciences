# -*- coding: utf-8 -*-
"""
Pipeline dual: Panel FE regression | XGBoost
--------------------------------------------
- Responde: “Si el PIB sube k%, ¿en cuánto % cambian las emisiones de CO2?”
- Modo FE: estima elasticidad (β) directamente.
- Modo XGB: simula predicción base vs. escenario con +k% PIB.
- Ambos modos soportan:
    --predict → baseline
    --split_year → train/test split para métricas out-of-sample

Ejemplo:
    python co2_gdp_model.py --input_file data/processed/wide_clean.csv \
        --output_dir outputs --model_type fe --split_year 2015 --predict

    python co2_gdp_model.py --input_file data/processed/wide_clean.csv \
        --output_dir outputs --model_type xgb --split_year 2015
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
import xgboost as xgb

try:
    from linearmodels.panel import PanelOLS
except Exception:
    PanelOLS = None


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


# ====================================================
#  MÉTRICAS
# ====================================================
def compute_metrics(y_true, y_pred, label="") -> dict:
    metrics = {
        f"rmse{label}": mean_squared_error(y_true, y_pred) ** 0.5,
        f"mae{label}": mean_absolute_error(y_true, y_pred),
        f"r2{label}": r2_score(y_true, y_pred),
        f"n_obs{label}": len(y_true),
    }
    return metrics


# ====================================================
#  MODO 1: FIXED EFFECTS
# ====================================================
def fit_panel_fe(X, ln_controls, use_time_effects, cluster_entity, single_country):
    Xp = X.set_index(["iso3c", "year"]).sort_index()
    rhs_vars = ["ln_GDP"] + (ln_controls if ln_controls else [])
    y = Xp["ln_CO2"]
    Xvars = Xp[rhs_vars]
    
    mod = PanelOLS(
        y, 
        Xvars, 
        entity_effects=not single_country,
        time_effects=use_time_effects,
        drop_absorbed=True,
        check_rank=False
    )
    cov_kw = {"cluster_entity": True} if (cluster_entity and not single_country) else {}
    res = mod.fit(cov_type="clustered" if cov_kw else "unadjusted", **cov_kw)
    beta = res.params["ln_GDP"]
    return res, float(beta)


# ====================================================
#  MODO 2: XGBOOST
# ====================================================
def fit_xgb(X, controls=None):
    if controls is None:
        controls = []
    features = ["ln_GDP"] + [f"ln_{c}" for c in controls if f"ln_{c}" in X.columns]
    dtrain = xgb.DMatrix(X[features], label=X["ln_CO2"])
    params = {"objective": "reg:squarederror", "max_depth": 4, "eta": 0.1, "subsample": 0.9}
    bst = xgb.train(params, dtrain, num_boost_round=300)
    return bst, features


def simulate_gdp_shock_xgb(model, features, X, shock_pct=10):
    X_base, X_shock = X.copy(), X.copy()
    shock_ln = np.log(1 + shock_pct / 100.0)
    X_shock["ln_GDP"] += shock_ln

    y_base = model.predict(xgb.DMatrix(X_base[features]))
    y_shock = model.predict(xgb.DMatrix(X_shock[features]))

    df_out = X[["iso3c", "Country", "year"]].copy()
    df_out["CO2_base"] = np.exp(y_base)
    df_out["CO2_scenario"] = np.exp(y_shock)
    df_out["expected_change_pct"] = (df_out["CO2_scenario"] / df_out["CO2_base"] - 1.0) * 100.0
    df_out["shock_gdp_pct"] = shock_pct
    return df_out


# ====================================================
#  MAIN
# ====================================================
def main():
    parser = argparse.ArgumentParser(description="Panel FE vs. XGBoost → forecast + escenario PIB")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target", default="EN.GHG.CO2.MT.CE.AR5")
    parser.add_argument("--gdp", default="NY.GDP.MKTP.CD")
    parser.add_argument("--controls", nargs="*", default=['SP.POP.TOTL', 'SP.URB.TOTL.IN.ZS', 'EG.FEC.RNEW.ZS'])
    parser.add_argument("--time_effects", action="store_true")
    parser.add_argument("--shock_pct", type=float, default=10.0)
    parser.add_argument("--model_type", choices=["fe", "xgb"], default="fe")
    parser.add_argument("--predict", action="store_true", help="Generar predicciones baseline (sin shock)")
    parser.add_argument("--split_year", type=int, default=None, help="Último año para train (resto = test)")
    args = parser.parse_args()

    _ensure_dir(args.output_dir)

    df = load_panel(args.input_file)
    X, ln_controls = build_features(df, target=args.target, gdp=args.gdp, controls=args.controls)

    if args.split_year:
        train = X[X["year"] <= args.split_year]
        test = X[X["year"] > args.split_year]
    else:
        train, test = X, pd.DataFrame()

    if args.model_type == "fe":
        single_country = train["iso3c"].nunique() == 1
        res, beta = fit_panel_fe(
            train, ln_controls, use_time_effects=args.time_effects,
            cluster_entity=True, single_country=single_country
        )
        
        Xp_train = train.set_index(["iso3c", "year"]).sort_index()[res.model.exog.vars]
        yhat_train = res.predict(Xp_train)
        metrics_train = compute_metrics(train["ln_CO2"], yhat_train, "_train")
        metrics = metrics_train

        if not test.empty:
            Xp_test = test.set_index(["iso3c", "year"]).sort_index()[res.model.exog.vars]
            yhat_test = res.predict(Xp_test)
            metrics_test = compute_metrics(test["ln_CO2"], yhat_test, "_test")
            metrics.update(metrics_test)

        pd.DataFrame([metrics]).to_csv(os.path.join(args.output_dir, "metrics_fe.csv"), index=False)

        if args.predict:
            all_pred = pd.concat([train, test], axis=0).reset_index(drop=True)
            Xp_all = all_pred.set_index(["iso3c", "year"]).sort_index()[res.model.exog.vars]
            yhat_all = res.predict(Xp_all)
            
            all_pred["ln_CO2_hat"] = yhat_all.reset_index(drop=True)
            all_pred["CO2_pred"] = np.exp(all_pred["ln_CO2_hat"])
            
            all_pred.to_csv(os.path.join(args.output_dir, "predictions_fe.csv"), index=False)
            print("[FE] Predicciones baseline guardadas")
        else:
            print(f"[FE] β(ln GDP → ln CO2) = {beta:.3f}")
            print(f"➡ {args.shock_pct}% más PIB ⇒ cambio esperado ≈ {beta * args.shock_pct:.2f}%")

    else:  # XGB
        bst, features = fit_xgb(train, controls=args.controls)

        yhat_train = bst.predict(xgb.DMatrix(train[features]))
        metrics_train = compute_metrics(train["ln_CO2"], yhat_train, "_train")
        metrics = metrics_train

        if not test.empty:
            yhat_test = bst.predict(xgb.DMatrix(test[features]))
            metrics_test = compute_metrics(test["ln_CO2"], yhat_test, "_test")
            metrics.update(metrics_test)

        pd.DataFrame([metrics]).to_csv(os.path.join(args.output_dir, "metrics_xgb.csv"), index=False)

        if args.predict:
            X_out = X[["iso3c", "Country", "year"]].copy()
            y_pred = bst.predict(xgb.DMatrix(X[features]))
            X_out["CO2_pred"] = np.exp(y_pred)
            X_out.to_csv(os.path.join(args.output_dir, "predictions_xgb.csv"), index=False)
            print("[XGB] Predicciones baseline guardadas")
        else:
            scen = simulate_gdp_shock_xgb(bst, features, X, shock_pct=args.shock_pct)
            scen.to_csv(os.path.join(args.output_dir, f"scenario_xgb_{int(args.shock_pct)}pct.csv"), index=False)
            print(f"[XGB] Escenario +{args.shock_pct}% PIB generado")


if __name__ == "__main__":
    main()

