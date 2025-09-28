# -*- coding: utf-8 -*-
"""
Renewable Investment Scenario – 5-year CO2 impact + prioritization
------------------------------------------------------------------
Usa tu pipeline XGB (ln_CO2 ~ ln_GDP + ln_controles) para:
1) Proyectar baseline 5 años por país (tendencia).
2) Simular inversión en renovables con un aumento planificado (pp) distribuido en 5 años.
3) Comparar baseline vs. escenario y estimar probabilidad de reducción.
4) Priorización tecnológica (hidro vs. no-hidro renovables) por país vía sensibilidad marginal.

Entradas:
- data/processed/wide_clean.csv (o el panel que uses)
Salidas:
- model_results/renew/renew_5y_country_results.csv
- model_results/renew/renew_5y_priorities.csv
- model_results/renew/renew_5y_global_summary.csv
"""

import argparse
import os
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from models.helpers import load_panel, build_features, extended_data
from models.co2_models import fit_xgb 

logger = logging.getLogger(__name__)


"""Projection helpers moved to models.helpers"""


# --------------------- Limpieza y manejo del mix eléctrico --------------------- #
MIX_COLS = [
    "EG.ELC.COAL.ZS", "EG.ELC.NGAS.ZS", "EG.ELC.PETR.ZS",
    "EG.ELC.NUCL.ZS", "EG.ELC.HYRO.ZS", "EG.ELC.RNEW.ZS"
]

def clean_mix(df: pd.DataFrame) -> pd.DataFrame:
    """NaN→0, %→fracción, clip a [0,1] y renormaliza si suma>1."""
    df = df.copy()
    for c in MIX_COLS:
        if c not in df.columns:
            df[c] = 0.0
    df[MIX_COLS] = df[MIX_COLS].fillna(0.0)
    for c in MIX_COLS:
        df[c] = np.where(df[c] > 1.0, df[c] / 100.0, df[c]).clip(0, 1)
    row_sum = df[MIX_COLS].sum(axis=1)
    over = row_sum > 1.0
    if over.any():
        df.loc[over, MIX_COLS] = df.loc[over, MIX_COLS].div(row_sum[over], axis=0)
    return df


def compute_grid_intensity(df: pd.DataFrame) -> pd.Series:
    """gCO2/kWh aproximado a partir del mix."""
    EF = {"coal": 900, "gas": 469, "oil": 700.0, "nuclear": 16, "hydro": 10, "renew": 50}
    mix = {
        "coal": "EG.ELC.COAL.ZS",
        "gas": "EG.ELC.NGAS.ZS",
        "oil": "EG.ELC.PETR.ZS",
        "nuclear": "EG.ELC.NUCL.ZS",
        "hydro": "EG.ELC.HYRO.ZS",
        "renew": "EG.ELC.RNEW.ZS",
    }
    for c in mix.values():
        if c not in df.columns:
            df[c] = 0.0
    df = clean_mix(df)
    gpkwh = sum(df[mix[k]].fillna(0) * v for k, v in EF.items())
    return gpkwh


def increase_share(df_sc: pd.DataFrame, inc_col: str, inc_pp_total: float, years: int = 5) -> pd.DataFrame:
    """
    Incrementa inc_col en 'inc_pp_total' puntos porcentuales acumulados a lo largo de 5 años,
    con rampa lineal. Reduce COAL+GAS+OIL proporcionalmente para mantener suma <= 1.
    Lógica defensiva: NaN→0, normaliza si suma>1, soporta caso todo-0.
    """
    foss_cols = ["EG.ELC.COAL.ZS", "EG.ELC.NGAS.ZS", "EG.ELC.PETR.ZS"]
    all_mix = foss_cols + ["EG.ELC.NUCL.ZS", "EG.ELC.HYRO.ZS", "EG.ELC.RNEW.ZS"]

    df = df_sc.copy()
    for c in all_mix:
        if c not in df.columns:
            df[c] = 0.0
    df = clean_mix(df)

    inc_total = inc_pp_total / 100.0
    steps = np.linspace(0, inc_total, years)

    out = []
    for iso, g in df.groupby("iso3c"):
        g = g.sort_values("year")
        last_year = int(g["year"].max())
        base = g[g["year"] == last_year][all_mix].iloc[0].copy()
        base = base.fillna(0.0)
        if not np.isfinite(base.sum()):
            base[:] = 0.0
        elif float(base.sum()) == 0.0:
            base[:] = 0.0

        for i, step in enumerate(steps, start=1):
            y = last_year + i
            new = base.copy()
            new[inc_col] = np.clip(new[inc_col] + step, 0, 0.95)

            # Reducir fósiles si exceden el espacio libre
            non_foss_cols = ["EG.ELC.NUCL.ZS", "EG.ELC.HYRO.ZS", "EG.ELC.RNEW.ZS"]
            non_foss_sum = float(new[non_foss_cols].sum())
            foss_sum = float(new[foss_cols].sum())
            target_free = max(0.0, 1.0 - non_foss_sum)

            if foss_sum > 0 and target_free < foss_sum:
                ratio = target_free / foss_sum if foss_sum > 0 else 1.0
                for fc in foss_cols:
                    new[fc] = new[fc] * ratio

            s = float(new.sum())
            if s > 1.0:
                new = new / s

            row = {"iso3c": iso, "year": y}
            row.update(new.to_dict())
            out.append(row)
    return pd.DataFrame(out)


# --------------------- Probabilidad desde reducción % --------------------- #
def calibrated_probability(reduction_pct: pd.Series, k: float = 0.08) -> pd.Series:
    """
    Sigmoide centrada en 0%: 0% -> ~0.50 ; -10% -> ~0.80 ; -20% -> ~0.93
    P = 1 / (1 + exp(-k * (-reduction_pct)))  (más reducción => mayor prob)
    """
    x = -reduction_pct
    return 1.0 / (1.0 + np.exp(-k * x))


# --------------------- Escenario principal --------------------- #
def run_scenario(
    panel_csv: str,
    output_dir: str,
    years: int,
    renew_pp_total: float,
    invest_target: str,
    controls: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    os.makedirs(output_dir, exist_ok=True)

    # 1) Cargar panel + limpieza global del mix
    df = load_panel(panel_csv)
    logger.info("df shape=%s years=%s→%s countries=%s",
                df.shape, df["year"].min(), df["year"].max(), df["iso3c"].nunique())
    df = clean_mix(df)

    # 2) Validaciones de columnas y cobertura
    req_cols = ["EN.GHG.CO2.MT.CE.AR5", "NY.GDP.MKTP.CD"]
    missing_req = [c for c in req_cols if c not in df.columns]
    if missing_req:
        raise ValueError(f"[FATAL] Faltan columnas obligatorias: {missing_req}")

    controls = [c for c in controls if c in df.columns]
    if not controls:
        logger.warning("Sin controles disponibles; se entrenará con ln_GDP solamente.")
    else:
        cover = df[controls].notna().mean().sort_values(ascending=False)
        use70 = [c for c in cover.index if cover[c] >= 0.70]
        controls = use70 if use70 else [c for c in cover.index if cover[c] >= 0.50]
        logger.info("Controles usados: %s", controls)

    # 3) Features completas e info entrenable
    X_full, ln_controls = build_features(
        df,
        target="EN.GHG.CO2.MT.CE.AR5",
        gdp="NY.GDP.MKTP.CD",
        controls=controls
    )
    logger.info("X_full shape: %s", X_full.shape)
    logger.info("ln_controls: %s", ln_controls)
    train_cols = ["ln_CO2", "ln_GDP"] + [c for c in ln_controls if c in X_full.columns]
    rows_ok = X_full[train_cols].dropna().shape[0]
    logger.info("Filas entrenables: %s", rows_ok)
    if rows_ok == 0:
        raise RuntimeError("[FATAL] X_full quedó vacío tras build_features. Revisa controles/NaNs/valores <=0.")

    # 4) Entrenar XGB
    for col in ["ln_CO2", "ln_GDP"]:
        if col not in X_full.columns:
            raise RuntimeError(f"[FATAL] Falta columna {col} en X_full.")
    bst, features = fit_xgb(X_full, controls=controls)
    logger.info("Modelo XGB entrenado. n_features=%s features=%s", len(features), features)

    # 5) Proyección base (CAGR) y features futuras BASE
    fut_base = extended_data(df, years=years)
    logger.info("Futuro baseline filas: %s", fut_base.shape)
    if fut_base.empty:
        raise RuntimeError("[FATAL] No se pudo extender el panel; verifica que cada país tenga ≥2 años válidos.")

    fut_base = clean_mix(fut_base)

    X_base, _ = build_features(
        pd.concat([df, fut_base], ignore_index=True),
        target="EN.GHG.CO2.MT.CE.AR5",
        gdp="NY.GDP.MKTP.CD",
        controls=controls
    )
    max_hist = int(df["year"].max())
    mask_base = X_base["year"] > max_hist
    logger.info("Filas futuras en X_base: %s", int(mask_base.sum()))

    # 6) Escenario de inversión y features futuras SCENARIO
    inc_mix = increase_share(df, invest_target, inc_pp_total=renew_pp_total, years=years)
    fut_scen = fut_base.merge(
        inc_mix[["iso3c", "year"] + MIX_COLS],
        on=["iso3c", "year"],
        how="left",
        suffixes=("", "_inc")
    )

    for c in MIX_COLS:
        if f"{c}_inc" in fut_scen.columns:
            fut_scen[c] = fut_scen[f"{c}_inc"].fillna(fut_scen.get(c, 0.0))
    fut_scen = fut_scen.drop(columns=[c for c in fut_scen.columns if c.endswith("_inc")])
    fut_scen = clean_mix(fut_scen)

    X_scen, _ = build_features(
        pd.concat([df, fut_scen], ignore_index=True),
        target="EN.GHG.CO2.MT.CE.AR5",
        gdp="NY.GDP.MKTP.CD",
        controls=controls
    )
    mask_scen = X_scen["year"] > max_hist
    logger.info("Filas futuras en X_scen: %s", int(mask_scen.sum()))

    # 7) Predicciones futuras base vs escenario
    Xb = X_base.loc[mask_base, features]
    Xs = X_scen.loc[mask_scen, features]
    logger.info("Xb.shape=%s Xs.shape=%s", Xb.shape, Xs.shape)
    if Xb.empty or Xs.empty:
        raise RuntimeError("[FATAL] Matrices futuras vacías (baseline o escenario). Revisa proyección/mezcla/merge.")

    yb = bst.predict(xgb.DMatrix(Xb))
    ys = bst.predict(xgb.DMatrix(Xs))

    fut_id_base = X_base.loc[mask_base, ["iso3c", "Country", "year"]].reset_index(drop=True)
    fut_id_scen = X_scen.loc[mask_scen, ["iso3c", "Country", "year"]].reset_index(drop=True)

    base_df = fut_id_base.copy()
    base_df["CO2_pred"] = np.exp(yb)

    scen_df = fut_id_scen.copy()
    scen_df["CO2_pred"] = np.exp(ys)

    # 8) Agregado a 5 años por país
    agg_base = base_df.groupby("iso3c").agg(CO2_base_5y=("CO2_pred", "mean")).reset_index()
    agg_scen = scen_df.groupby("iso3c").agg(CO2_scen_5y=("CO2_pred", "mean")).reset_index()
    res = agg_base.merge(agg_scen, on="iso3c", how="inner")

    last_snap = (df.sort_values("year").groupby("iso3c").tail(1))[["iso3c", "Country", "region"] + MIX_COLS].copy()
    last_snap = clean_mix(last_snap)
    last_snap["grid_g_per_kwh"] = compute_grid_intensity(last_snap)
    res = res.merge(last_snap, on="iso3c", how="left")

    res["reduction_pct_5y"] = (res["CO2_scen_5y"] / res["CO2_base_5y"] - 1.0) * 100.0
    res["prob_reduce_5y"] = calibrated_probability(res["reduction_pct_5y"])

    # 9) Priorización (sensibilidad +5 pp hidro vs +5 pp no-hidro)
    prio_rows = []
    for tech_col in ["EG.ELC.HYRO.ZS", "EG.ELC.RNEW.ZS"]:
        fut_alt_mix = increase_share(df, tech_col, inc_pp_total=5.0, years=years)
        fut_alt = fut_base.merge(
            fut_alt_mix[["iso3c", "year"] + MIX_COLS],
            on=["iso3c", "year"], how="left", suffixes=("", "_inc")
        )
        for c in MIX_COLS:
            if f"{c}_inc" in fut_alt.columns:
                fut_alt[c] = fut_alt[f"{c}_inc"].fillna(fut_alt.get(c, 0.0))
        fut_alt = fut_alt.drop(columns=[c for c in fut_alt.columns if c.endswith("_inc")])
        fut_alt = clean_mix(fut_alt)

        X_alt, _ = build_features(pd.concat([df, fut_alt], ignore_index=True),
                                  target="EN.GHG.CO2.MT.CE.AR5",
                                  gdp="NY.GDP.MKTP.CD",
                                  controls=controls)
        mask_alt = X_alt["year"] > max_hist
        if mask_alt.sum() == 0:
            continue
        ya = bst.predict(xgb.DMatrix(X_alt.loc[mask_alt, features]))
        alt_df = X_alt.loc[mask_alt, ["iso3c", "Country", "year"]].copy()
        alt_df["CO2_pred"] = np.exp(ya)
        agg_alt = alt_df.groupby("iso3c").agg(CO2_alt_5y=("CO2_pred", "mean")).reset_index()
        prio_rows.append((tech_col, agg_alt))

    prio = res[["iso3c", "Country"]].copy()
    for tech_col, agg_alt in prio_rows:
        prio = prio.merge(agg_alt.rename(columns={"CO2_alt_5y": f"{tech_col}_5y"}), on="iso3c", how="left")
    prio = prio.merge(base_df.groupby("iso3c").agg(CO2_base_5y=("CO2_pred", "mean")).reset_index(),
                      on="iso3c", how="left")

    for tech_col in ["EG.ELC.HYRO.ZS", "EG.ELC.RNEW.ZS"]:
        if f"{tech_col}_5y" in prio.columns:
            prio[f"delta_{tech_col}"] = (prio[f"{tech_col}_5y"] / prio["CO2_base_5y"] - 1.0) * 100.0
        else:
            prio[f"delta_{tech_col}"] = np.nan

    def pick(row):
        d_h = row.get("delta_EG.ELC.HYRO.ZS", np.nan)
        d_r = row.get("delta_EG.ELC.RNEW.ZS", np.nan)
        if np.isnan(d_h) and np.isnan(d_r):
            return "insufficient_data"
        return "hydro" if (not np.isnan(d_h) and (np.isnan(d_r) or d_h < d_r)) else "non_hydro_renewables"

    prio["priority_tech"] = prio.apply(pick, axis=1)

    # 10) Guardar salidas
    out1 = os.path.join(output_dir, "renew_5y_country_results.csv")
    out2 = os.path.join(output_dir, "renew_5y_priorities.csv")
    out3 = os.path.join(output_dir, "renew_5y_global_summary.csv")

    cols_basic = ["iso3c", "Country", "region", "grid_g_per_kwh",
                  "CO2_base_5y", "CO2_scen_5y", "reduction_pct_5y", "prob_reduce_5y"]
    res_out = res[cols_basic].copy()
    res_out.to_csv(out1, index=False)

    prio_cols = ["iso3c", "Country", "CO2_base_5y",
                 "delta_EG.ELC.HYRO.ZS", "delta_EG.ELC.RNEW.ZS", "priority_tech"]
    prio_out = prio[prio_cols].copy()
    prio_out.to_csv(out2, index=False)

    global_summary = pd.DataFrame({
        "countries_count": [res.shape[0]],
        "avg_reduction_pct_5y": [float(np.nanmean(res["reduction_pct_5y"]))],
        "share_with_prob>0.7": [float(np.mean(res["prob_reduce_5y"] > 0.7))],
        "invest_target": [invest_target],
        "renew_pp_total": [renew_pp_total],
        "years": [years],
    })
    global_summary.to_csv(out3, index=False)

    logger.info("Outputs saved: %s | %s | %s", out1, out2, out3)

    return res_out, prio_out, global_summary


def parse_args():
    p = argparse.ArgumentParser(description="5-year renewables investment scenario + prioritization")
    p.add_argument("--input_file", default="data/processed/wide_clean.csv", help="Panel CSV")
    p.add_argument("--output_dir", default="model_results/renew", help="Carpeta de salida")
    p.add_argument("--years", type=int, default=5, help="Horizonte de proyección (años)")
    p.add_argument("--renew_pp_total", type=float, default=10.0,
                   help="Aumento acumulado en p.p. de la serie objetivo en 5 años")
    p.add_argument("--invest_target", choices=["EG.ELC.RNEW.ZS", "EG.ELC.HYRO.ZS"],
                   default="EG.ELC.RNEW.ZS", help="Dónde enfocar la inversión principal")
    p.add_argument("--controls", nargs="*", default=[
        "SP.POP.TOTL", "SP.URB.TOTL.IN.ZS",
        "EG.FEC.RNEW.ZS",
        "EG.ELC.COAL.ZS", "EG.ELC.NGAS.ZS", "EG.ELC.PETR.ZS",
        "EG.ELC.NUCL.ZS", "EG.ELC.HYRO.ZS", "EG.ELC.RNEW.ZS"
    ], help="Controles por defecto (ajusta según cobertura)")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    run_scenario(
        panel_csv=args.input_file,
        output_dir=args.output_dir,
        years=args.years,
        renew_pp_total=args.renew_pp_total,
        invest_target=args.invest_target,
        controls=args.controls
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
