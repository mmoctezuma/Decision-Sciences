# -*- coding: utf-8 -*-

"""
EV50 Scenario – Fermi-style estimate of CO2 impact if 50% of the population adopts EVs.

Inputs
------
- wide_clean.csv: panel con CO2 total (y de transporte si existe), población y mezcla eléctrica.
- ev_adoption_v2.csv: flota total, adopción EV actual por país.

Outputs
-------
- ev50_results.csv                : resultados sin tope (raw)
- ev50_results_capped.csv         : resultados con tope (usa CO2 transporte o % de CO2 total)
- ev50_sensitivity_capped.csv     : sensibilidad global (grid ICE g/km x km/año), con tope
- ev50_top_abs.csv                : top por reducción absoluta (MtCO2), con tope
- ev50_top_pct.csv                : top por % del CO2 total, con tope
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def setup_logger(level: str = "INFO") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(levelname)s:%(name)s:%(message)s"
    )


def norm_country(s: str) -> str:
    return str(s).strip().lower()


def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def load_inputs(wide_csv: Path, ev_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    wide = pd.read_csv(wide_csv)
    ev = pd.read_csv(ev_csv)
    return wide, ev


def pick_latest_snapshot(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Selecciona, por país, el último año con CO2 total y mezcla eléctrica no nulos.
    """
    req_mix = ["EG.ELC.COAL.ZS","EG.ELC.NGAS.ZS","EG.ELC.PETR.ZS","EG.ELC.HYRO.ZS","EG.ELC.RNEW.ZS"]
    cols_needed = ["iso3c","Country","region","year","SP.POP.TOTL","EN.GHG.CO2.MT.CE.AR5","EN.GHG.CO2.TR.MT.CE.AR5"] + req_mix
    missing = [c for c in cols_needed if c not in wide.columns]
    if missing:
        raise ValueError(f"Faltan columnas en wide_clean.csv: {missing}")

    df = (
        wide.sort_values("year")
            .dropna(subset=["EN.GHG.CO2.MT.CE.AR5","SP.POP.TOTL"])
            .groupby("iso3c", as_index=False, group_keys=False)
            .apply(lambda g: g.dropna(subset=req_mix).tail(1))
            .reset_index(drop=True)
    )

    df = df.rename(columns={
        "EN.GHG.CO2.MT.CE.AR5": "co2_total_mt",
        "EN.GHG.CO2.TR.MT.CE.AR5": "co2_transport_mt",
        "SP.POP.TOTL": "population",
        "EG.ELC.COAL.ZS": "share_coal",
        "EG.ELC.NGAS.ZS": "share_gas",
        "EG.ELC.PETR.ZS": "share_oil",
        "EG.ELC.NUCL.ZS": "share_nuclear",
        "EG.ELC.HYRO.ZS": "share_hydro",
        "EG.ELC.RNEW.ZS": "share_renew",
    })
    return df[[
        "iso3c","Country","region","year","population","co2_total_mt","co2_transport_mt",
        "share_coal","share_gas","share_oil","share_nuclear","share_hydro","share_renew"
    ]].copy()


def merge_ev(snapshot: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """
    Une por nombre de país normalizado.
    Se esperan en ev: 'country','Total motor vehicles','EV_adoption'
    """
    expected = ["country","Total motor vehicles","EV_adoption"]
    miss = [c for c in expected if c not in ev.columns]
    if miss:
        raise ValueError(f"Faltan columnas en ev_adoption_v2.csv: {miss}")

    snap = snapshot.copy()
    snap["country_key"] = snap["Country"].apply(norm_country)

    ev2 = ev.copy()
    ev2["country_key"] = ev2["country"].apply(norm_country)

    merged = snap.merge(
        ev2[["country_key","Total motor vehicles","EV_adoption"]],
        on="country_key",
        how="left"
    )
    merged = merged.drop(columns=["country_key"])
    return merged


def fix_mix_shares(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte porcentajes >1 a fracciones (divide entre 100).
    """
    mix_cols = ["share_coal","share_gas","share_oil","share_nuclear","share_hydro","share_renew"]
    for c in mix_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = np.where(df[c] > 1.0, df[c] / 100.0, df[c])
        df[c] = df[c].clip(lower=0)
    return df


def compute_grid_intensity(df: pd.DataFrame, ef: Dict[str, float]) -> pd.Series:
    """
    gCO2/kWh a partir de la mezcla eléctrica (fracciones) y factores por tecnología.
    """
    grid = (
        df["share_coal"]    * ef["coal"] +
        df["share_gas"]     * ef["gas"] +
        df["share_oil"]     * ef["oil"] +
        df["share_nuclear"] * ef["nuclear"] +
        df["share_hydro"]   * ef["hydro"] +
        df["share_renew"]   * ef["renew"]
    )
    return grid


def run_scenario(
    merged: pd.DataFrame,
    target_share: float,
    ice_g_per_km: float,
    ev_kwh_per_km: float,
    km_per_vehicle: int,
    transport_cap_pct: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calcula resultados raw y con tope, y la sensibilidad global con tope.
    """

    df = merged.copy()

    # Defaults/limpieza
    df["EV_adoption"] = pd.to_numeric(df["EV_adoption"], errors="coerce").fillna(0).clip(lower=0)
    df["Total motor vehicles"] = pd.to_numeric(df["Total motor vehicles"], errors="coerce").fillna(0).clip(lower=0)
    df["population"] = pd.to_numeric(df["population"], errors="coerce")

    # Factores de emisión eléctricos por tecnología (gCO2/kWh)
    EF = {"coal": 900, "gas": 469, "oil": 700.0, "nuclear": 16, "hydro": 10, "renew": 50}
    df["grid_g_per_kwh"] = compute_grid_intensity(df, EF)

    # g/km para EV
    df["ev_g_per_km"] = df["grid_g_per_kwh"] * ev_kwh_per_km

    # Delta de adopción para llegar al target
    df["delta_adoption"] = np.maximum(0.0, target_share - df["EV_adoption"])

    # Kilometraje anual total y desplazado de ICE a EV
    df["annual_km_total"] = df["Total motor vehicles"] * float(km_per_vehicle)
    df["km_shifted"] = df["annual_km_total"] * df["delta_adoption"]

    # Evitado (ton) = km_shifted * (ICE_gpkm - EV_gpkm)/1000
    delta_gpkm = np.maximum(0.0, ice_g_per_km - df["ev_g_per_km"])
    df["avoided_tonnes"] = df["km_shifted"] * delta_gpkm / 1000.0
    df["avoided_Mt"] = df["avoided_tonnes"] / 1e6

    res_raw_cols = [
        "iso3c","Country","region","year","population",
        "Total motor vehicles","EV_adoption","grid_g_per_kwh","ev_g_per_km",
        "co2_total_mt","co2_transport_mt",
        "delta_adoption","annual_km_total","km_shifted","avoided_Mt"
    ]
    results_raw = df[res_raw_cols].copy().sort_values("avoided_Mt", ascending=False)

    cap_series = df["co2_transport_mt"].copy()
    fallback = pd.to_numeric(df["co2_total_mt"], errors="coerce") * float(transport_cap_pct)
    cap_series = np.where(pd.notnull(cap_series) & (cap_series > 0), cap_series, fallback)
    cap_series = np.nan_to_num(cap_series, nan=0.0)

    df["transport_cap_mt"] = cap_series
    df["avoided_Mt_capped"] = np.minimum(df["avoided_Mt"], df["transport_cap_mt"])
    df["pct_total_co2_reduced"] = np.where(
        (df["co2_total_mt"] > 0),
        df["avoided_Mt_capped"] / df["co2_total_mt"] * 100.0,
        np.nan
    )

    res_cap_cols = res_raw_cols + ["transport_cap_mt","avoided_Mt_capped","pct_total_co2_reduced"]
    results_cap = df[res_cap_cols].copy().sort_values("avoided_Mt_capped", ascending=False)

    km_opts = [10000, 12000, 15000]
    ice_opts = [150.0, 180.0, 200.0]
    sens_rows: List[Dict] = []
    for km in km_opts:
        for ice in ice_opts:
            km_shift = df["Total motor vehicles"] * float(km) * df["delta_adoption"]
            delta_g = np.maximum(0.0, float(ice) - df["ev_g_per_km"])
            avoided_tonnes = km_shift * delta_g / 1000.0
            avoided_Mt = avoided_tonnes / 1e6
            avoided_capped = np.minimum(avoided_Mt, df["transport_cap_mt"])
            sens_rows.append({
                "KM_PER_VEHICLE": km,
                "ICE_G_PER_KM": ice,
                "Global_Reduction_MtCO2": float(np.nansum(avoided_capped))
            })
    sens_df = pd.DataFrame(sens_rows).sort_values(["KM_PER_VEHICLE","ICE_G_PER_KM"])

    return results_raw, results_cap, sens_df


def save_outputs(
    outdir: Path,
    results_raw: pd.DataFrame,
    results_cap: pd.DataFrame,
    sens_df: pd.DataFrame,
    top_n: int,
    min_total_co2_for_pct: float
) -> None:
    f_raw = outdir / "ev50_results.csv"
    f_cap = outdir / "ev50_results_capped.csv"
    f_sens = outdir / "ev50_sensitivity_capped.csv"
    results_raw.to_csv(f_raw, index=False)
    results_cap.to_csv(f_cap, index=False)
    sens_df.to_csv(f_sens, index=False)

    # Top lists (con tope)
    top_abs = results_cap[[
        "iso3c","Country","avoided_Mt_capped","transport_cap_mt","co2_total_mt","pct_total_co2_reduced"
    ]].head(top_n)
    f_top_abs = outdir / "ev50_top_abs.csv"
    top_abs.to_csv(f_top_abs, index=False)

    mask = pd.to_numeric(results_cap["co2_total_mt"], errors="coerce") > float(min_total_co2_for_pct)
    top_pct = (results_cap[mask]
               .sort_values("pct_total_co2_reduced", ascending=False)
               [["iso3c","Country","pct_total_co2_reduced","avoided_Mt_capped","co2_total_mt"]]
               .head(top_n))
    f_top_pct = outdir / "ev50_top_pct.csv"
    top_pct.to_csv(f_top_pct, index=False)

    logging.info(f"Guardado: {f_raw}")
    logging.info(f"Guardado: {f_cap}")
    logging.info(f"Guardado: {f_sens}")
    logging.info(f"Guardado: {f_top_abs}")
    logging.info(f"Guardado: {f_top_pct}")

    # Resumen global
    global_mt = float(np.nansum(results_cap["avoided_Mt_capped"]))
    logging.info(f"EV50 – reducción global (con tope): {global_mt:,.2f} MtCO₂/año")


def main():
    parser = argparse.ArgumentParser(description="EV50 Scenario – 50% EV adoption impact on CO2")
    parser.add_argument("--wide_csv", required=True, type=Path, help="Ruta a wide_clean.csv")
    parser.add_argument("--ev_csv", required=True, type=Path, help="Ruta a ev_adoption_v2.csv")
    parser.add_argument("--outdir", required=True, type=Path, help="Directorio de salida")

    # Supuestos
    parser.add_argument("--target_share", type=float, default=0.50, help="Objetivo de adopción EV (0-1)")
    parser.add_argument("--ice_g_per_km", type=float, default=180.0, help="gCO2/km promedio ICE")
    parser.add_argument("--ev_kwh_per_km", type=float, default=0.18, help="kWh/km promedio EV")
    parser.add_argument("--km_per_vehicle", type=int, default=12000, help="km/año por vehículo")
    parser.add_argument("--transport_cap_pct", type=float, default=0.20,
                        help="Si falta CO2 transporte, usar este % de CO2 total como tope (0-1)")

    # Reporteo
    parser.add_argument("--top_n", type=int, default=15, help="Top N para tablas de resumen")
    parser.add_argument("--min_total_co2_for_pct", type=float, default=50.0,
                        help="Mínimo MtCO2 total para incluir en ranking por porcentaje")
    parser.add_argument("--log_level", type=str, default="INFO", help="Nivel de logging (DEBUG|INFO|WARNING|ERROR)")

    args = parser.parse_args()
    setup_logger(args.log_level)

    ensure_outdir(args.outdir)
    logging.info("Cargando insumos...")
    wide, ev = load_inputs(args.wide_csv, args.ev_csv)

    logging.info("Construyendo snapshot más reciente por país...")
    snap = pick_latest_snapshot(wide)
    snap = fix_mix_shares(snap)

    logging.info("Uniendo con EV adoption y flota...")
    merged = merge_ev(snap, ev)

    logging.info("Corriendo escenario EV50...")
    res_raw, res_cap, sens_df = run_scenario(
        merged=merged,
        target_share=args.target_share,
        ice_g_per_km=args.ice_g_per_km,
        ev_kwh_per_km=args.ev_kwh_per_km,
        km_per_vehicle=args.km_per_vehicle,
        transport_cap_pct=args.transport_cap_pct
    )

    logging.info("Guardando resultados...")
    save_outputs(
        outdir=args.outdir,
        results_raw=res_raw,
        results_cap=res_cap,
        sens_df=sens_df,
        top_n=args.top_n,
        min_total_co2_for_pct=args.min_total_co2_for_pct
    )

    logging.info("Listo.")


if __name__ == "__main__":
    main()

