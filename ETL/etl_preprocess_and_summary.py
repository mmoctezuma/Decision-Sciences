# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize


# =========================
#  Utils
# =========================
def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def _detect_year_cols(cols):
    return [c for c in cols if isinstance(c, str) and c.startswith("YR") and c[2:].isdigit()]


def _normalize_column_names(df):
    rename_map = {}
    for k in ["iso3c", "series", "Series", "Country"]:
        if k not in df.columns:
            for c in df.columns:
                if c.lower() == k.lower():
                    rename_map[c] = k
                    break
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


# =========================
#  Load & reshape
# =========================
def load_and_reshape_from_YR(input_file: str) -> pd.DataFrame:
    """
    Lee CSV con columnas YR#### y devuelve panel wide:
      (iso3c, Country, year, <series-code columns...>)
    """
    dfw = pd.read_csv(input_file)
    dfw = _normalize_column_names(dfw)

    required = ["iso3c", "series", "Series", "Country", "region"]
    missing_req = [c for c in required if c not in dfw.columns]
    if missing_req:
        raise ValueError(f"Faltan columnas requeridas: {missing_req}")

    year_cols = _detect_year_cols(dfw.columns)
    if not year_cols:
        raise ValueError("No se detectaron columnas de año tipo 'YR1960', 'YR1961', ...")

    # Wide -> Long por años
    dfl = dfw.melt(
        id_vars=["iso3c", "Country", "region", "series", "Series"],
        value_vars=year_cols,
        var_name="year_str",
        value_name="value"
    )
    dfl["year"] = dfl["year_str"].str[2:].astype(int)
    dfl = dfl.drop(columns=["year_str"]).sort_values(["iso3c", "series", "year"], ignore_index=True)

    # Long -> Wide por series (códigos)
    wide = (
        dfl.pivot_table(
            index=["iso3c", "Country", "region", "year"],
            columns="series",
            values="value",
            aggfunc="mean"
        )
        .sort_index()
        .reset_index()
    )

    # Cast numérico
    id_cols = ["iso3c", "Country", "region", "year"]
    for c in wide.columns:
        if c not in id_cols:
            wide[c] = pd.to_numeric(wide[c], errors="coerce")

    # Mapa código->nombre de serie
    series_name_map = (
        dfl[["series", "Series"]]
        .drop_duplicates()
        .set_index("series")["Series"]
        .to_dict()
    )

    return wide, series_name_map, dfl


# =========================
#  Heurística de política por serie
# =========================
SERIES_LEVEL_GROWTH = {
    "EN.GHG.CO2.MT.CE.AR5", "EN.GHG.CO2.LU.MT.CE.AR5",
    "NY.GDP.MKTP.CD", "EN.ATM.CO2E.KT", "SP.POP.TOTL",
    "EN.ATM.METH.KT.CE", "EN.ATM.NOXE.KT.CE"
}
SERIES_RATE_0_100 = {
    "SP.URB.TOTL.IN.ZS", "EG.FEC.RNEW.ZS",
    "EG.ELC.COAL.ZS", "EG.ELC.NGAS.ZS", "EG.ELC.PETR.ZS",
    "EG.ELC.HYRO.ZS", "EG.ELC.RNEW.ZS",
    "IT.NET.USER.ZS", "EN.CO2.TRAN.ZS",
}


def classify_series(code: str) -> dict:
    if code in SERIES_LEVEL_GROWTH:
        return {"series": code, "sclass": "level_growth", "interp": "log", "bounds_lo": None, "bounds_hi": None}
    if code in SERIES_RATE_0_100:
        return {"series": code, "sclass": "share_rate", "interp": "linear", "bounds_lo": 0, "bounds_hi": 100}
    return {"series": code, "sclass": "other", "interp": "linear", "bounds_lo": None, "bounds_hi": None}


# =========================
#  Summarize BEFORE preprocess
# =========================
def summarize_before_preprocess(wide: pd.DataFrame, outdir: str) -> pd.DataFrame:
    _ensure_dir(outdir)
    id_cols = ["iso3c", "Country", "year", "region"]
    series_cols = [c for c in wide.columns if c not in id_cols]

    cov_rows = []
    for s in series_cols:
        col = wide[s]
        coverage = col.notna().mean()
        years_with = wide.loc[col.notna(), "year"]
        first_y = int(years_with.min()) if not years_with.empty else np.nan
        last_y  = int(years_with.max()) if not years_with.empty else np.nan
        cov_rows.append({"series": s, "coverage": coverage, "first_year": first_y, "last_year": last_y})
    series_coverage = pd.DataFrame(cov_rows).sort_values("coverage", ascending=False)
    series_coverage.to_csv(f"{outdir}/series_coverage.csv", index=False)

    gap_rows = []
    g = wide.sort_values(["iso3c", "year"])
    for s in series_cols:
        for iso, sub in g[["iso3c", "year", s]].groupby("iso3c"):
            mask = sub[s].notna().to_numpy()
            years = sub["year"].to_numpy()
            if mask.sum() < 2:
                n_internal_gaps, max_gap = 0, np.nan
            else:
                idx = np.where(mask)[0]
                diffs = np.diff(years[idx])
                internal_gaps = diffs[diffs > 1] - 1
                n_internal_gaps = len(internal_gaps)
                max_gap = int(internal_gaps.max()) if internal_gaps.size else 0
            gap_rows.append({"series": s, "iso3c": iso, "n_internal_gaps": n_internal_gaps, "max_internal_gap": max_gap})
    pd.DataFrame(gap_rows).to_csv(f"{outdir}/country_series_gaps.csv", index=False)

    stats_rows = []
    for s in series_cols:
        vals = pd.to_numeric(wide[s], errors="coerce")
        stats_rows.append({
            "series": s,
            "min": float(np.nanmin(vals)) if vals.notna().any() else np.nan,
            "p01": float(np.nanpercentile(vals, 1)) if vals.notna().any() else np.nan,
            "p50": float(np.nanpercentile(vals, 50)) if vals.notna().any() else np.nan,
            "p99": float(np.nanpercentile(vals, 99)) if vals.notna().any() else np.nan,
            "max": float(np.nanmax(vals)) if vals.notna().any() else np.nan,
            "missing_pct": float(vals.isna().mean())
        })
    pd.DataFrame(stats_rows).to_csv(f"{outdir}/stats_preview.csv", index=False)

    # Recomendaciones por serie (editable por el usuario si quiere)
    recs = [classify_series(s) for s in series_cols]
    recs_df = pd.DataFrame(recs)
    recs_df.to_csv(f"{outdir}/recommendations.csv", index=False)
    return recs_df


# =========================
#  Interpoladores
# =========================
def _interp_no_extrap(series: pd.Series, kind="linear"):
    """Interpola solo entre el primer y último dato (no extrapola bordes). kind: 'linear' o 'log'."""
    x = series.index.to_numpy()
    y = series.to_numpy(dtype=float)
    m = np.isfinite(y)
    if m.sum() < 2:
        return series
    xk, yk = x[m], y[m]

    if kind == "log":
        pos = yk > 0
        if pos.sum() < 2:
            return series
        xk2, yk2 = xk[pos], np.log(yk[pos])
        xmin, xmax = xk2.min(), xk2.max()
        xp = x[(x >= xmin) & (x <= xmax)]
        yp = np.interp(xp, xk2, yk2)
        out = series.copy().to_numpy(dtype=float)
        out[(x >= xmin) & (x <= xmax)] = np.exp(yp)
        return pd.Series(out, index=series.index)

    xmin, xmax = xk.min(), xk.max()
    xp = x[(x >= xmin) & (x <= xmax)]
    yp = np.interp(xp, xk, yk)
    out = series.copy().to_numpy(dtype=float)
    out[(x >= xmin) & (x <= xmax)] = yp
    return pd.Series(out, index=series.index)


def compute_cagr(series):
    valid = series.dropna()
    if len(valid) < 2:
        return None
    start, end = valid.iloc[0], valid.iloc[-1]
    years = len(valid) - 1
    if start > 0 and end > 0 and years > 0:
        return (end/start)**(1/years) - 1 
    else:
        return np.nan


def extrap_series(series, years=10):
    g = compute_cagr(series)
    if g is None:
        return pd.Series([np.nan]*years, 
                         index=range(series.index[-1]+1, series.index[-1]+years+1))
    
    projections = [series.iloc[-1] * ((1+g)**i) for i in range(1, years+1)]
    return pd.Series(projections, 
                     index=range(series.index[-1]+1, series.index[-1]+years+1))


def _clamp(s: pd.Series, lo, hi):
    if lo is None and hi is None:
        return s
    return s.clip(lower=lo if lo is not None else -np.inf,
                  upper=hi if hi is not None else  np.inf)

# =========================
#  Preprocess with policy
# =========================
def preprocess_with_policy(
    wide: pd.DataFrame,
    start: int = 1960,
    end: int = 2022,
    winsorize_percap_vars: bool = True,
    winsor_limits=(0.01, 0.01),
    recommendations_csv: str | None = None
) -> pd.DataFrame:

    df = wide[(wide["year"] >= start) & (wide["year"] <= end)].copy()
    df = df.sort_values(["iso3c", "year"])
    id_cols = ["iso3c", "Country", "year"]
    series_cols = [c for c in df.columns if c not in id_cols]

    # Cargar/crear política por serie
    if recommendations_csv and os.path.exists(recommendations_csv):
        recs_df = pd.read_csv(recommendations_csv)
        recs_df = recs_df.set_index("series").to_dict(orient="index")
        def rule_for(s):
            r = recs_df.get(s, None)
            if r is None:
                return classify_series(s)
            return {
                "series": s,
                "sclass": r.get("sclass", "other"),
                "interp": r.get("interp", "linear"),
                "bounds_lo": r.get("bounds_lo", None),
                "bounds_hi": r.get("bounds_hi", None),
            }
    else:
        def rule_for(s):
            return classify_series(s)

    # Aplicar política por país
    for s in series_cols:
        rule = rule_for(s)
        interp_kind = rule["interp"]
        bounds_lo, bounds_hi = rule["bounds_lo"], rule["bounds_hi"]

        if interp_kind != "none":
            def _apply_group(g):
                ser = pd.to_numeric(g[s], errors="coerce")
                tmp = ser.groupby(g["year"]).mean()
                tmp_interp = _interp_no_extrap(tmp, kind=interp_kind)
                
                if tmp_interp.notna().any():
                    last_valid = tmp_interp.last_valid_index()
                    if last_valid is not None and last_valid < tmp_interp.index.max():
                        missing_years = tmp_interp.index[(tmp_interp.index > last_valid) & (tmp_interp.isna())]
                        if len(missing_years) > 0:
                            proj = extrap_series(tmp_interp.loc[:last_valid], years=len(missing_years))
                            for y in missing_years:
                                if y in proj.index:
                                    tmp_interp.loc[y] = proj.loc[y]

                out = tmp_interp.reindex(g["year"]).to_numpy()
                return pd.Series(out, index=g.index, dtype=float)
            
            df[s] = df.groupby("iso3c", group_keys=False).apply(_apply_group)

        # Clamp 0–100 si aplica
        if bounds_lo is not None or bounds_hi is not None:
            df[s] = (
                df.groupby("iso3c", group_keys=False)[s]
                  .apply(lambda x: _clamp(x, bounds_lo, bounds_hi))
            )

    # Winsorizar SOLO per-cápita / tasas / other_linear (no niveles agregados)
    if winsorize_percap_vars:
        for s in series_cols:
            rule = rule_for(s)
            if rule["sclass"] in ("share_rate", "other"):
                arr = pd.to_numeric(df[s], errors="coerce").to_numpy(dtype=float)
                if np.isfinite(arr).sum() > 0:
                    df[s] = winsorize(arr, limits=winsor_limits)

    return df


# =========================
#  Post-summaries (opcionales)
# =========================
def _max_consecutive_valid(years_sorted, mask_sorted):
    # Devuelve la racha máxima consecutiva de años con dato (True) dentro del vector ordenado por año
    max_run = run = 0
    prev_year = None
    for y, ok in zip(years_sorted, mask_sorted):
        if not ok:
            run = 0
            prev_year = y
            continue
        if prev_year is None or y == prev_year + 1:
            run += 1
        else:
            run = 1
        max_run = max(max_run, run)
        prev_year = y
    return max_run


def compute_country_coverage_by_series(wide: pd.DataFrame, outdir: str, min_years: int = 11):
    """
    Calcula cobertura por serie a nivel país:
      - países con >=1 dato
      - países con >=min_years datos (no consecutivos)
      - países con >=min_years consecutivos (para ventanas de 10 años del classifier)
    Guarda CSV en summary_first/country_coverage_by_series.csv
    """
    os.makedirs(outdir, exist_ok=True)
    id_cols = ["iso3c", "Country", "year", "region"]
    series_cols = [c for c in wide.columns if c not in id_cols]
    n_countries_total = wide["iso3c"].nunique()

    g = wide.sort_values(["iso3c", "year"])
    rows = []
    for s in series_cols:
        any_cnt = 0
        ge_minyrs_cnt = 0
        ge_minyrs_consec_cnt = 0

        for iso, sub in g[["iso3c", "year", s]].groupby("iso3c"):
            vals = pd.to_numeric(sub[s], errors="coerce")
            mask = vals.notna().to_numpy()
            years = sub["year"].to_numpy()

            any_has = mask.any()
            if any_has:
                any_cnt += 1

            ge_minyrs = (mask.sum() >= min_years)
            if ge_minyrs:
                ge_minyrs_cnt += 1

            # consecutivos
            max_run = _max_consecutive_valid(years, mask)
            if max_run >= min_years:
                ge_minyrs_consec_cnt += 1

        rows.append({
            "series": s,
            "n_countries_total": n_countries_total,
            "n_countries_any": any_cnt,
            "pct_countries_any": 100.0 * any_cnt / n_countries_total if n_countries_total else 0.0,
            "n_countries_ge_minyrs": ge_minyrs_cnt,
            "pct_countries_ge_minyrs": 100.0 * ge_minyrs_cnt / n_countries_total if n_countries_total else 0.0,
            "n_countries_ge_minyrs_consec": ge_minyrs_consec_cnt,
            "pct_countries_ge_minyrs_consec": 100.0 * ge_minyrs_consec_cnt / n_countries_total if n_countries_total else 0.0,
            "min_years": min_years
        })

    out = pd.DataFrame(rows).sort_values(
        ["pct_countries_ge_minyrs_consec", "pct_countries_ge_minyrs", "pct_countries_any"],
        ascending=False
    )
    out.to_csv(os.path.join(outdir, "country_coverage_by_series.csv"), index=False)
    return out



def summarize_after_preprocess(wide_clean: pd.DataFrame, outdir: str, focus_year: int = 2020):
    _ensure_dir(outdir)
    id_cols = ["iso3c", "Country", "year", "region"]
    feat_cols = [c for c in wide_clean.columns if c not in id_cols]

    # Estadísticas descriptivas
    wide_clean[feat_cols].describe().T.to_csv(f"{outdir}/summary_statistics.csv")

    # Correlaciones
    wide_clean[feat_cols].corr().to_csv(f"{outdir}/correlations.csv")

    # Top emisores por CO2 total si existe
    if "EN.ATM.CO2E.KT" in feat_cols:
        (wide_clean[wide_clean["year"] == focus_year]
         .nlargest(10, "EN.ATM.CO2E.KT")[["Country", "EN.ATM.CO2E.KT"]]
        ).to_csv(f"{outdir}/top_emitters_{focus_year}.csv", index=False)

    # Cambio CO2 pc 2000–2020 si existe
    if "EN.ATM.CO2E.PC" in feat_cols:
        co2_change = (
            wide_clean[wide_clean["year"].between(2000, 2020)]
            .groupby("iso3c")["EN.ATM.CO2E.PC"]
            .agg(first="first", last="last")
            .assign(change_pct=lambda x: (x["last"] - x["first"]) / x["first"] * 100)
            .sort_values("change_pct")
        )
        co2_change.to_csv(f"{outdir}/co2_change_2000_2020.csv")


# =========================
#  Orchestrator / CLI
# =========================
def main():
    parser = argparse.ArgumentParser(description="ETL + Summarize + Preprocess (WDI CSV con columnas YR####).")
    parser.add_argument("--input", required=True, help="CSV con columnas YR####, series, Series, iso3c, Country.")
    parser.add_argument("--workdir", default="data/processed", help="Directorio de trabajo/salidas.")
    parser.add_argument("--start", type=int, default=1960, help="Año inicial (filtro).")
    parser.add_argument("--end", type=int, default=2022, help="Año final (filtro).")
    parser.add_argument("--focus_year", type=int, default=2020, help="Año para rankings/cortes.")
    parser.add_argument("--winsor", action="store_true", help="Activar winsorización en per-cápita/tasas.")
    parser.add_argument("--winsor_limits", type=float, nargs=2, default=(0.01, 0.01), help="Límites winsor (low high).")
    parser.add_argument("--policy_csv", type=str, default=None, help="CSV de recomendaciones para sobreescribir la política.")
    parser.add_argument("--stage", choices=["summarize", "preprocess", "all"], default="all",
                   help="Fase a ejecutar: summarize (antes), preprocess (después) o all.")
    args = parser.parse_args()

    _ensure_dir(args.workdir)

    # 1) Load & reshape
    wide, series_name_map, dfl = load_and_reshape_from_YR(args.input)
    wide.to_csv(os.path.join(args.workdir, "wide_raw.csv"), index=False)
    # dfl.to_csv(os.path.join(args.workdir, "long_format.csv"), index=False)
    pd.Series(series_name_map).to_csv(os.path.join(args.workdir, "series_name_map.csv"))

    # 2) Summarize BEFORE
    if args.stage in ("summarize", "all"):
        summarize_dir = os.path.join(args.workdir, "summary_first")
        summarize_before_preprocess(wide, summarize_dir)

    # 3) Preprocess with policy
    if args.stage in ("preprocess", "all"):
        wide_clean = preprocess_with_policy(
            wide,
            start=args.start,
            end=args.end,
            winsorize_percap_vars=args.winsor,
            winsor_limits=tuple(args.winsor_limits),
            recommendations_csv=args.policy_csv
        )
        wide_clean.to_csv(os.path.join(args.workdir, "wide_clean.csv"), index=False)

        # 4) Summaries AFTER (opcionales)
        summarize_after_preprocess(wide_clean, os.path.join(args.workdir, "summary_after"), args.focus_year)
        compute_country_coverage_by_series(wide, args.workdir, min_years=11)

    print("ETL finalizado.")

if __name__ == "__main__":
    main()

