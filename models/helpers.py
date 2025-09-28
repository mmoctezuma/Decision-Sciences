# -*- coding: utf-8 -*-
"""
Shared helpers for modeling and projection.

Includes:
- load_panel: basic CSV loader for panel data
- build_features: log-transform target/GDP and optional controls
- Projection utilities: compute_cagr, project_series, extend_country, extended_data
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------------
# IO helpers
# ----------------------------------
def load_panel(csv_path: str) -> pd.DataFrame:
    """Load panel CSV and ensure basic types."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["iso3c", "year"])  # minimal requirements
    df["year"] = df["year"].astype(int)
    return df


def ensure_dir(path: str) -> None:
    """Create directory tree if missing (no-op if exists)."""
    import os
    os.makedirs(path, exist_ok=True)


def build_features(
    df: pd.DataFrame,
    target: str = "EN.GHG.CO2.MT.CE.AR5",
    gdp: str = "NY.GDP.MKTP.CD",
    controls: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create ln_CO2, ln_GDP and ln_<control> features; drop rows with NaNs on these.
    Returns (features_df, ln_controls_list).
    """
    if controls is None:
        controls = ["SP.POP.TOTL", "SP.URB.TOTL.IN.ZS", "EN.GHG.CO2.LU.MT.CE.AR5"]

    keep = ["iso3c", "Country", "year", target, gdp] + [c for c in controls if c in df.columns]
    X = df.loc[:, [c for c in keep if c in df.columns]].copy()

    def _safe_log(s: pd.Series, eps: float = 1e-9) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        min_pos = s[s > 0].min()
        floor = float(min_pos) if pd.notna(min_pos) else eps
        return np.log(s.clip(lower=floor))

    X["ln_CO2"] = _safe_log(X[target])
    X["ln_GDP"] = _safe_log(X[gdp])

    ln_controls: List[str] = []
    for c in controls:
        if c in X.columns:
            X[f"ln_{c}"] = _safe_log(X[c])
            ln_controls.append(f"ln_{c}")

    X = X.dropna(subset=["ln_CO2", "ln_GDP"] + ln_controls)
    return X, ln_controls


# ----------------------------------
# Projection helpers (CAGR-based)
# ----------------------------------
def compute_cagr(series: pd.Series) -> Optional[float]:
    """Compute CAGR from indexed series; returns None if insufficient data."""
    valid = series.dropna()
    if len(valid) < 2:
        return None
    valid = valid.sort_index()
    start, end = valid.iloc[0], valid.iloc[-1]
    years = len(valid) - 1
    if start > 0 and end > 0 and years > 0:
        return (end / start) ** (1 / years) - 1
    return np.nan


def project_series(series: pd.Series, years: int = 5) -> pd.Series:
    """Project forward by CAGR for a given number of years."""
    g = compute_cagr(series)
    last_idx = series.index[-1]
    if g is None or not np.isfinite(g):
        return pd.Series([np.nan] * years, index=range(last_idx + 1, last_idx + years + 1))
    base = series.iloc[-1]
    vals = [base * ((1 + g) ** i) for i in range(1, years + 1)]
    return pd.Series(vals, index=range(last_idx + 1, last_idx + years + 1))


def extend_country(df_country: pd.DataFrame, years: int = 5) -> Optional[pd.DataFrame]:
    """Project all numeric columns of a single country forward by 'years'."""
    results = []
    df_country = df_country.sort_values("year")
    for col in df_country.columns:
        if col in ["iso3c", "Country", "year", "region"]:
            continue
        s = df_country.set_index("year")[col].dropna()
        if len(s) > 1:
            proj = project_series(s, years)
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


def extended_data(df: pd.DataFrame, years: int = 5) -> pd.DataFrame:
    """Apply extend_country to all countries and concatenate results."""
    out = []
    for iso, g in df.groupby("iso3c"):
        res = extend_country(g, years=years)
        if res is not None and not res.empty:
            out.append(res)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)
