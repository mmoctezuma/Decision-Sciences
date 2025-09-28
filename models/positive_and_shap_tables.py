# -*- coding: utf-8 -*-
"""
make_positive_and_shap_tables.py

Produce CSV tables (no Markdown) for:
  1) Positive class by country from compare_10years.csv
  2) Regional share of positive class (if region column exists)
  3) Optional SHAP global importance table (no plots), using a saved XGBoost model and X_test
"""

import os
import json
import argparse
import logging
from typing import Optional, List

import pandas as pd

try:
    import shap
    import xgboost as xgb
except Exception:
    shap = None
    xgb = None

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
logger = logging.getLogger("make_positive_and_shap_tables")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def try_read_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded CSV: {path} (rows={len(df)})")
        return df
    except Exception as e:
        logger.error(f"Failed to read CSV '{path}': {e}")
        return None


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
    logger.info(f"Saved: {path} (rows={len(df)}, cols={len(df.columns)})")


def build_positive_table(compare: pd.DataFrame, prefer_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Return positive-class table sorted by iso3c (if present)."""
    if "likely_reduce_CO2" not in compare.columns:
        raise ValueError("Column 'likely_reduce_CO2' not found in compare CSV.")
    pos = compare[compare["likely_reduce_CO2"] == 1].copy()
    if pos.empty:
        logger.warning("No positive-class rows found (likely_reduce_CO2 == 1). Returning empty table.")
        return pd.DataFrame()

    selected = []
    if prefer_cols:
        selected = [c for c in prefer_cols if c in pos.columns]
    if not selected:
        candidates = ["iso3c", "Country", "country", "name", "region", "Region", "wb_region", "delta"]
        selected = [c for c in candidates if c in pos.columns]
    if not selected:
        selected = list(pos.columns)
    
    sort_col = "iso3c" if "iso3c" in pos.columns else selected[0]
    pos = pos[selected].sort_values(sort_col).reset_index(drop=True)
    return pos


def build_regional_share(compare: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Compute share of positive class by region if region column exists."""
    region_col = next((c for c in ["region", "Region", "wb_region"] if c in compare.columns), None)
    if not region_col:
        logger.info("No region column found; skipping regional share table.")
        return None
    if "likely_reduce_CO2" not in compare.columns:
        raise ValueError("Column 'likely_reduce_CO2' not found for regional share computation.")
    reg = compare.groupby(region_col)["likely_reduce_CO2"].mean().reset_index()
    reg["share_positive_pct"] = (100 * reg["likely_reduce_CO2"]).round(1)
    reg = reg[[region_col, "share_positive_pct"]].sort_values(region_col).reset_index(drop=True)
    return reg


def compute_shap_global_importance(xgb_model_path: str, x_test_csv: str, features_json: Optional[str]) -> pd.DataFrame:
    """Compute SHAP global importance: mean |SHAP| per feature (no plots)."""
    if shap is None or xgb is None:
        raise ImportError("xgboost/shap not available. Install them or omit --add_shap.")

    bst = xgb.Booster()
    bst.load_model(xgb_model_path)
    X_test = pd.read_csv(x_test_csv)

    if features_json and os.path.exists(features_json):
        with open(features_json, "r", encoding="utf-8") as f:
            feature_list = json.load(f)
        cols = [c for c in feature_list if c in X_test.columns]
        if cols:
            X_test = X_test[cols]
    elif isinstance(X_test.columns, pd.MultiIndex):
        X_test.columns = ["__".join(map(str, c)) for c in X_test.columns]

    explainer = shap.TreeExplainer(bst)
    shap_values = explainer.shap_values(X_test)
    import numpy as np
    abs_vals = np.abs(shap_values)
    mean_abs = abs_vals.mean(axis=0)
    mean_raw = shap_values.mean(axis=0)
    std_abs = abs_vals.std(axis=0)

    features = list(X_test.columns)
    out = pd.DataFrame({
        "feature": features,
        "mean_abs_shap": mean_abs,
        "mean_shap": mean_raw,
        "std_abs_shap": std_abs,
        "n_samples": X_test.shape[0],
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate CSV tables: positive class by country, regional share, and optional SHAP importance.")
    parser.add_argument("--compare_csv", required=True, help="Path to compare_10years.csv")
    parser.add_argument("--outdir", required=True, help="Directory to save CSV outputs")
    parser.add_argument("--positive_table", action="store_true", help="Save positive_class_by_country.csv")
    parser.add_argument("--regional_share", action="store_true", help="Save regional_positive_share.csv if region column exists")
    parser.add_argument("--add_shap", action="store_true", help="Compute SHAP global importance and save shap_global_importance.csv (requires xgboost, shap)")
    parser.add_argument("--xgb_model_path", help="Path to XGBoost model .json (required if --add_shap)")
    parser.add_argument("--x_test_csv", help="CSV with X_test features (required if --add_shap)")
    parser.add_argument("--features_json", help="Optional JSON with training feature order to align X_test")
    args = parser.parse_args()

    ensure_outdir(args.outdir)

    compare = try_read_csv(args.compare_csv)
    if compare is None or compare.empty:
        raise SystemExit("compare_10years.csv missing or empty.")

    if args.positive_table:
        pos = build_positive_table(compare)
        out_path = os.path.join(args.outdir, "positive_class_by_country.csv")
        save_csv(pos, out_path)

    if args.regional_share:
        reg = build_regional_share(compare)
        if reg is not None:
            out_path = os.path.join(args.outdir, "regional_positive_share.csv")
            save_csv(reg, out_path)

    if args.add_shap:
        if not (args.xgb_model_path and args.x_test_csv):
            logger.error("--add_shap requires --xgb_model_path and --x_test_csv")
        else:
            try:
                shap_df = compute_shap_global_importance(args.xgb_model_path, args.x_test_csv, args.features_json)
                out_path = os.path.join(args.outdir, "shap_global_importance.csv")
                save_csv(shap_df, out_path)
            except Exception as e:
                logger.error(f"SHAP importance computation failed: {e}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
