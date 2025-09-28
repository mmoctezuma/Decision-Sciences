# -*- coding: utf-8 -*-
import argparse
import logging
import os

import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, classification_report
)
import xgboost as xgb
import shap
import joblib


def build_features(df, series, target, group_col="iso3c", year_col="year", window=10):
    """
    Construye features a partir de una ventana de N años por país.
    Para cada serie: último valor, slope, CAGR, std.
    Features usan datos hasta t-1, target es el valor en t.
    """
    feature_rows = []
    countries = df[group_col].unique()

    for iso in countries:
        df_iso = df[df[group_col] == iso].sort_values(year_col)
        years = df_iso[year_col].unique()

        if len(years) < window + 1:
            continue

        for t in years[window:]:
            df_win = df_iso[df_iso[year_col].between(t - window, t - 1)]

            feats = {group_col: iso, year_col: t}
            for s in series:
                vals = df_win[s].dropna()
                if len(vals) == 0:
                    continue
                feats[f"{s}_last"] = vals.iloc[-1]
                feats[f"{s}_std"] = vals.std()
                if len(vals) > 1:
                    x = np.arange(len(vals))
                    slope = np.polyfit(x, vals.values, 1)[0]
                    feats[f"{s}_slope"] = slope
                    start, end = vals.iloc[0], vals.iloc[-1]
                    if start > 0 and end > 0:
                        feats[f"{s}_cagr"] = (end / start) ** (1 / len(vals)) - 1
                    else:
                        feats[f"{s}_cagr"] = np.nan

            feats[target] = df_iso.loc[df_iso[year_col] == t, target].values[0]
            feature_rows.append(feats)

    return pd.DataFrame(feature_rows)


def evaluate(y_true, y_pred, y_prob):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="CO2 classifier")
    parser.add_argument("--input_file", required=True, help="Path to input CSV")
    parser.add_argument("--output_dir", required=True, help="Where to save results")
    parser.add_argument("--features", nargs="+", required=True, help="List of feature series")
    parser.add_argument("--target", default="likely_reduce_CO2", help="Binary target variable")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Loading data...")
    df = pd.read_csv(args.input_file)
    df = df.dropna(subset=[args.target])

    logger.info("Building features...")
    feat_df = build_features(df, args.features, args.target)

    X = feat_df.drop(columns=[args.target, "iso3c", "year"])
    y = feat_df[args.target]
    groups = feat_df["iso3c"]
    
    logit = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l1", solver="liblinear",
            class_weight="balanced", max_iter=500
        ))
    ])

    params_xgb = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 5,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": (y.value_counts()[0] / y.value_counts()[1]),
        "seed": 42,
    }

    # --- Temporal validation ---
    logger.info("Running temporal validation...")
    time_splits = [2000,2005,2010]

    preds_logit, probs_logit, y_true = [], [], []
    preds_xgb, probs_xgb = [], []

    for split_year in time_splits:
        train_mask = feat_df["year"] <= split_year
        test_mask = (feat_df["year"] > split_year)

        if test_mask.sum() == 0:
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # --- Logistic Regression ---
        logit.fit(X_train, y_train)
        p = logit.predict(X_test)
        pr = logit.predict_proba(X_test)[:, 1]
        preds_logit.extend(p)
        probs_logit.extend(pr)
        y_true.extend(y_test)

        # --- XGBoost ---
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        bst = xgb.train(params_xgb, dtrain, num_boost_round=500,
                        evals=[(dtest, "test")], early_stopping_rounds=20)
        p = (bst.predict(dtest) > 0.5).astype(int)
        pr = bst.predict(dtest)
        preds_xgb.extend(p)
        probs_xgb.extend(pr)

    # --- Metrics ---
    metrics_logit = evaluate(y_true, preds_logit, probs_logit)
    metrics_xgb = evaluate(y_true, preds_xgb, probs_xgb)
    logger.info(f"Logit metrics: {metrics_logit}")
    logger.info(f"XGBoost metrics: {metrics_xgb}")

    # --- Save artifacts ---
    joblib.dump(logit, os.path.join(args.output_dir, "logit_model.pkl"))
    bst.save_model(os.path.join(args.output_dir, "xgb_classifier_model.json"))

    results = [
        {"model": "Logit", "metrics": metrics_logit},
        {"model": "XGBoost", "metrics": metrics_xgb},
    ]
    pd.DataFrame(results).to_csv(
        os.path.join(args.output_dir, "metrics_classifier.csv"), index=False
    )

    logger.info("Done.")



if __name__ == "__main__":
    main()

