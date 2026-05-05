#!/usr/bin/env python3
"""
predict.py --- Run a trained model on new / external data
=========================================================

Loads the model and preprocessing state saved by train.py and generates
predictions for any new dataset.

The preprocessing pipeline (imputation, winsorisation, feature selection,
scaling, clinical feature engineering) is applied identically to how it
was applied during training, using the fitted state saved in
weights/preprocessor.pkl.

USAGE
-----
python predict.py --data /path/to/new_data.csv
python predict.py --data /path/to/new_data.csv --out predictions.csv

OUTPUT COLUMNS
--------------
sample_id  --- sample_id if present in input, otherwise row index
prob_ati   --- predicted probability of ATI (0--1)
pred_label --- hard prediction: 0 = No ATI, 1 = ATI (threshold 0.5)
true_label --- ground-truth (0/1) if an 'ati' column is present; else omitted
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)

_SCRIPT_DIR      = Path(__file__).resolve().parent
_MODEL_PATH      = _SCRIPT_DIR / "weights" / "model.pkl"
_PREPROCESSOR_PATH = _SCRIPT_DIR / "weights" / "preprocessor.pkl"
_FEATURES_PATH   = _SCRIPT_DIR / "weights" / "feature_cols.json"


def parse_args():
    p = argparse.ArgumentParser(
        description="Run a trained ATI model on new proteomics data."
    )
    p.add_argument(
        "--data",
        required=True,
        help="Path to input CSV (same raw columns as training data)",
    )
    p.add_argument(
        "--out",
        default="predictions.csv",
        help="Output CSV path (default: predictions.csv)",
    )
    return p.parse_args()


def load_model(model_path: str, preprocessor_path: str, features_path: str):
    """
    Load the trained model and preprocessing state from disk.
    """
    logging.info(f"Loading model         : {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    logging.info(f"Loading preprocessor  : {preprocessor_path}")
    with open(preprocessor_path, "rb") as f:
        preproc_state = pickle.load(f)

    logging.info(f"Loading feature list  : {features_path}")
    with open(features_path) as f:
        feature_cols = json.load(f)

    logging.info(f"Model expects {len(feature_cols)} features")
    return model, preproc_state, feature_cols


def prepare_features(df: pd.DataFrame, preproc_state: dict):
    """
    Run the full preprocessing pipeline on new data using the fitted
    state from training (fit=False so no transforms are re-fitted).
    """
    from preprocess import build_features_and_labels

    X, y, feature_cols, _ = build_features_and_labels(
        df, fit=False, preproc_state=preproc_state
    )

    ids = df["sample_id"] if "sample_id" in df.columns else pd.RangeIndex(len(df))

    # Align ids to rows that survived preprocessing (dropna on ati)
    if "ati" in df.columns and y is not None:
        # build_features_and_labels drops rows with missing ati
        valid_mask = df["ati"].notna().values
        ids = ids[valid_mask] if hasattr(ids, '__getitem__') else ids

    return X, y, ids


def run_predict(model, X: np.ndarray, ids, y_true=None) -> pd.DataFrame:
    """
    Generate ATI predictions for a preprocessed feature matrix.
    """
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    out = pd.DataFrame({
        "sample_id" : ids,
        "prob_ati"  : y_prob,
        "pred_label": y_pred,
    })
    if y_true is not None:
        out["true_label"] = np.asarray(y_true)
    return out


def evaluate(results: pd.DataFrame):
    """Print classification metrics when ground-truth labels are available."""
    from sklearn.metrics import classification_report, roc_auc_score, log_loss

    y_true = results["true_label"].values
    y_pred = results["pred_label"].values
    y_prob = results["prob_ati"].values

    print(f"\n{'=' * 50}")
    print(f"  Samples : {len(results)} | "
          f"No ATI: {(y_true==0).sum()} | ATI: {(y_true==1).sum()}")
    print(f"{'=' * 50}")
    if len(np.unique(y_true)) > 1:
        print(classification_report(y_true, y_pred, target_names=["No ATI", "ATI"]))
        print(f"AUC (ROC)  : {roc_auc_score(y_true, y_prob):.3f}")
        print(f"Log loss   : {log_loss(y_true, y_prob):.3f}")
    else:
        print("  (Only one class present in labels --- AUC not defined)")


def main():
    args = parse_args()

    model, preproc_state, feature_cols = load_model(
        str(_MODEL_PATH),
        str(_PREPROCESSOR_PATH),
        str(_FEATURES_PATH),
    )

    logging.info(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data, low_memory=False, na_values=[".", ""])
    logging.info(f"  {len(df)} samples loaded")

    logging.info("Running preprocessing pipeline...")
    X, y_true, ids = prepare_features(df, preproc_state)
    logging.info(f"  Preprocessed shape: {X.shape}")

    results = run_predict(model, X, ids, y_true=y_true)

    if y_true is not None:
        evaluate(results)

    results.to_csv(args.out, index=False)
    logging.info(f"Predictions saved to: {args.out}")


if __name__ == "__main__":
    main()
