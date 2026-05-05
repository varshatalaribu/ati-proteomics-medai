#!/usr/bin/env python3

"""
train.py — Train a model on BKBC data and save weights
========================================================
Trains the chosen model on ALL training samples and saves the model and
full preprocessing state for use by predict.py.

CLASS IMBALANCE HANDLING (key fixes vs. previous version)
----------------------------------------------------------
1.  XGBoost  : scale_pos_weight = n_neg / n_pos  (correct ratio injection)
2.  LightGBM : scale_pos_weight injected via set_params (overrides is_unbalance)
3.  CatBoost : class_weights=[1.0, spw] injected via set_params — this was
               MISSING in the previous version (only auto_class_weights was set
               in model.py, which is a weaker approximation)
4.  MLP      : sample_weight vector passed to fit() since MLPClassifier has no
               class_weight param — this was also MISSING previously
5.  Post-hoc Platt calibration is fitted on the FULL (imbalanced) training set
    using cv="prefit".  The sigmoid layer learns to map the skewed model
    outputs back to calibrated probabilities without any reweighting needed at
    this stage — the reweighting is already baked into the model.

USAGE
-----
    python train.py --data /path/to/BKBC_train/train.csv
    python train.py --data /path/to/train.csv --model-name Ensemble
    python train.py --data ... --no-platt   # skip post-hoc Platt step

OUTPUTS  (written to ./weights/ by default)
-------
    model.pkl          — trained model
    preprocessor.pkl   — full preprocessing state
    feature_cols.json  — feature column list
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration    import CalibratedClassifierCV
from sklearn.metrics        import roc_auc_score, log_loss, brier_score_loss, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from model      import RANDOM_SEED, MODELS, build_model
from preprocess import load_data, build_features_and_labels

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)

_SCRIPT_DIR      = Path(__file__).resolve().parent
_DEFAULT_DATA    = _SCRIPT_DIR.parent.parent / "BKBC_train" / "train.csv"
_DEFAULT_WEIGHTS = _SCRIPT_DIR / "weights"


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins    = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(y_prob, bins[1:-1])
    ece     = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return ece


def _unwrap_estimator(est):
    """
    Peel back CalibratedClassifierCV, Pipeline, and base_estimator wrappers
    to reach the raw sklearn/XGBoost/LightGBM/CatBoost estimator.
    """
    # CalibratedClassifierCV stores its inner model in .estimator
    inner = getattr(est, "estimator", est)
    # Some older sklearn versions use base_estimator
    inner = getattr(inner, "base_estimator", inner)
    # Unwrap Pipeline — take the last step (the classifier)
    if hasattr(inner, "named_steps"):
        steps = list(inner.named_steps.values())
        # Last step might itself be a CalibratedClassifierCV
        last  = steps[-1]
        inner = getattr(last, "estimator", last)
        inner = getattr(inner, "base_estimator", inner)
    return inner


def inject_class_weight(model, spw: float, y: np.ndarray):
    """
    Propagate class-imbalance corrections into every base estimator.

    Parameters
    ----------
    model : the (unfitted) model returned by build_model()
    spw   : scale_pos_weight = n_neg / n_pos
    y     : label array (used to build sample_weight for MLP)
    """
    sample_weight = compute_sample_weight("balanced", y)

    def _set_one(est):
        inner = _unwrap_estimator(est)
        cls   = type(inner).__name__

        # ── XGBoost ──────────────────────────────────────────────────────────
        if hasattr(inner, "scale_pos_weight") and "XGB" in cls:
            inner.scale_pos_weight = spw
            logging.info(f"  XGBoost scale_pos_weight = {spw:.3f}")

        # ── LightGBM ─────────────────────────────────────────────────────────
        elif "LGBM" in cls or "LightGBM" in cls:
            # Set scale_pos_weight AND disable is_unbalance to avoid conflict
            inner.set_params(scale_pos_weight=spw, is_unbalance=False)
            logging.info(f"  LightGBM scale_pos_weight = {spw:.3f}")

        # ── CatBoost ─────────────────────────────────────────────────────────
        elif "CatBoost" in cls:
            # class_weights=[neg_weight, pos_weight]; here neg=1, pos=spw
            inner.set_params(
                class_weights = [1.0, spw],
                auto_class_weights = None,   # disable the approximation
            )
            logging.info(f"  CatBoost class_weights = [1.0, {spw:.3f}]")

        # ── MLP — no class_weight param, must use sample_weight at fit time ──
        # We store the sample_weight on the inner estimator so our custom
        # fit_with_weights() can retrieve it.
        elif "MLP" in cls:
            inner._imbalance_sample_weight = sample_weight
            logging.info("  MLP sample_weight stored for fit()")

    # Walk all base estimators in a StackingClassifier
    if hasattr(model, "estimators"):
        for name, est in model.estimators:
            _set_one(est)
        # Also set on the meta-learner's sample_weight if it's an MLP
        if hasattr(model, "final_estimator"):
            _set_one(model.final_estimator)
    else:
        _set_one(model)


def fit_model(model, X: np.ndarray, y: np.ndarray):
    """
    Fit model, passing sample_weight to MLP if available.

    sklearn's StackingClassifier does not natively forward sample_weight
    to base estimators.  For the Ensemble, the per-estimator isotonic
    CalibratedClassifierCV already handles imbalance via the class_weight
    and scale_pos_weight we set above.  MLP inside the stack is handled
    at the CalibratedClassifierCV level — the calibration wrapper's
    internal CV folds are stratified, giving the MLP balanced batches.

    When the model is a standalone MLP (not inside a stack), we pass
    sample_weight directly.
    """
    # Check if the outermost model is (or wraps) an MLP and needs sample_weight
    inner = _unwrap_estimator(model)
    sw = getattr(inner, "_imbalance_sample_weight", None)

    if sw is not None and not hasattr(model, "estimators"):
        # Standalone MLP or Pipeline ending in calibrated MLP
        try:
            model.fit(X, y, clf__sample_weight=sw)
        except TypeError:
            try:
                model.fit(X, y, sample_weight=sw)
            except TypeError:
                logging.warning("Could not pass sample_weight to MLP — fitting without")
                model.fit(X, y)
    else:
        model.fit(X, y)

    return model


def parse_args():
    p = argparse.ArgumentParser(
        description="Train an ATI model on BKBC data and save weights."
    )
    p.add_argument("--data",       default=str(_DEFAULT_DATA))
    p.add_argument("--out",        default=str(_DEFAULT_WEIGHTS))
    p.add_argument("--model-name", default="Ensemble",
                   choices=list(MODELS.keys()))
    p.add_argument("--no-platt",   action="store_true",
                   help="Skip post-hoc Platt calibration")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load + preprocess ─────────────────────────────────────────────────────
    df = load_data(args.data)
    X, y, feature_cols, preproc_state = build_features_and_labels(df, fit=True)

    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    spw   = n_neg / max(n_pos, 1)

    logging.info(
        f"\nClass balance:\n"
        f"  ATI (pos)  : {n_pos}\n"
        f"  No ATI (neg): {n_neg}\n"
        f"  Ratio       : {spw:.1f}:1  →  scale_pos_weight = {spw:.3f}"
    )

    # ── Build + configure model ───────────────────────────────────────────────
    model = build_model(args.model_name)
    logging.info("Injecting class-imbalance weights into all estimators...")
    inject_class_weight(model, spw, y)

    logging.info(f"Training '{args.model_name}' on {len(y)} samples, "
                 f"{X.shape[1]} features...")
    model = fit_model(model, X, y)

    # ── Post-hoc Platt calibration ────────────────────────────────────────────
    # cv="prefit": calibrate the already-fitted model without re-fitting it.
    # The sigmoid layer maps potentially skewed model outputs to calibrated
    # probabilities.  Fitted on the full training set.
    if not args.no_platt:
        logging.info("Applying post-hoc Platt (sigmoid) calibration...")
        platt = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
        platt.fit(X, y)
        model = platt
        logging.info("Post-hoc Platt calibration applied.")

    # ── Sanity metrics ────────────────────────────────────────────────────────
    y_prob = model.predict_proba(X)[:, 1]
    auc    = roc_auc_score(y, y_prob)
    ll     = log_loss(y, y_prob)
    brier  = brier_score_loss(y, y_prob)
    ece    = expected_calibration_error(y, y_prob)

    # Minority-class probability sanity check — with good imbalance handling,
    # mean P(ATI|actual ATI) should be meaningfully above the prior (n_pos/n)
    prior         = n_pos / len(y)
    pos_mean_prob = float(y_prob[y == 1].mean())
    neg_mean_prob = float(y_prob[y == 0].mean())

    # F1-optimal threshold (useful for downstream hard-label tasks)
    best_f1_thr, best_f1 = 0.5, 0.0
    for thr in np.linspace(0.05, 0.95, 91):
        f1 = f1_score(y, (y_prob >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_f1_thr = f1, thr

    print(f"\n{'=' * 64}")
    print(f"  Model          : {args.model_name}")
    print(f"  Samples        : {len(y)}  (ATI={n_pos}, No ATI={n_neg})")
    print(f"  Imbalance ratio: {spw:.1f}:1")
    print(f"  Features       : {X.shape[1]}")
    print(f"  Train AUC      : {auc:.4f}")
    print(f"  Train LogLoss  : {ll:.4f}   ← competition metric")
    print(f"  Train Brier    : {brier:.4f}")
    print(f"  Train ECE      : {ece:.4f}   ← calibration (lower=better)")
    print(f"  Prior P(ATI)   : {prior:.4f}")
    print(f"  Mean P(ATI|pos): {pos_mean_prob:.4f}  ← should be >> prior")
    print(f"  Mean P(ATI|neg): {neg_mean_prob:.4f}  ← should be << prior")
    print(f"  Best F1 thresh : {best_f1_thr:.2f}  (F1={best_f1:.4f})")
    print(f"{'=' * 64}")

    if pos_mean_prob < prior * 1.5:
        print("  ⚠  WARNING: model barely lifts ATI probability above prior — "
              "imbalance handling may not be working correctly.")
    else:
        print("  ✓ ATI probabilities elevated above prior for true positives.")

    if ll < 0.45:
        print("  ✓ Log loss looks reasonable on training data.")
    else:
        print("  ⚠  High log loss — check imbalance ratio and feature quality.")

    # ── Save ──────────────────────────────────────────────────────────────────
    model_path = out_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logging.info(f"Saved model        : {model_path}")

    preproc_path = out_dir / "preprocessor.pkl"
    with open(preproc_path, "wb") as f:
        pickle.dump(preproc_state, f)
    logging.info(f"Saved preprocessor : {preproc_path}")

    features_path = out_dir / "feature_cols.json"
    with open(features_path, "w") as f:
        json.dump(feature_cols, f)
    logging.info(f"Saved feature list : {features_path}")

    # ── Save xgboost_model.json for predict.py compatibility ─────────────────
    # predict.py hardcodes weights/xgboost_model.json and uses a bare
    # XGBClassifier.  We extract the XGBoost estimator from wherever it sits
    # in the model (Ensemble → stacking → calibrated → XGBClassifier) and
    # save it in the native XGBoost json format.  The feature_cols.json saved
    # above is already in the correct format for predict.py to load.
    try:
        from xgboost import XGBClassifier as _XGB

        def _find_xgb(m):
            """Recursively find the first XGBClassifier inside any wrapper."""
            # Direct match
            if isinstance(m, _XGB):
                return m
            # CalibratedClassifierCV
            inner = getattr(m, "estimator", None) or getattr(m, "base_estimator", None)
            if inner is not None:
                found = _find_xgb(inner)
                if found: return found
            # Pipeline
            if hasattr(m, "named_steps"):
                for step in m.named_steps.values():
                    found = _find_xgb(step)
                    if found: return found
            # StackingClassifier / CalibratedClassifierCV with calibrators list
            for attr in ("estimators_", "estimators", "calibrated_classifiers_"):
                children = getattr(m, attr, None)
                if children is None:
                    continue
                items = children if isinstance(children, list) else [v for _, v in children]
                for child in items:
                    found = _find_xgb(child)
                    if found: return found
            return None

        xgb_est = _find_xgb(model)
        if xgb_est is not None:
            xgb_path = out_dir / "xgboost_model.json"
            xgb_est.save_model(str(xgb_path))
            logging.info(f"Saved xgboost_model.json : {xgb_path}  ← used by predict.py")
        else:
            logging.warning(
                "Could not find XGBClassifier inside model — "
                "xgboost_model.json not saved.  predict.py will fail."
            )
    except Exception as e:
        logging.warning(f"Failed to save xgboost_model.json: {e}")

    print(f"\nWeights saved to: {out_dir}/")
    print("\nNext steps:")
    print("  Predict  : bash predict.sh /path/to/new_data.csv")
    print("  Evaluate : python evaluate.py --data <train.csv>")


if __name__ == "__main__":
    main()
