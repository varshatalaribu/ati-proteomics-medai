#!/usr/bin/env python3
"""
evaluate.py --- Cross-validation for model development (v2)
===========================================================

Runs stratified k-fold CV and reports AUC, log loss, Brier, ECE, AUPRC.
Generates ROC, PR, calibration, and log-loss comparison plots.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, log_loss, brier_score_loss,
    average_precision_score, precision_recall_curve,
)
from sklearn.calibration import calibration_curve

from model import MODELS, CV_FOLDS, RANDOM_SEED, build_model
from preprocess import load_data, build_features_and_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

_SCRIPT_DIR   = Path(__file__).resolve().parent
_DEFAULT_DATA = _SCRIPT_DIR.parent.parent / "BKBC_train" / "train.csv"
_DEFAULT_OUT  = _SCRIPT_DIR.parent.parent / "results"


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(y_prob, bins[1:-1])
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return ece


def parse_args():
    p = argparse.ArgumentParser(description="Cross-validate ATI models.")
    p.add_argument("--data",   default=str(_DEFAULT_DATA))
    p.add_argument("--out",    default=str(_DEFAULT_OUT))
    p.add_argument("--folds",  type=int, default=CV_FOLDS)
    p.add_argument("--models", nargs="+", default=None)
    return p.parse_args()


def run_cv(model, X, y, n_folds, name):
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    logging.info(f"[{name}] {n_folds}-fold CV...")
    y_prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    fold_results = []
    for fold_idx, (_, val_idx) in enumerate(cv.split(X, y)):
        yv = y[val_idx]
        pv = y_prob[val_idx]
        auc   = roc_auc_score(yv, pv) if len(np.unique(yv)) > 1 else np.nan
        ll    = log_loss(yv, pv)
        brier = brier_score_loss(yv, pv)
        ece   = expected_calibration_error(yv, pv)
        auprc = average_precision_score(yv, pv) if len(np.unique(yv)) > 1 else np.nan
        fold_results.append({
            "model": name, "fold": fold_idx, "n_samples": len(val_idx),
            "n_ati": int(yv.sum()), "auc": auc, "log_loss": ll,
            "brier": brier, "ece": ece, "auprc": auprc,
        })
    return y_pred, y_prob, fold_results


def print_metrics(y, y_pred, y_prob, title):
    print(f"\n{'=' * 62}")
    print(f"  {title}")
    print(f"{'=' * 62}")
    print(classification_report(y, y_pred, target_names=["No ATI", "ATI"]))
    if len(np.unique(y)) > 1:
        print(f"  AUC-ROC  : {roc_auc_score(y, y_prob):.4f}")
        print(f"  AUPRC    : {average_precision_score(y, y_prob):.4f}")
        print(f"  Log Loss : {log_loss(y, y_prob):.4f}")
        print(f"  Brier    : {brier_score_loss(y, y_prob):.4f}")
        print(f"  ECE      : {expected_calibration_error(y, y_prob):.4f}")


def plot_confusion(y, y_pred, path, title, n_folds):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y, y_pred),
                           display_labels=["No ATI", "ATI"]).plot(ax=ax, colorbar=False)
    ax.set_title(f"ATI -- {n_folds}-Fold CV -- {title}")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_logloss(summary, path):
    df = summary.sort_values("mean_log_loss")
    colours = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(df))]
    fig, ax = plt.subplots(figsize=(max(6, len(df) * 1.2), 5))
    bars = ax.bar(df["model"], df["mean_log_loss"], yerr=df["std_log_loss"],
                  capsize=4, color=colours, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Mean CV Log Loss"); ax.set_title("Model Comparison")
    ax.set_xticklabels(df["model"], rotation=25, ha="right")
    for bar, val in zip(bars, df["mean_log_loss"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_roc(roc_data, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    colours = cm.tab10(np.linspace(0, 1, len(roc_data)))
    for (name, (fpr, tpr, auc_val)), col in zip(roc_data.items(), colours):
        ax.plot(fpr, tpr, lw=1.8, color=col, label=f"{name} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC -- OOF CV")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_pr(pr_data, path, baseline):
    fig, ax = plt.subplots(figsize=(7, 6))
    colours = cm.tab10(np.linspace(0, 1, len(pr_data)))
    for (name, (prec, rec, auprc)), col in zip(pr_data.items(), colours):
        ax.plot(rec, prec, lw=1.8, color=col, label=f"{name} (AUPRC={auprc:.3f})")
    ax.axhline(baseline, color="k", ls="--", lw=0.8, label=f"Random ({baseline:.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("PR -- OOF CV")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_calibration(calib_data, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    colours = cm.tab10(np.linspace(0, 1, len(calib_data)))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect")
    for (name, (fp, mp, ece)), col in zip(calib_data.items(), colours):
        ax.plot(mp, fp, "o-", lw=1.5, color=col, label=f"{name} (ECE={ece:.3f})")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Observed")
    ax.set_title("Calibration -- OOF CV"); ax.legend(fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    X, y, feature_cols, _ = build_features_and_labels(df, fit=True)

    model_names = args.models if args.models else list(MODELS.keys())
    bad = [n for n in model_names if n not in MODELS]
    if bad:
        raise ValueError(f"Unknown model(s): {bad}")

    all_folds = []
    roc_data, pr_data, calib_data = {}, {}, {}

    for name in model_names:
        model = build_model(name)
        y_pred, y_prob, folds = run_cv(model, X, y, args.folds, name)
        all_folds.extend(folds)
        print_metrics(y, y_pred, y_prob, f"{name} -- {args.folds}-Fold CV")

        slug = name.lower().replace(" ", "_")
        plot_confusion(y, y_pred, out_dir / f"cv_cm_{slug}.png", name, args.folds)

        if len(np.unique(y)) > 1:
            fpr, tpr, _ = roc_curve(y, y_prob)
            prec, rec, _ = precision_recall_curve(y, y_prob)
            fp, mp = calibration_curve(y, y_prob, n_bins=10)
            ece = expected_calibration_error(y, y_prob)
            auprc = average_precision_score(y, y_prob)
            roc_data[name]   = (fpr, tpr, roc_auc_score(y, y_prob))
            pr_data[name]    = (prec, rec, auprc)
            calib_data[name] = (fp, mp, ece)

    # ── Save results ───────────────────────────────────────────────────
    results_df = pd.DataFrame(all_folds)
    results_df.to_csv(out_dir / "cv_results.csv", index=False)

    summary = results_df.groupby("model").agg(
        mean_auc=("auc", "mean"), std_auc=("auc", "std"),
        mean_log_loss=("log_loss", "mean"), std_log_loss=("log_loss", "std"),
        mean_brier=("brier", "mean"), std_brier=("brier", "std"),
        mean_ece=("ece", "mean"), mean_auprc=("auprc", "mean"),
    ).reset_index().sort_values("mean_log_loss")
    summary.to_csv(out_dir / "cv_summary.csv", index=False)

    # ── Console summary ────────────────────────────────────────────────
    print(f"\n{'=' * 82}")
    print(f"{'Model':<14} | {'AUC':>7} | {'LogLoss':>9} | {'Brier':>7} | {'ECE':>7} | {'AUPRC':>7}")
    print(f"{'-' * 82}")
    for i, (_, row) in enumerate(summary.iterrows()):
        star = " *" if i == 0 else "  "
        print(f"{row['model']:<14} |"
              f" {row['mean_auc']:>7.4f} |"
              f" {row['mean_log_loss']:>7.4f}+/-{row['std_log_loss']:.3f} |"
              f" {row['mean_brier']:>7.4f} |"
              f" {row['mean_ece']:>7.4f} |"
              f" {row['mean_auprc']:>7.4f}{star}")
    print(f"{'=' * 82}")
    best = summary.iloc[0]
    print(f"\n  * Best: {best['model']} (LogLoss={best['mean_log_loss']:.4f})")

    # ── Plots ──────────────────────────────────────────────────────────
    plot_logloss(summary, out_dir / "cv_logloss.png")
    if roc_data:
        plot_roc(roc_data, out_dir / "cv_roc.png")
        plot_pr(pr_data, out_dir / "cv_pr.png", baseline_rate=y.mean())
        plot_calibration(calib_data, out_dir / "cv_calibration.png")

    logging.info(f"All outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
