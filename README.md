# Acute Tubular Injury Prediction from Plasma Proteomics
### Boston Kidney Biopsy Cohort (BKBC) — MedAI Hackathon Submission

> **Challenge**: Binary classification of acute tubular injury (ATI) from high-dimensional plasma proteomics  
> **Team**: BKBC  
> **Primary Metric**: Log Loss on external held-out test cohort

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Dataset](#2-dataset)
3. [Approach Overview](#3-approach-overview)
4. [Feature Engineering](#4-feature-engineering)
5. [Model Architecture](#5-model-architecture)
6. [Training Strategy](#6-training-strategy)
7. [Results](#7-results)
8. [Repository Structure](#8-repository-structure)
9. [Quickstart](#9-quickstart)
10. [Detailed Usage](#10-detailed-usage)
11. [Dependencies](#11-dependencies)
12. [Research Notes](#12-research-notes)

---

## 1. Problem Statement

Acute tubular injury (ATI) is a common and clinically significant form of acute kidney injury (AKI) that is traditionally diagnosed via renal biopsy — an invasive and costly procedure. This challenge asks: **can ATI be predicted non-invasively from plasma proteomics?**

We frame this as a **binary classification problem**:

| Label | Meaning |
|-------|---------|
| `0` | No ATI |
| `1` | ATI present |

The model is evaluated on a held-out external test cohort using **log loss**, which simultaneously rewards accurate probability estimates and penalises overconfident wrong predictions — making calibration as important as discrimination.

---

## 2. Dataset

**Training data path (HPC):**
```
/projectnb/medaihack/BKBC-hackathon/BKBC_train/train.csv
```

> **Important**: Training on any data source other than the above path is grounds for disqualification.

### Data Characteristics

| Property | Value |
|----------|-------|
| Patients | 426 |
| ATI-positive | 200 (46.9%) |
| ATI-negative | 226 (53.1%) |
| Total features | 6,595 columns |
| Protein features | 6,590 (`feature_XXXX`) |
| Clinical features | `age`, `sex`, `baseline_egfr_23` |
| Identifiers | `sample_id` |

### Feature Description

| Column Group | Description |
|---|---|
| `sample_id` | Anonymised patient identifier |
| `ati` | Binary ATI label — prediction target |
| `age` | Age (10-year bin midpoint) |
| `sex` | Sex (1 = Male, 2 = Female) |
| `baseline_egfr_23` | Baseline eGFR (ml/min/1.73 m²) |
| `feature_XXXX` × 6,590 | Log₂-normalised, ComBat batch-corrected SomaScan plasma protein abundances |

All protein features are **reference-ComBat batch-corrected**, making them directly comparable between training and test cohorts. Missing values are present but sparse.

---

## 3. Approach Overview

Our solution is a **multi-view stacking ensemble** that:

1. Applies a 10-stage feature engineering pipeline to reduce 6,590 proteins to ~250–400 biologically meaningful features
2. Trains 7 diverse base learners on the preprocessed features
3. Learns a Ridge meta-learner to combine base-learner predictions via stacked generalisation
4. Applies dual-level probability calibration (isotonic + Platt) to ensure well-calibrated outputs

The full architecture is illustrated in the pipeline diagram below.

```
train.csv (426 × 6,595)
        │
        ▼
┌────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING PIPELINE                       │
│  1. Clinical feature engineering (interactions, CKD staging)   │
│  2. Protein imputation (median) + clinical imputation (KNN-5)  │
│  3. Rank-based inverse-normal transform per protein             │
│  4. Variance filter (threshold = 0.01)                         │
│  5. ANOVA pre-filter → top 3,500 proteins                      │
│  6. Stability selection (L1 logistic, 150 bootstrap runs)      │
│  7. WGCNA-lite co-expression modules (25 clusters → PCAs)      │
│  8. Pairwise interaction features (top-8 stable proteins)      │
│  9. Robust scaling of clinical block                            │
│ 10. Feature assembly → ~250–400 final features                  │
└────────────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────────────┐
│                    STACKING ENSEMBLE (V2)                       │
│                                                                  │
│  Base learners (each with isotonic calibration):                │
│   ├── XGBoost (hist, 600 trees)                                 │
│   ├── LightGBM (DART, 600 iter)                                 │
│   ├── CatBoost (ordered, 600 iter)                              │
│   ├── ExtraTrees (500 estimators)                               │
│   ├── MLP Neural Network ([256, 64])                            │
│   ├── Elastic-Net Logistic Regression                           │
│   └── SGD Logistic Regression (elastic-net)                     │
│                                                                  │
│  Meta-learner: Ridge Logistic Regression (C=0.2)               │
│  Stacking strategy: 5-fold CV → predict_proba                  │
│                                                                  │
│  Post-hoc Platt (sigmoid) calibration on full training set      │
└────────────────────────────────────────────────────────────────┘
        │
        ▼
  predictions.csv  (prob_ati ∈ [0, 1])
```

---

## 4. Feature Engineering

The preprocessing pipeline (`preprocess.py`) transforms the raw 6,590-protein matrix into a compact, biologically interpretable feature set through 10 sequential stages.

### Stage 1 — Clinical Feature Engineering

Raw clinical covariates are expanded into domain-informed features:

| Engineered Feature | Formula | Rationale |
|---|---|---|
| `age_x_egfr` | `age × baseline_egfr_23` | Captures age-renal function interaction |
| `log_egfr` | `log(baseline_egfr_23 + 1)` | Linearises right-skewed eGFR distribution |
| `egfr_stage` | 6-bin CKD staging | Maps eGFR to clinical CKD stage (G1–G5) |
| `egfr_sq` | `baseline_egfr_23²` | Captures non-linear kidney function effects |
| `age_bin` | 5-bin discretisation | Reduces noise from ordinal age representation |

### Stage 2 — Imputation

- **Protein features**: Median imputation (column-wise)
- **Clinical features**: KNN imputation (k=5, distance-weighted)

### Stage 3 — Rank-Based Inverse-Normal Transform

Each protein column is independently transformed:
1. Rank samples within column
2. Map ranks to uniform quantiles on (0, 1)
3. Apply inverse normal CDF (probit transform)
4. Clip to ±4σ to suppress extreme outliers

This removes all distributional assumptions from protein features and renders the pipeline robust to batch effects and outlier proteins.

### Stage 4 — Variance Filter

Near-constant proteins (variance < 0.01) are removed. These carry no discriminative information and can destabilise downstream estimators.

### Stage 5 — ANOVA Pre-filter

`SelectKBest(f_classif, k=3500)` retains the top 3,500 proteins by univariate ANOVA F-statistic. This reduces dimensionality ~3× before the computationally intensive stability selection step.

### Stage 6 — Stability Selection

A bootstrapped L1 logistic regression procedure identifies robustly predictive proteins:

- **150 bootstrap iterations**, each sampling 80% of patients without replacement
- Per iteration, fit L1 logistic regression (C=0.05) and record selected features
- Retain features selected in **≥50% of runs** as "stable biomarkers"
- Minimum floor: always retain at least 30 features

Stability selection strongly reduces false discovery rates compared to a single LASSO fit and identifies proteins that are consistently informative rather than noise-exploiting.

**Tunable constants:**

| Constant | Default | Meaning |
|---|---|---|
| `STABILITY_N_ITER` | 150 | Bootstrap runs |
| `STABILITY_SUBSAMPLE` | 0.80 | Fraction of patients per run |
| `STABILITY_THRESH` | 0.50 | Selection frequency threshold |
| `STABILITY_C` | 0.05 | L1 regularisation strength |

### Stage 7 — Protein Co-expression Modules (WGCNA-lite)

Inspired by weighted gene co-expression network analysis (WGCNA), we cluster stable proteins into co-expression modules:

1. Compute the inter-protein Pearson correlation matrix (on stable proteins)
2. Apply hierarchical agglomerative clustering with average linkage
3. Cut dendrogram into **25 modules** (or fewer if stable protein count is low)
4. For each module, compute the **1st principal component** (module eigengene)

Module eigengenes capture co-expressed protein groups as single summary features, reducing collinearity while preserving biologically coherent signal.

### Stage 8 — Pairwise Interaction Features

For the **top-8 most stable proteins** (highest selection frequency), we compute:
- All pairwise **products** (ratio captures relative abundance relationships)
- All pairwise **ratios** (captures relative expression)

This generates ~56 additional non-linear features reflecting known proteomic interaction patterns relevant to kidney injury.

### Stage 9 — Clinical Feature Scaling

All clinical features (original + engineered) are scaled with `RobustScaler` (median/IQR normalisation), which is robust to the outliers common in clinical measurements.

### Stage 10 — Feature Assembly

Final feature matrix combines:
- Stable protein features (after all filtering)
- Module eigengene features (25 PCs)
- Interaction features (~56)
- Scaled clinical features (8)

**Typical final dimension**: 250–400 features from an original 6,590.

All preprocessing state (imputers, scalers, selection masks, PCA models) is serialised to `weights/preprocessor.pkl` for identical application to the test cohort.

---

## 5. Model Architecture

### Base Learners

Seven diverse base learners are trained on the preprocessed feature matrix, each targeting a different inductive bias:

| Model | Key Hyperparameters | Imbalance Strategy |
|---|---|---|
| **XGBoost** | `hist` booster, 600 trees, depth=4, lr=0.012, colsample=0.08 | `scale_pos_weight = n_neg/n_pos` |
| **LightGBM** | DART boosting, 600 iter, 24 leaves, lr=0.012, subsample=0.70 | `scale_pos_weight` |
| **CatBoost** | Ordered boosting, 600 iter, depth=4, l2=5.0 | `class_weights = [1.0, spw]` |
| **ExtraTrees** | 500 estimators, depth=8, `max_features="sqrt"`, min_leaf=20 | `class_weight="balanced"` |
| **MLP** | Layers=[256,64], ReLU, L2=2e-3, early stopping (25 iter) | `sample_weight` vector |
| **Elastic-Net LR** | C=0.04, l1_ratio=0.5, max_iter=4000 | `class_weight="balanced"` |
| **SGD LR** | `log_loss`, elastic-net, l1_ratio=0.15, α=1e-4 | `class_weight="balanced"` |

Each base learner is wrapped in **isotonic calibration** (cv=3) to produce reliable probability outputs before being passed to the meta-learner.

### Meta-Learner

```
StackingClassifier(
    estimators = [7 calibrated base learners],
    final_estimator = LogisticRegression(C=0.2, penalty='l2'),
    stack_method = 'predict_proba',
    cv = 5  (stratified)
)
```

The meta-learner is a Ridge-regularised logistic regression that learns optimal weights for combining base-learner probability outputs. Stacked generalisation via out-of-fold predictions prevents the meta-learner from simply memorising the training data.

### Post-hoc Calibration

After fitting the stacking ensemble, a **Platt (sigmoid) calibration** is applied using `CalibratedClassifierCV(cv='prefit')` on the full training set. This final calibration step corrects any residual overconfidence or underconfidence in the ensemble's probability outputs — critical for minimising log loss.

---

## 6. Training Strategy

### Class Imbalance

Class imbalance is handled **per-estimator** rather than via data augmentation (SMOTE/oversampling), which can introduce artefacts in high-dimensional proteomics data. Each model receives its imbalance correction natively:

```python
spw = n_negative / n_positive  # scale_pos_weight

# XGBoost / LightGBM: scale_pos_weight = spw
# CatBoost: class_weights = [1.0, spw]
# ExtraTrees / Elastic-Net / SGD LR: class_weight = "balanced"
# MLP: sample_weight = y * (spw - 1) + 1  (manual per-sample weights)
```

### Serialised Weights

After training on all 426 samples, the following artefacts are saved to `weights/`:

| File | Contents |
|---|---|
| `model.pkl` | Fitted stacking ensemble with Platt calibration |
| `preprocessor.pkl` | All fitted preprocessing state |
| `feature_cols.json` | Final feature column list |
| `xgboost_model.json` | Native XGBoost JSON format (compatibility) |

These are checked into `weights.zip` and are ready to use without re-training.

---

## 7. Results

> **Note**: The metrics below are computed on the **training set** using the fully-fitted model (in-sample). They are optimistic relative to true generalisation performance. Cross-validation metrics should be used to assess generalisation.

### Training-Set Performance (426 samples)

| Metric | Score |
|--------|-------|
| **Log Loss** | **0.4223** |
| **AUC-ROC** | **0.9695** |
| **Brier Score** | 0.1252 |
| Sensitivity (Recall) | 0.800 |
| Specificity | 0.951 |
| Precision | 0.936 |
| F1 Score | 0.863 |

### Confusion Matrix (Training Set, threshold = 0.5)

|  | Predicted No ATI | Predicted ATI |
|--|--|--|
| **True No ATI** (226) | 215 (TN) | 11 (FP) |
| **True ATI** (200) | 40 (FN) | 160 (TP) |

### Class Distribution

| Class | Count | Fraction |
|-------|-------|----------|
| ATI-positive | 200 | 46.9% |
| ATI-negative | 226 | 53.1% |

### Evaluation Outputs

Running `evaluate.py` generates the following artefacts in `results/`:

| File | Description |
|------|-------------|
| `cv_results.csv` | Per-fold metrics (AUC, log loss, Brier, ECE, AUPRC) |
| `cv_summary.csv` | Mean ± std across 5 folds |
| `cv_roc.png` | ROC curves per fold + mean |
| `cv_pr.png` | Precision-recall curves |
| `cv_calibration.png` | Reliability diagrams |
| `cv_logloss.png` | Model comparison bar chart |
| `cv_cm_*.png` | Per-fold confusion matrices |

---

## 8. Repository Structure

```
BKBC/
├── README.md              ← This file
├── requirements.txt       ← Python dependencies (119 packages)
│
├── preprocess.py          ← 10-stage feature engineering pipeline
├── model.py               ← Model definitions, ensemble factory, constants
├── train.py               ← Train on full dataset, serialise weights
├── evaluate.py            ← Stratified k-fold CV with full visualisation
├── predict.py             ← Inference on new/external data
├── predict.sh             ← Bash wrapper for easy prediction calls
│
├── predictions.csv        ← Pre-generated predictions on training set
│
├── weights.zip            ← Pre-trained model weights (ready-to-use)
│   └── (unpacked: weights/)
│       ├── model.pkl          ← Fitted stacking ensemble + Platt calibration
│       ├── preprocessor.pkl   ← Fitted preprocessing state
│       ├── feature_cols.json  ← Final feature column names
│       └── xgboost_model.json ← Native XGBoost JSON
│
└── results/               ← Generated by evaluate.py (not committed)
    ├── cv_results.csv
    ├── cv_summary.csv
    └── *.png
```

---

## 9. Quickstart

### Prerequisites

```bash
module load medaihack/spring-2026
module load python3/3.12.4
```

### First-Time Setup

```bash
# Clone or navigate to the BKBC directory
cd BKBC/

# Create and activate virtual environment
virtualenv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Unpack pre-trained weights
unzip weights.zip

# Verify installation
python model.py
```

### Run Predictions (No Re-training Needed)

Pre-trained weights are included. To generate predictions on new data immediately:

```bash
bash predict.sh /path/to/new_data.csv
# with custom output path:
bash predict.sh /path/to/new_data.csv my_predictions.csv
```

To evaluate on the training set (sanity check):

```bash
bash predict.sh /projectnb/medaihack/BKBC-hackathon/BKBC_train/train.csv
```

---

## 10. Detailed Usage

### Step 1 — Cross-Validation (Model Development)

Use `evaluate.py` to iterate on model development with stratified 5-fold cross-validation:

```bash
python evaluate.py --data /projectnb/medaihack/BKBC-hackathon/BKBC_train/train.csv

# Evaluate specific models:
python evaluate.py --data /path/to/train.csv --models Ensemble XGBoost ElasticNet

# Change number of folds:
python evaluate.py --data /path/to/train.csv --folds 10
```

**Output**: Per-fold and mean metrics (AUC, log loss, Brier, ECE, AUPRC), confusion matrices, ROC/PR/calibration plots saved to `results/`.

### Step 2 — Train Final Model

Once satisfied with cross-validation performance, train on all 426 samples:

```bash
python train.py --data /projectnb/medaihack/BKBC-hackathon/BKBC_train/train.csv

# Skip Platt post-hoc calibration:
python train.py --data /path/to/train.csv --no-platt

# Select a specific model (default: Ensemble):
python train.py --data /path/to/train.csv --model-name XGBoost
```

**Output**: Saves all weights to `weights/` directory.

### Step 3 — Predict on New Data

```bash
# Via bash wrapper (recommended):
bash predict.sh /path/to/new_data.csv [output.csv]

# Direct Python:
python predict.py --data /path/to/new_data.csv --out predictions.csv
```

If the input file contains an `ati` column, classification metrics are printed automatically.

**Output columns:**

| Column | Description |
|--------|-------------|
| `sample_id` | Patient identifier |
| `prob_ati` | Predicted probability of ATI (0–1) |
| `pred_label` | Hard prediction (0 = No ATI, 1 = ATI) |
| `true_label` | Ground truth (only if `ati` column present) |

### CLI Help

```bash
python train.py    --help
python evaluate.py --help
python predict.py  --help
```

---

## 11. Dependencies

Core dependencies (see `requirements.txt` for full pinned list of 119 packages):

| Package | Version | Role |
|---------|---------|------|
| `scikit-learn` | 1.8.0 | Ensemble, calibration, metrics, preprocessing |
| `xgboost` | 3.2.0 | Gradient boosting base learner |
| `lightgbm` | latest | DART boosting base learner |
| `catboost` | latest | Ordered boosting base learner |
| `numpy` | 2.4.4 | Numerical computation |
| `pandas` | 3.0.2 | Data I/O and manipulation |
| `scipy` | 1.17.1 | Statistics, clustering, distance metrics |
| `shap` | 0.51.0 | SHAP feature importance |
| `matplotlib` | 3.10.8 | Visualisation (ROC, PR, calibration plots) |

---

## 12. Research Notes

### Methodological Contributions

This submission makes several contributions beyond the baseline starter code:

1. **Advanced Biomarker Discovery via Stability Selection**: The 150-bootstrap stability selection procedure identifies proteins that are robustly and consistently associated with ATI, controlling the false discovery rate substantially better than a single LASSO fit. The selected proteins are candidates for validation as non-invasive ATI biomarkers.

2. **Protein Co-expression Module Features**: By computing module eigengenes via WGCNA-lite clustering, we capture coordinated proteomic responses — analogous to pathway-level analysis — rather than treating each protein independently. This reflects the biological reality that proteins act in networks, not in isolation.

3. **Per-Estimator Class Imbalance Correction**: Rather than globally oversampling (SMOTE) or undersampling, we inject class weights directly into each estimator's native mechanism. This avoids potential artefacts from synthetic sample generation in the high-dimensional (6,590-protein) input space.

4. **Dual Calibration for Log-Loss Optimisation**: The combination of isotonic calibration per base learner followed by post-hoc Platt calibration directly targets the log-loss evaluation metric. Well-calibrated probabilities are more clinically meaningful than raw discriminative scores.

5. **Rank-Based Inverse-Normal Transform**: This non-parametric normalisation approach is well-suited to proteomics data, where individual proteins can exhibit heavy-tailed or skewed distributions that violate the Gaussian assumptions of many downstream methods.

### Clinical Relevance

Acute tubular injury is a histological diagnosis requiring renal biopsy. A plasma proteomic classifier that can reliably predict ATI non-invasively could:
- Guide clinical decision-making about whether biopsy is warranted
- Enable real-time monitoring of AKI progression
- Identify mechanistic biomarkers of tubular injury for therapeutic development

The proteins selected by stability selection across bootstrap runs represent the most reproducibly informative candidates for further biological investigation and external validation.

### Potential Limitations

- **In-sample metrics**: All reported performance metrics reflect training-set performance. True generalisation performance will be assessed by the held-out test cohort.
- **Cohort specificity**: The BKBC cohort was recruited at a single centre. External validity to other institutions, ethnicities, or AKI aetiologies should be evaluated.
- **Model interpretability**: Stacking ensembles with 7 base learners are difficult to interpret clinically. SHAP values (`shap` library, integrated) can provide feature-level explanations for individual predictions.

---

*Submitted to MedAI Hackathon — Boston Kidney Biopsy Cohort (BKBC) Challenge*  
*April 2026*
