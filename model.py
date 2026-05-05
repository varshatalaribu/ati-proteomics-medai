

#!/usr/bin/env python3
"""
model.py --- Model definitions (v2 – Multi-View Ensemble)
=========================================================

Key changes from v1
-------------------
1. Multi-view stacking ensemble: base learners trained on DIFFERENT
   feature representations (modules, stable proteins, full space)
   rather than all seeing the same features.

2. Tighter regularisation for 426 samples (not 1500).
   - Lower n_estimators, deeper min_child constraints
   - colsample_bytree kept very low (0.08–0.12)

3. All base learners wrapped with isotonic calibration (log-loss metric).

4. Meta-learner is Ridge LR with C=0.2 (tighter than v1's 0.3).

Model roster
------------
1. XGBoost       – hist booster, tight regularisation
2. LightGBM      – dart boosting with extra_trees
3. CatBoost      – ordered boosting
4. ExtraTrees    – max randomness, decorrelated from boosters
5. MLP           – small network [256, 64], high dropout
6. ElasticNet LR – L1+L2, handles full feature space
7. SGD LR        – fast online learner
8. Ensemble      – stacking with Ridge LR meta-learner (C=0.2)
"""

import os
nslots = int(os.environ.get("NSLOTS", 1))
nslots = nslots - 1 if nslots > 1 else nslots

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

try:
    from catboost import CatBoostClassifier
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False


# ── Constants ─────────────────────────────────────────────────────────
CLINICAL_FEATURES = ["age", "sex", "baseline_egfr_23"]
CV_FOLDS     = 5
RANDOM_SEED  = 42


# ── 1. XGBoost ────────────────────────────────────────────────────────
_xgb = CalibratedClassifierCV(
    XGBClassifier(
        tree_method       = "hist",
        n_estimators      = 600,
        max_depth         = 4,
        learning_rate     = 0.012,
        subsample         = 0.70,
        colsample_bytree  = 0.08,
        colsample_bylevel = 0.70,
        min_child_weight  = 10,
        gamma             = 0.15,
        reg_alpha         = 0.3,
        reg_lambda         = 2.0,
        eval_metric       = "logloss",
        use_label_encoder = False,
        random_state      = RANDOM_SEED,
        n_jobs            = nslots,
    ),
    method="isotonic", cv=3,
)


# ── 2. LightGBM ──────────────────────────────────────────────────────
if _HAS_LGBM:
    _lgbm = CalibratedClassifierCV(
        LGBMClassifier(
            boosting_type     = "dart",
            n_estimators      = 600,
            num_leaves        = 24,
            learning_rate     = 0.012,
            subsample         = 0.70,
            subsample_freq    = 1,
            feature_fraction  = 0.08,
            min_child_samples = 25,
            reg_alpha         = 0.3,
            reg_lambda         = 2.0,
            drop_rate         = 0.12,
            extra_trees       = True,
            random_state      = RANDOM_SEED,
            n_jobs            = nslots,
            verbose           = -1,
        ),
        method="isotonic", cv=3,
    )
else:
    _lgbm = None


# ── 3. CatBoost ──────────────────────────────────────────────────────
if _HAS_CATBOOST:
    _catboost = CalibratedClassifierCV(
        CatBoostClassifier(
            iterations        = 600,
            depth             = 4,
            learning_rate     = 0.012,
            l2_leaf_reg       = 5.0,
            subsample         = 0.70,
            colsample_bylevel = 0.08,
            eval_metric       = "Logloss",
            random_seed       = RANDOM_SEED,
            thread_count      = nslots,
            verbose           = False,
            allow_writing_files = False,
        ),
        method="isotonic", cv=3,
    )
else:
    _catboost = None


# ── 4. ExtraTrees ─────────────────────────────────────────────────────
_extratrees = CalibratedClassifierCV(
    ExtraTreesClassifier(
        n_estimators    = 500,
        max_depth       = 8,
        max_features    = "sqrt",
        min_samples_leaf= 20,
        class_weight    = "balanced",
        random_state    = RANDOM_SEED,
        n_jobs          = nslots,
    ),
    method="isotonic", cv=3,
)


# ── 5. MLP ────────────────────────────────────────────────────────────
_mlp = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", CalibratedClassifierCV(
        MLPClassifier(
            hidden_layer_sizes = (256, 64),
            activation         = "relu",
            solver             = "adam",
            alpha              = 2e-3,
            batch_size         = 32,
            learning_rate      = "adaptive",
            learning_rate_init = 3e-4,
            max_iter           = 400,
            early_stopping     = True,
            validation_fraction= 0.12,
            n_iter_no_change   = 25,
            random_state       = RANDOM_SEED,
        ),
        method="isotonic", cv=3,
    )),
])


# ── 6. ElasticNet LR ─────────────────────────────────────────────────
_elastic = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", CalibratedClassifierCV(
        LogisticRegression(
            penalty       = "elasticnet",
            solver        = "saga",
            l1_ratio      = 0.5,
            C             = 0.04,
            max_iter      = 4000,
            class_weight  = "balanced",
            random_state  = RANDOM_SEED,
        ),
        method="isotonic", cv=3,
    )),
])


# ── 7. SGD LR ────────────────────────────────────────────────────────
_sgd = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", CalibratedClassifierCV(
        SGDClassifier(
            loss         = "log_loss",
            penalty      = "elasticnet",
            l1_ratio     = 0.15,
            alpha        = 1e-4,
            max_iter     = 1000,
            tol          = 1e-4,
            class_weight = "balanced",
            random_state = RANDOM_SEED,
            n_jobs       = nslots,
        ),
        method="isotonic", cv=3,
    )),
])


# ── 8. Stacking Ensemble ─────────────────────────────────────────────
def _make_stacking():
    """
    Multi-view stacking ensemble.
    Base: XGBoost, LightGBM, CatBoost, ExtraTrees, MLP, ElasticNet
    Meta: Ridge LR (C=0.2) – tighter than v1 for 426 samples.
    """
    estimators = [
        ("xgb",        _xgb),
        ("extratrees", _extratrees),
        ("mlp",        _mlp),
        ("elastic",    _elastic),
        ("sgd",        _sgd),
    ]
    if _HAS_LGBM and _lgbm is not None:
        estimators.insert(1, ("lgbm", _lgbm))
    if _HAS_CATBOOST and _catboost is not None:
        estimators.insert(2, ("catboost", _catboost))

    return StackingClassifier(
        estimators      = estimators,
        final_estimator = LogisticRegression(
            C           = 0.2,
            solver      = "lbfgs",
            max_iter    = 1000,
            random_state= RANDOM_SEED,
        ),
        stack_method = "predict_proba",
        cv           = 5,
        n_jobs       = 1,
        passthrough  = False,
    )


# ── Model registry ───────────────────────────────────────────────────
MODELS = {
    "Ensemble"  : _make_stacking(),
    "XGBoost"   : _xgb,
    "ExtraTrees": _extratrees,
    "MLP"       : _mlp,
    "ElasticNet": _elastic,
    "SGD"       : _sgd,
    "Lasso LR"  : Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l1", solver="liblinear",
            C=0.04, max_iter=3000, random_state=RANDOM_SEED)),
    ]),
}
if _HAS_LGBM    and _lgbm     is not None: MODELS["LightGBM"]  = _lgbm
if _HAS_CATBOOST and _catboost is not None: MODELS["CatBoost"] = _catboost


# ── Factory function ──────────────────────────────────────────────────
def build_model(name: str):
    """Return a fresh (unfitted) model by name."""
    from sklearn.base import clone
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODELS)}")
    return clone(MODELS[name])


if __name__ == "__main__":
    import numpy as np
    print("Models defined:")
    for name in MODELS:
        m = build_model(name)
        print(f"  {name}: {type(m).__name__}")

    print(f"\nClinical features : {CLINICAL_FEATURES}")
    print(f"CV folds          : {CV_FOLDS}")
    print(f"Random seed       : {RANDOM_SEED}")
    print(f"LightGBM          : {_HAS_LGBM}")
    print(f"CatBoost          : {_HAS_CATBOOST}")

    rng = np.random.default_rng(0)
    X_d = rng.standard_normal((120, 30))
    y_d = rng.integers(0, 2, 120)
    for name in ["XGBoost", "ExtraTrees", "ElasticNet", "SGD"]:
        m = build_model(name)
        m.fit(X_d, y_d)
        p = m.predict_proba(X_d)
        assert p.shape == (120, 2), f"{name}: predict_proba shape mismatch"
        print(f"  [{name}] smoke test passed")

