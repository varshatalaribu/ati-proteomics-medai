#!/usr/bin/env python3
"""
preprocess.py --- Data loading and feature engineering (v2 – Advanced)
======================================================================

Key improvements over v1
------------------------
1. Protein co-expression modules (WGCNA-lite)
   Hierarchical clustering on protein correlation -> ~25 modules.
   Each patient gets a module eigenvector score (1st PC per module).

2. Stability selection (replaces single ANOVA pass)
   L1-logistic on 150 random 80%-subsamples. Proteins selected in >=50%
   of runs are stable biomarkers.

3. Rank-based inverse-normal transform
   Rank -> U(0,1) -> inverse-normal per protein. Removes outliers cleanly.

4. Top-protein interaction features
   Pairwise ratios and products of the 8 most stable proteins.

5. Extra clinical features
   eGFR squared, age bins, age x eGFR, log(eGFR), CKD stage.
"""

import logging
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, StandardScaler
from model import CLINICAL_FEATURES

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

# ── Tunable constants ─────────────────────────────────────────────────
VAR_THRESHOLD = 0.01
ANOVA_K = 3500
STABILITY_N_ITER = 150
STABILITY_SUBSAMPLE = 0.80
STABILITY_THRESH = 0.50
STABILITY_C = 0.05
MODULE_N_CLUSTERS = 25
MODULE_MIN_SIZE = 5
N_INTERACT_PROTEINS = 8

USE_MODULES = True
USE_STABILITY = True
USE_INTERACTIONS = True
USE_RANK_NORM = True


# ═══════════════════════════════════════════════════════════════════════
# Clinical feature engineering
# ═══════════════════════════════════════════════════════════════════════

def _egfr_stage(egfr):
    bins = [-np.inf, 15, 30, 45, 60, 90, np.inf]
    labels = [0, 1, 2, 3, 4, 5]
    return pd.cut(egfr, bins=bins, labels=labels).astype(float)


def _add_clinical_features(df):
    df = df.copy()
    if "age" in df.columns and "baseline_egfr_23" in df.columns:
        df["age_x_egfr"] = df["age"] * df["baseline_egfr_23"]
    if "baseline_egfr_23" in df.columns:
        df["log_egfr"] = np.log1p(df["baseline_egfr_23"].clip(lower=0))
        df["egfr_stage"] = _egfr_stage(df["baseline_egfr_23"])
        df["egfr_sq"] = df["baseline_egfr_23"] ** 2
    if "age" in df.columns:
        df["age_bin"] = pd.cut(
            df["age"], bins=[0, 30, 45, 60, 75, 200],
            labels=[0, 1, 2, 3, 4]
        ).astype(float)
    return df


# ═══════════════════════════════════════════════════════════════════════
# Rank-based inverse-normal transform
# ═══════════════════════════════════════════════════════════════════════

def _rank_inv_norm_fit(X):
    n = X.shape[0]
    X_out = np.empty_like(X)
    col_stats = []
    for j in range(X.shape[1]):
        col = X[:, j]
        ranked = stats.rankdata(col, method='average') / (n + 1)
        X_out[:, j] = stats.norm.ppf(ranked)
        col_stats.append(
            {'mean': np.nanmean(col), 'std': np.nanstd(col) + 1e-10})
    return np.clip(X_out, -4, 4), col_stats


def _rank_inv_norm_transform(X, col_stats):
    X_out = np.empty_like(X)
    for j in range(X.shape[1]):
        z = (X[:, j] - col_stats[j]['mean']) / col_stats[j]['std']
        X_out[:, j] = np.clip(z, -4, 4)
    return X_out


# ═══════════════════════════════════════════════════════════════════════
# Stability Selection
# ═══════════════════════════════════════════════════════════════════════

def _stability_selection(X, y, n_iter=150, subsample=0.80, C=0.05,
                         threshold=0.50, seed=42):
    rng = np.random.RandomState(seed)
    n, p = X.shape
    counts = np.zeros(p)
    n_sub = int(n * subsample)
    Xs = StandardScaler().fit_transform(X)

    for i in range(n_iter):
        idx = rng.choice(n, n_sub, replace=False)
        try:
            lr = LogisticRegression(
                penalty='l1', solver='liblinear', C=C,
                max_iter=2000, random_state=i, class_weight='balanced')
            lr.fit(Xs[idx], y[idx])
            counts += (np.abs(lr.coef_[0]) > 1e-8)
        except Exception:
            continue

    freq = counts / n_iter
    mask = freq >= threshold
    if mask.sum() < 30:
        mask[np.argsort(freq)[-30:]] = True
    logging.info(
        f"Stability selection: {mask.sum()}/{p} features (>={threshold*100:.0f}% of {n_iter} runs)")
    return mask, freq


# ═══════════════════════════════════════════════════════════════════════
# Protein Co-expression Modules (WGCNA-lite)
# ═══════════════════════════════════════════════════════════════════════

def _build_modules(X, n_clusters=25, min_size=5):
    p = X.shape[1]
    if p < n_clusters * 2:
        n_clusters = max(5, p // 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = np.corrcoef(X.T)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    dist = np.clip(1 - np.abs(corr), 0, 2)
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist, checks=False), method='ward')
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')

    pcas, members = {}, {}
    for m in sorted(set(labels)):
        idx = np.where(labels == m)[0]
        if len(idx) < min_size:
            continue
        pca = PCA(n_components=1, random_state=42)
        pca.fit(X[:, idx])
        pcas[m] = pca
        members[m] = idx
    logging.info(f"Built {len(pcas)} protein modules from {p} proteins")
    return pcas, members


def _module_scores(X, pcas, members):
    scores, names = [], []
    for m, pca in sorted(pcas.items()):
        scores.append(pca.transform(X[:, members[m]])[:, 0])
        names.append(f"mod_{m}")
    if scores:
        return np.column_stack(scores), names
    return np.empty((X.shape[0], 0)), []


# ═══════════════════════════════════════════════════════════════════════
# Interaction Features
# ═══════════════════════════════════════════════════════════════════════

def _interaction_features(X, top_idx, names_list):
    feats, fnames = [], []
    for i in range(len(top_idx)):
        for j in range(i + 1, len(top_idx)):
            a, b = top_idx[i], top_idx[j]
            na = names_list[a] if a < len(names_list) else f"f{a}"
            nb = names_list[b] if b < len(names_list) else f"f{b}"
            feats.append(X[:, a] * X[:, b])
            fnames.append(f"ix_{na}_x_{nb}")
            denom = X[:, b].copy()
            denom[np.abs(denom) < 1e-6] = 1e-6
            feats.append(np.clip(X[:, a] / denom, -10, 10))
            fnames.append(f"ix_{na}_d_{nb}")
    if feats:
        return np.column_stack(feats), fnames
    return np.empty((X.shape[0], 0)), []


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════

def load_data(path):
    logging.info(f"Loading data from {path}...")
    df = pd.read_csv(path, low_memory=False, na_values=[".", ""])
    logging.info(f"  {len(df)} samples, {len(df.columns)} columns")
    return df


def build_features_and_labels(df, fit=True, preproc_state=None):
    # ── 1. Clinical engineering ────────────────────────────────────────
    df = _add_clinical_features(df)
    derived = ["age_x_egfr", "log_egfr", "egfr_stage", "egfr_sq", "age_bin"]
    ext_clin = CLINICAL_FEATURES + [c for c in derived if c in df.columns]

    # ── 2. Column lists ────────────────────────────────────────────────
    prot_cols = sorted([c for c in df.columns if c.startswith("feature_")])
    all_cols = prot_cols + ext_clin if fit else preproc_state["all_cols"]
    avail = [c for c in all_cols if c in df.columns]

    # ── 3. Label ───────────────────────────────────────────────────────
    has_y = "ati" in df.columns
    if has_y:
        work = df[avail + ["ati"]].copy().dropna(subset=["ati"])
        y = work["ati"].astype(int).values
    else:
        work = df[avail].copy()
        y = None

    X_raw = work[avail].values.astype(float)
    n_prot = len([c for c in avail if c.startswith("feature_")])
    prot_nm = [c for c in avail if c.startswith("feature_")]

    # ── 4. Imputation ──────────────────────────────────────────────────
    if fit:
        med_imp = SimpleImputer(strategy="median")
        X_raw[:, :n_prot] = med_imp.fit_transform(X_raw[:, :n_prot])
        knn_imp = KNNImputer(n_neighbors=5, weights="distance")
        X_raw[:, n_prot:] = knn_imp.fit_transform(X_raw[:, n_prot:])
    else:
        med_imp = preproc_state["med_imp"]
        knn_imp = preproc_state["knn_imp"]
        X_raw[:, :n_prot] = med_imp.transform(X_raw[:, :n_prot])
        X_raw[:, n_prot:] = knn_imp.transform(X_raw[:, n_prot:])

    prot_block = X_raw[:, :n_prot]
    clin_block = X_raw[:, n_prot:]

    # ── 5. Rank-based inverse-normal ───────────────────────────────────
    if USE_RANK_NORM:
        if fit:
            prot_block, rnk_stats = _rank_inv_norm_fit(prot_block)
        else:
            rnk_stats = preproc_state["rnk_stats"]
            prot_block = _rank_inv_norm_transform(prot_block, rnk_stats)
    else:
        rnk_stats = None
        if fit:
            p_lo = np.percentile(prot_block, 1.0, axis=0)
            p_hi = np.percentile(prot_block, 99.0, axis=0)
        else:
            p_lo, p_hi = preproc_state["p_lo"], preproc_state["p_hi"]
        prot_block = np.clip(prot_block, p_lo, p_hi)

    # ── 6. Variance filter ─────────────────────────────────────────────
    if fit:
        var_mask = prot_block.var(axis=0) >= VAR_THRESHOLD
        logging.info(f"Variance filter: {var_mask.sum()}/{n_prot} kept")
    else:
        var_mask = preproc_state["var_mask"]
    prot_filt = prot_block[:, var_mask]
    prot_nm_f = [n for n, k in zip(prot_nm, var_mask) if k]

    # ── 7. ANOVA pre-filter ────────────────────────────────────────────
    if ANOVA_K is not None and fit and y is not None:
        k = min(ANOVA_K, prot_filt.shape[1])
        anova_sel = SelectKBest(f_classif, k=k)
        prot_filt = anova_sel.fit_transform(prot_filt, y)
        anova_mask = anova_sel.get_support()
        prot_nm_a = [n for n, k2 in zip(prot_nm_f, anova_mask) if k2]
        logging.info(f"ANOVA filter: top {k}/{var_mask.sum()} proteins")
    elif not fit:
        anova_sel = preproc_state["anova_sel"]
        anova_mask = preproc_state["anova_mask"]
        prot_nm_a = preproc_state["prot_nm_a"]
        if anova_sel is not None:
            prot_filt = anova_sel.transform(prot_filt)
    else:
        anova_sel = None
        anova_mask = np.ones(prot_filt.shape[1], dtype=bool)
        prot_nm_a = prot_nm_f

    # ── 8. Stability selection ─────────────────────────────────────────
    if USE_STABILITY:
        if fit and y is not None:
            stab_mask, stab_freq = _stability_selection(
                prot_filt, y, STABILITY_N_ITER, STABILITY_SUBSAMPLE,
                STABILITY_C, STABILITY_THRESH)
            stable_idx = np.where(stab_mask)[0]
        else:
            stab_mask = preproc_state["stab_mask"]
            stab_freq = preproc_state["stab_freq"]
            stable_idx = np.where(stab_mask)[0]
    else:
        stab_mask = np.ones(prot_filt.shape[1], dtype=bool)
        stab_freq = np.ones(prot_filt.shape[1])
        stable_idx = np.arange(prot_filt.shape[1])

    # ── 9. Module features ─────────────────────────────────────────────
    if USE_MODULES:
        if fit:
            mod_pcas, mod_members = _build_modules(
                prot_filt, MODULE_N_CLUSTERS, MODULE_MIN_SIZE)
        else:
            mod_pcas = preproc_state["mod_pcas"]
            mod_members = preproc_state["mod_members"]
        mod_X, mod_names = _module_scores(prot_filt, mod_pcas, mod_members)
    else:
        mod_X, mod_names = np.empty((prot_filt.shape[0], 0)), []
        mod_pcas, mod_members = {}, {}

    # ── 10. Interaction features ───────────────────────────────────────
    if USE_INTERACTIONS and len(stable_idx) >= 2:
        if fit:
            freq_stable = stab_freq[stable_idx]
            top_n = min(N_INTERACT_PROTEINS, len(stable_idx))
            top_local = np.argsort(freq_stable)[-top_n:]
            interact_idx = stable_idx[top_local]
        else:
            interact_idx = preproc_state["interact_idx"]
        ix_X, ix_names = _interaction_features(
            prot_filt, interact_idx, prot_nm_a)
    else:
        ix_X, ix_names = np.empty((prot_filt.shape[0], 0)), []
        interact_idx = np.array([], dtype=int)

    # ── 11. Scale clinical ─────────────────────────────────────────────
    if fit:
        clin_scaler = RobustScaler()
        clin_scaled = clin_scaler.fit_transform(clin_block)
    else:
        clin_scaler = preproc_state["clin_scaler"]
        clin_scaled = clin_scaler.transform(clin_block)

    # ── 12. Assemble ───────────────────────────────────────────────────
    clin_names = list(avail[n_prot:]) if isinstance(avail, list) else [
        avail[i] for i in range(n_prot, len(avail))]
    X = np.hstack([prot_filt, mod_X, ix_X, clin_scaled])
    feat_names = prot_nm_a + mod_names + ix_names + clin_names

    if y is not None:
        logging.info(
            f"Samples: {len(y)} | No ATI: {(y == 0).sum()} | ATI: {(y == 1).sum()}")
    logging.info(
        f"Final: {prot_filt.shape[1]} proteins + {mod_X.shape[1]} modules "
        f"+ {ix_X.shape[1]} interactions + {clin_scaled.shape[1]} clinical "
        f"= {X.shape[1]} features"
    )

    # ── 13. Pack state ─────────────────────────────────────────────────
    if fit:
        preproc_state = {
            "all_cols":     all_cols,
            "med_imp":      med_imp,
            "knn_imp":      knn_imp,
            "rnk_stats":    rnk_stats if USE_RANK_NORM else None,
            "p_lo":         None if USE_RANK_NORM else p_lo,
            "p_hi":         None if USE_RANK_NORM else p_hi,
            "var_mask":     var_mask,
            "anova_sel":    anova_sel,
            "anova_mask":   anova_mask,
            "prot_nm_a":    prot_nm_a,
            "stab_mask":    stab_mask,
            "stab_freq":    stab_freq,
            "mod_pcas":     mod_pcas,
            "mod_members":  mod_members,
            "interact_idx": interact_idx,
            "clin_scaler":  clin_scaler,
            "feature_cols": feat_names,
            "n_prot_raw":   n_prot,
        }

    return X, y, feat_names, preproc_state
