"""
Model pipelines and cross-validation for the Home Credit scorecard.

* Every pipeline = preprocessing (src/preprocessing.py) + estimator, so all learned steps are fitted on the
  training part of each fold only.
* Resampling (undersampling) is applied to the TRAINING indices of each fold only; validation folds are never
  resampled.
* Every comparison uses the same saved folds, so differences can be judged fold by fold (paired).
"""
from __future__ import annotations

import gc
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .evaluation import ranking_metrics
from .preprocessing import lr_preprocessor, tree_categorical_mask, tree_preprocessor

SEED = 42


def restrict(cols: dict, keep: set) -> dict:
    """Column lists from preprocessing.decide_columns, limited to the features in `keep`."""
    return {k: [c for c in v if c in keep] for k, v in cols.items() if k != "audit"}


def feature_set(cols: dict) -> set:
    return set(cols["continuous"]) | set(cols["binary"]) | set(cols["categorical"])


def make_lr(cols: dict, C: float = 1.0, class_weight=None) -> Pipeline:
    """Logistic Regression with the Step 4 preprocessing (variant D: impute, winsorise, log, bins, scale, one-hot)."""
    return Pipeline([("prep", lr_preprocessor(cols, clip=True, log=True, bins=True)),
                     ("model", LogisticRegression(C=C, class_weight=class_weight, max_iter=5000))])


def make_hgb(cols: dict, class_weight=None, **params) -> Pipeline:
    """Histogram gradient boosting with native missing-value and categorical handling."""
    return Pipeline([("prep", tree_preprocessor(cols)),
                     ("model", HistGradientBoostingClassifier(categorical_features=tree_categorical_mask(cols),
                                                              class_weight=class_weight, random_state=SEED,
                                                              **params))])


def undersample(train_idx: np.ndarray, y: np.ndarray, neg_per_pos: float, seed: int) -> np.ndarray:
    """Keep every positive and a random subset of negatives (training indices only)."""
    rng = np.random.default_rng(seed)
    pos = train_idx[y[train_idx] == 1]
    neg = train_idx[y[train_idx] == 0]
    keep = rng.choice(neg, size=min(len(neg), int(neg_per_pos * len(pos))), replace=False)
    return np.sort(np.concatenate([pos, keep]))


def _n_iter(pipe) -> int | None:
    it = getattr(pipe[-1], "n_iter_", None)
    return None if it is None else int(np.max(it))


def cross_validate(make_pipe, X: pd.DataFrame, y: np.ndarray, folds: list, label: str,
                   fold_ids=None, undersample_ratio: float | None = None):
    """Fit `make_pipe()` on each training fold, score the validation fold.
    Returns (per-fold metrics DataFrame, out-of-fold prediction array with NaN for folds not run)."""
    rows, oof = [], np.full(len(y), np.nan)
    for k, (tri, vai) in enumerate(folds):
        if fold_ids is not None and k not in fold_ids:
            continue
        if undersample_ratio:
            tri = undersample(tri, y, undersample_ratio, SEED + k)
        t = time.time()
        pipe = make_pipe().fit(X.iloc[tri], y[tri])
        fit_s = time.time() - t
        t = time.time()
        p = pipe.predict_proba(X.iloc[vai])[:, 1]
        pred_s = time.time() - t
        oof[vai] = p
        m = ranking_metrics(y[vai], p)
        m.update(label=label, fold=k, fit_seconds=fit_s, predict_seconds_per_1k=1000 * pred_s / len(vai),
                 n_iter=_n_iter(pipe), train_rows=len(tri))
        rows.append(m)
        del pipe
        gc.collect()
    return pd.DataFrame(rows), oof


def paired_summary(results: pd.DataFrame, ref: str, labels: list | None = None) -> pd.DataFrame:
    """Mean metrics per label plus the fold-by-fold ROC-AUC difference against `ref` (common folds only)."""
    labels = labels or list(dict.fromkeys(results.label))
    base = results[results.label == ref].set_index("fold")
    rows = []
    for lab in labels:
        g = results[results.label == lab].set_index("fold")
        common = g.index.intersection(base.index)
        diff = g.loc[common, "roc_auc"] - base.loc[common, "roc_auc"]
        rows.append({"label": lab, "folds": len(g), "roc_auc": g.roc_auc.mean(), "roc_auc_std": g.roc_auc.std(),
                     "pr_auc": g.pr_auc.mean(), "ks": g.ks.mean(), "brier": g.brier.mean(),
                     "log_loss": g.log_loss.mean(), "auc_diff_vs_ref": diff.mean(),
                     "folds_better_than_ref": int((diff > 0).sum()), "common_folds": len(common),
                     "fit_seconds": g.fit_seconds.mean()})
    return pd.DataFrame(rows).set_index("label")
