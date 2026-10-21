"""
Evaluation helpers for the Home Credit scorecard.

Threshold-independent metrics (ranking and probability quality) are kept separate from threshold-dependent
ones (confusion matrix, precision, recall, F1), because a credit decision threshold is a business choice (Step 10).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, log_loss,
                             precision_recall_curve, roc_auc_score)


def ks_statistic(y, p) -> float:
    """Kolmogorov-Smirnov distance between the score distributions of TARGET=1 and TARGET=0."""
    y, p = np.asarray(y), np.asarray(p)
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def ranking_metrics(y, p) -> dict:
    """Threshold-independent metrics."""
    y, p = np.asarray(y), np.asarray(p)
    auc = roc_auc_score(y, p)
    return {"roc_auc": auc, "gini": 2 * auc - 1, "pr_auc": average_precision_score(y, p),
            "ks": ks_statistic(y, p), "brier": brier_score_loss(y, p),
            "log_loss": log_loss(y, np.clip(p, 1e-15, 1 - 1e-15))}


def threshold_metrics(y, p, threshold: float) -> dict:
    """Threshold-dependent metrics; 'flagged' = predicted high risk (p >= threshold)."""
    y = np.asarray(y)
    pred = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = 2 * precision * recall / (precision + recall) if (tp and (precision + recall)) else 0.0
    return {"threshold": threshold, "flagged_pct": 100 * pred.mean(), "tp": int(tp), "fp": int(fp),
            "tn": int(tn), "fn": int(fn), "precision": precision, "recall": recall, "f1": f1}


def f1_optimal_threshold(y, p) -> float:
    """Threshold maximising F1 - a statistical reference only, not a lending policy."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[int(np.nanargmax(f1[:-1]))])


def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed rate by quantile bin of the predicted probability."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df.p, n_bins, labels=False, duplicates="drop")
    t = df.groupby("bin").agg(n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
    t["gap"] = t.observed_rate - t.mean_predicted
    return t


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    t = calibration_table(y, p, n_bins)
    return float((t.n * t.gap.abs()).sum() / t.n.sum())


def cv_isotonic(oof, y, fold) -> np.ndarray:
    """Cross-validated isotonic calibration of out-of-fold predictions: for each fold the calibrator is fitted
    on the OTHER folds' predictions only, so the calibrated values are themselves out-of-sample."""
    oof, y, fold = np.asarray(oof), np.asarray(y), np.asarray(fold)
    out = np.empty_like(oof, dtype="float64")
    for k in np.unique(fold):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        iso.fit(oof[fold != k], y[fold != k])
        out[fold == k] = iso.predict(oof[fold == k])
    return out
