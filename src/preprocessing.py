"""
Step 4 - Preprocessing for the Home Credit scorecard.

Two kinds of decisions live here:

1. Column-set rules (decided once on the DEVELOPMENT folds, never on the holdout):
   * identifiers / target / fold are never features;
   * CODE_GENDER is excluded by policy (kept in the data for the Step 11 fairness check);
   * drift-risk features are excluded by default because their reporting differs drastically between
     train and scoring data: monthly-bureau features (BB_*) and bureau-annuity features; Step 9 tests them;
   * near-constant columns with < MIN_MINORITY minority cases are dropped;
   * AVG/MODE/MEDI building triplets keep only the AVG version;
   * for Logistic Regression only, one feature of each highly correlated pair (|Spearman| >= 0.95)
     is dropped, keeping the one with the higher development ROC-AUC.

2. Learned transformations (imputation, clipping, binning, encoding, scaling): implemented as
   scikit-learn transformers inside a Pipeline, so they are fitted on the training part of each
   fold only and merely applied to validation / holdout / scoring rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer, OneHotEncoder, OrdinalEncoder, StandardScaler

NON_FEATURES = ["SK_ID_CURR", "TARGET", "fold"]
SENSITIVE_EXCLUDED = ["CODE_GENDER"]
_BB = "monthly bureau status available for 35% of train vs 99.98% of scoring applicants with bureau records"
_ANN = "bureau annuity missing for 75.9% of active credits in train vs 22.7% in scoring (reporting drift)"
DRIFT_RISK = {"BB_N_MONTHS_KNOWN": _BB, "BB_WORST_STATUS": _BB, "BB_WORST_STATUS_12M": _BB,
              "BB_SHARE_MONTHS_DPD": _BB, "HAS_BUREAU_BALANCE": _BB,
              "BUR_ACTIVE_ANNUITY_SUM": _ANN, "X_TOTAL_ANNUITY_TO_INCOME": _ANN + "; built on BUR_ACTIVE_ANNUITY_SUM"}
MIN_MINORITY = 30          # near-constant rule: fewer minority cases than this -> no reliable signal
CORR_THRESHOLD = 0.95      # redundancy rule for Logistic Regression
SKEW_THRESHOLD = 2.0       # |skew| above this -> signed log1p for Logistic Regression
NONMONOTONIC = ["APP_CREDIT_TERM", "APP_CREDIT_TO_INCOME", "APP_CREDIT_TO_GOODS", "APP_ANNUITY_TO_INCOME",
                "INST_MAX_DAYS_LATE", "BUR_ACTIVE_DEBT_RATIO"]   # Step 3 showed non-linear risk curves


# --------------------------------------------------------------------------- folds
def predefined_folds(fold: pd.Series):
    """CV iterator reproducing the saved fold assignment (development rows only, fold >= 0)."""
    fold = fold.to_numpy()
    return [(np.flatnonzero(fold != k), np.flatnonzero(fold == k)) for k in np.unique(fold)]


# --------------------------------------------------------------------------- column rules
def is_text(s: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def univariate_auc(x: pd.Series, y: pd.Series) -> float:
    m = x.notna()
    if m.sum() < 1000 or x[m].nunique() < 2:
        return np.nan
    a = roc_auc_score(y[m], x[m])
    return max(a, 1 - a)


def decide_columns(dev: pd.DataFrame, include_drift: bool = False, seed: int = 42) -> dict:
    """Apply the column-set rules on development data. Returns lists plus an audit log of every drop."""
    log = []
    cols = [c for c in dev.columns if c not in NON_FEATURES]
    for c in SENSITIVE_EXCLUDED:
        log.append((c, "excluded by policy (sensitive attribute; kept for fairness analysis)"))
    if not include_drift:
        for c, why in DRIFT_RISK.items():
            log.append((c, f"excluded by default (drift risk): {why}; tested in Step 9"))
    drop = set(SENSITIVE_EXCLUDED) | (set() if include_drift else set(DRIFT_RISK))

    for c in cols:
        if c.endswith(("_MODE", "_MEDI")) and f"{c.rsplit('_', 1)[0]}_AVG" in dev.columns:
            drop.add(c); log.append((c, "building triplet: AVG version kept"))

    for c in cols:
        if c in drop:
            continue
        vc = dev[c].value_counts(dropna=False)
        if len(vc) <= 1 or (vc.iloc[1:].sum() < MIN_MINORITY):
            drop.add(c); log.append((c, f"near-constant: {int(vc.iloc[1:].sum())} minority cases < {MIN_MINORITY}"))

    keep = [c for c in cols if c not in drop]
    cat = [c for c in keep if is_text(dev[c])]
    num = [c for c in keep if c not in cat]
    binary = [c for c in num if dev[c].dropna().nunique() <= 2]
    cont = [c for c in num if c not in binary]

    # Redundancy (Logistic Regression only): greedy over pairs, strongest correlation first
    sample = dev[cont].sample(min(60_000, len(dev)), random_state=seed).astype("float64")
    corr = sample.corr(method="spearman")
    pairs = (corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1)).stack().rename("rho").reset_index())
    pairs = pairs[pairs.rho.abs() >= CORR_THRESHOLD].sort_values("rho", key=abs, ascending=False)
    auc = {c: univariate_auc(dev[c], dev.TARGET) for c in set(pairs.level_0) | set(pairs.level_1)}
    lr_drop = set()
    for a, b, rho in pairs.itertuples(index=False):
        if a in lr_drop or b in lr_drop:
            continue
        ra, rb = np.nan_to_num(auc[a]), np.nan_to_num(auc[b])
        if abs(ra - rb) < 1e-6:          # identical ranking power: keep the interpretable engineered unit
            loser = b if a.startswith(("APP_", "X_")) and not b.startswith(("APP_", "X_")) else a
        else:
            loser = a if ra < rb else b
        winner = b if loser == a else a
        lr_drop.add(loser)
        why = ("identical ranking power; kept the interpretable engineered unit" if abs(ra - rb) < 1e-6
               else f"kept higher dev AUC ({auc[winner]:.4f} vs {auc[loser]:.4f})")
        log.append((loser, f"LR only: |rho|={abs(rho):.3f} with {winner}; {why}"))
    lr_cont = [c for c in cont if c not in lr_drop]
    skewed = [c for c in lr_cont if abs(sample[c].skew()) > SKEW_THRESHOLD]
    return {"categorical": cat, "binary": binary, "continuous": cont, "lr_continuous": lr_cont,
            "lr_skewed": skewed, "nonmonotonic": [c for c in NONMONOTONIC if c in lr_cont],
            "audit": pd.DataFrame(log, columns=["column", "reason"])}


# --------------------------------------------------------------------------- custom transformers
class QuantileClipper(BaseEstimator, TransformerMixin):
    """Winsorise each column at quantiles learned on the training data (NaN preserved)."""

    def __init__(self, lower: float = 0.005, upper: float = 0.995):
        self.lower, self.upper = lower, upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype="float64")
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype="float64"), self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def signed_log1p(X):
    X = np.asarray(X, dtype="float64")
    return np.sign(X) * np.log1p(np.abs(X))


def _identity_names(transformer, input_features):
    return input_features


# --------------------------------------------------------------------------- preprocessors
def lr_preprocessor(cols: dict, clip: bool = True, log: bool = True, bins: bool = False,
                    min_frequency: float = 0.001) -> ColumnTransformer:
    """Logistic Regression: impute (+ missing indicators), optional clip / log / bins, scale, one-hot."""
    skewed = [c for c in cols["lr_skewed"]] if log else []
    plain = [c for c in cols["lr_continuous"] if c not in skewed]

    def numeric_branch(with_log: bool):
        steps = [SimpleImputer(strategy="median", keep_empty_features=True)]
        if clip:
            steps.append(QuantileClipper())
        if with_log:
            steps.append(FunctionTransformer(signed_log1p, feature_names_out=_identity_names))
        steps.append(StandardScaler())
        return make_pipeline(*steps)

    transformers = [
        ("num", numeric_branch(False), plain),
        # Missing-value indicators in their own branch: never clipped or log-transformed (a rare 0/1
        # indicator would otherwise be clipped to a constant). Learned on the training fold.
        ("missing", MissingIndicator(features="missing-only", error_on_new=False),
         cols["lr_continuous"] + cols["binary"]),
        ("bin01", SimpleImputer(strategy="most_frequent"), cols["binary"]),
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="Missing"),
                              OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency,
                                            sparse_output=False)), cols["categorical"]),
    ]
    if skewed:
        transformers.insert(1, ("num_log", numeric_branch(True), skewed))
    if bins and cols["nonmonotonic"]:
        transformers.append(("bins", make_pipeline(
            SimpleImputer(strategy="median"),
            KBinsDiscretizer(n_bins=10, encode="onehot-dense", strategy="quantile", quantile_method="averaged_inverted_cdf")),
            cols["nonmonotonic"]))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)


def tree_preprocessor(cols: dict) -> ColumnTransformer:
    """Gradient boosting: numbers pass through with NaN kept (native missing handling); text -> ordinal codes
    with missing and unseen categories mapped to NaN (treated as their own branch by the model)."""
    return ColumnTransformer([
        ("num", "passthrough", cols["continuous"] + cols["binary"]),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                               encoded_missing_value=np.nan), cols["categorical"]),
    ], remainder="drop", verbose_feature_names_out=False)


def tree_categorical_mask(cols: dict) -> list[bool]:
    n_num = len(cols["continuous"]) + len(cols["binary"])
    return [False] * n_num + [True] * len(cols["categorical"])
