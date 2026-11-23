"""Process-sensor fail prediction on SECOM.

Everything that learns from data (which sensors to drop, imputation values,
scaling) is fitted on the training runs only, inside a scikit-learn Pipeline,
so no information from the test period leaks into training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class SensorCleaner(BaseEstimator, TransformerMixin):
    """Drop uninformative and redundant sensors, then median-impute.

    1. drop sensors missing in more than `max_missing` of training runs
    2. drop sensors that are constant in the training runs
    3. median-impute the rest (medians from training runs)
    4. drop one sensor from every pair with |r| > `max_corr` (keeps the first)
    """

    def __init__(self, max_missing: float = 0.5, max_corr: float = 0.95):
        self.max_missing = max_missing
        self.max_corr = max_corr

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        keep = X.columns[X.isna().mean() <= self.max_missing]
        keep = [c for c in keep if X[c].nunique(dropna=True) > 1]
        self.medians_ = X[keep].median()
        filled = X[keep].fillna(self.medians_)

        corr = filled.corr().abs().to_numpy()
        np.fill_diagonal(corr, 0)
        dropped = set()
        for i in range(len(keep)):
            if i in dropped:
                continue
            for j in np.flatnonzero(corr[i, i + 1 :] > self.max_corr) + i + 1:
                dropped.add(j)
        self.columns_ = [c for k, c in enumerate(keep) if k not in dropped]
        self.n_input_ = X.shape[1]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X[self.columns_].fillna(self.medians_[self.columns_])

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_)


def time_split(timestamps: pd.Series, train_frac: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    """Earliest `train_frac` of runs for training, the rest for testing."""
    order = np.argsort(pd.to_datetime(timestamps).to_numpy(), kind="stable")
    cut = int(len(order) * train_frac)
    return order[:cut], order[cut:]


def make_logreg(C: float = 0.05) -> Pipeline:
    """L1-regularised logistic regression: sparse, so it also selects sensors."""
    import sklearn

    major, minor = (int(v) for v in sklearn.__version__.split(".")[:2])
    # scikit-learn 1.8 replaced penalty="l1" with l1_ratio=1; support both.
    l1 = {"l1_ratio": 1.0} if (major, minor) >= (1, 8) else {"penalty": "l1"}
    clf = LogisticRegression(C=C, solver="liblinear", class_weight="balanced", max_iter=5000, **l1)
    return Pipeline([("clean", SensorCleaner()), ("scale", StandardScaler()), ("clf", clf)])


def make_lgbm(**params) -> Pipeline:
    import lightgbm as lgb

    defaults = dict(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=7,
        min_child_samples=10,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.5,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    defaults.update(params)
    return Pipeline([("clean", SensorCleaner()), ("clf", lgb.LGBMClassifier(**defaults))])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def recall_at_fpr(y_true, score, max_fpr: float = 0.05) -> float:
    """Share of failing runs caught when at most `max_fpr` of passing runs are flagged.

    This is how a fab would use the model: engineers can afford to inspect a
    fixed small share of good runs, and want to catch as many bad ones as possible.
    """
    fpr, tpr, _ = roc_curve(y_true, score)
    return float(tpr[fpr <= max_fpr].max())


def score_all(y_true, score, max_fpr: float = 0.05) -> dict:
    y_true = np.asarray(y_true).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, score)),
        "roc_auc": float(roc_auc_score(y_true, score)),
        f"recall_at_{int(max_fpr * 100)}pct_fpr": recall_at_fpr(y_true, score, max_fpr),
        "base_rate": float(y_true.mean()),
    }


def bootstrap_ci(y_true, score, metric=average_precision_score, n: int = 2000, seed: int = 0):
    """95% bootstrap interval; with ~25 failing test runs this is wide, and should be shown."""
    rng = np.random.default_rng(seed)
    y_true, score = np.asarray(y_true), np.asarray(score)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():
            continue
        vals.append(metric(y_true[idx], score[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def lgbm_shap(pipe: Pipeline, X) -> pd.DataFrame:
    """Per-run SHAP values from LightGBM's built-in TreeSHAP (no extra dependency)."""
    Xc = pipe.named_steps["clean"].transform(X)
    contrib = pipe.named_steps["clf"].booster_.predict(Xc, pred_contrib=True)
    return pd.DataFrame(contrib[:, :-1], columns=Xc.columns, index=Xc.index)
