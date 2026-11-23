import numpy as np
import pandas as pd
import pytest

from wafer import secom as S


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n = 300
    y = (rng.random(n) < 0.15).astype(int)
    X = pd.DataFrame(rng.normal(size=(n, 8)), columns=[f"f{i:03d}" for i in range(8)])
    X["f000"] += 2.0 * y  # informative sensor
    X["f001"] = X["f000"] * 3 + 0.001  # duplicate of f000
    X["f002"] = 1.0  # constant
    X.loc[rng.random(n) < 0.8, "f003"] = np.nan  # mostly missing
    X.loc[rng.random(n) < 0.1, "f004"] = np.nan  # a few missing
    ts = pd.Series(pd.date_range("2008-07-01", periods=n, freq="6h"))
    return X, y, ts


def test_cleaner_drops_useless_sensors_and_imputes(toy):
    X, y, _ = toy
    c = S.SensorCleaner().fit(X)
    assert "f000" in c.columns_
    assert not {"f001", "f002", "f003"} & set(c.columns_)
    out = c.transform(X)
    assert not out.isna().any().any()


def test_cleaner_uses_training_statistics_only(toy):
    X, y, _ = toy
    c = S.SensorCleaner().fit(X.iloc[:100])
    test = X.iloc[100:].copy()
    test["f004"] = np.nan
    out = c.transform(test)
    assert (out["f004"] == X["f004"].iloc[:100].median()).all()


def test_time_split_is_ordered(toy):
    X, y, ts = toy
    tr, te = S.time_split(ts.sample(frac=1, random_state=0).reset_index(drop=True), 0.7)
    assert len(tr) == 210 and len(te) == 90


def test_models_learn_and_metrics(toy):
    X, y, ts = toy
    tr, te = S.time_split(ts)
    for model in (S.make_logreg(0.5), S.make_lgbm(n_estimators=50)):
        p = model.fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
        r = S.score_all(y[te], p)
        assert r["roc_auc"] > 0.75
        lo, hi = S.bootstrap_ci(y[te], p, n=200)
        assert lo <= r["pr_auc"] <= hi

    shap = S.lgbm_shap(model, X.iloc[tr])
    assert shap.abs().mean().idxmax() == "f000"


def test_recall_at_fpr():
    y = np.array([0, 0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.95])
    assert S.recall_at_fpr(y, s, 0.0) == pytest.approx(0.5)
    assert S.recall_at_fpr(y, s, 0.25) == pytest.approx(1.0)
