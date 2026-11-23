import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from wafer import eda  # noqa: E402
from wafer.data import clean_wm811k, load_secom  # noqa: E402


def test_clean_wm811k_unwraps_labels_and_computes_yield(raw_wm811k):
    df = clean_wm811k(raw_wm811k)

    assert df["failure_type"].tolist()[:4] == ["Center", "Edge-Ring", "none", "Scratch"]
    assert (df.loc[4:, "failure_type"] == "unlabeled").all()
    assert df["is_labeled"].sum() == 4

    m = df.at[0, "wafer_map"]
    expected = 1 - (m == 2).sum() / (m > 0).sum()
    assert np.isclose(df.at[0, "die_yield"], expected)
    assert df["die_yield"].between(0, 1).all()
    assert (df["rows"] == df["wafer_map"].map(lambda a: a.shape[0])).all()


def test_wm811k_eda_helpers(raw_wm811k):
    df = clean_wm811k(raw_wm811k)
    summary = eda.label_summary(df)
    assert summary["count"].sum() == 4
    assert np.isclose(summary["share"].sum(), 1.0)

    by_type = eda.yield_by_failure_type(df).dropna(subset=["mean"])
    assert set(by_type.index) == {"Center", "Edge-Ring", "none", "Scratch"}

    eda.plot_wafer_grid(df, per_class=2)
    eda.plot_yield_by_type(df)


def test_load_secom(secom_dir):
    X, meta = load_secom(secom_dir)
    assert X.shape == (40, 12)
    assert list(X.columns[:2]) == ["f000", "f001"]
    assert meta["fail"].sum() == 6
    assert pd.api.types.is_datetime64_any_dtype(meta["timestamp"])


def test_secom_eda_helpers(secom_dir):
    X, meta = load_secom(secom_dir)
    low = eda.low_information_features(X)
    assert "f003" in low["constant"]
    assert "f005" in low["mostly_missing"]

    pairs = eda.correlated_pairs(X, threshold=0.99)
    assert {"f006", "f007"} == set(pairs.iloc[0][["a", "b"]])

    prof = eda.missing_profile(X)
    assert prof.iloc[0] >= 1

    weekly = eda.fail_rate_over_time(meta)
    assert weekly["runs"].sum() == 40
