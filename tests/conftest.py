"""Synthetic stand-ins for the real datasets, so tests and CI run without downloads."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _wafer(size: int, fail_frac: float, rng) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    c = (size - 1) / 2
    on_wafer = (yy - c) ** 2 + (xx - c) ** 2 <= c**2
    m = on_wafer.astype(int)
    fails = on_wafer & (rng.random((size, size)) < fail_frac)
    m[fails] = 2
    return m


@pytest.fixture
def raw_wm811k() -> pd.DataFrame:
    """Mimics LSWMD.pkl's raw layout, including its nested-array labels."""
    rng = np.random.default_rng(0)
    rows = []
    labels = ["Center", "Edge-Ring", "none", "Scratch", None, None]
    for i, lab in enumerate(labels):
        size = 20 + i
        rows.append(
            {
                "waferMap": _wafer(size, 0.05 + 0.05 * i, rng),
                "dieSize": float(size * size),
                "lotName": f"lot{i // 3}",
                "waferIndex": float(i + 1),
                "trianTestLabel": np.array([["Training"]]) if lab else np.array([]),
                "failureType": np.array([[lab]]) if lab else np.array([]),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def secom_dir(tmp_path):
    rng = np.random.default_rng(1)
    n, p = 40, 12
    X = rng.normal(size=(n, p))
    X[:, 3] = 1.0  # constant feature
    X[rng.random((n, p)) < 0.1] = np.nan
    X[:, 5] = np.nan  # fully missing feature
    X[:, 7] = X[:, 6] * 2 + 0.001  # perfectly correlated pair
    lines = [" ".join("NaN" if np.isnan(v) else f"{v:.4f}" for v in row) for row in X]
    (tmp_path / "secom.data").write_text("\n".join(lines) + "\n")

    labels = []
    for i in range(n):
        lab = 1 if i % 7 == 0 else -1
        labels.append(f'{lab} "{(i % 28) + 1:02d}/07/2008 11:55:{i % 60:02d}"')
    (tmp_path / "secom_labels.data").write_text("\n".join(labels) + "\n")
    return tmp_path


def make_patterned_wafers(n_per_class: int = 30, size: int = 40, seed: int = 0):
    """Synthetic wafers with recognisable patterns, for end-to-end model tests.

    Returns (maps, labels, lots) where labels index FAILURE_TYPES.
    """
    from wafer.data import FAILURE_TYPES

    rng = np.random.default_rng(seed)
    c = (size - 1) / 2
    yy, xx = np.mgrid[:size, :size]
    r = np.hypot(yy - c, xx - c) / c
    theta = np.arctan2(yy - c, xx - c)
    on = r <= 1

    def make(kind):
        fail = rng.random((size, size)) < 0.03
        if kind == "Center":
            fail |= r < 0.3
        elif kind == "Donut":
            fail |= (r > 0.35) & (r < 0.6)
        elif kind == "Edge-Ring":
            fail |= r > 0.88
        elif kind == "Edge-Loc":
            a = rng.uniform(-np.pi, np.pi)
            fail |= (r > 0.75) & (np.abs(np.angle(np.exp(1j * (theta - a)))) < 0.5)
        elif kind == "Loc":
            cy, cx = rng.uniform(c - c / 2, c + c / 2, 2)
            fail |= np.hypot(yy - cy, xx - cx) < size * 0.1
        elif kind == "Near-full":
            fail |= rng.random((size, size)) < 0.85
        elif kind == "Random":
            fail |= rng.random((size, size)) < 0.35
        elif kind == "Scratch":
            row = rng.integers(size // 4, 3 * size // 4)
            fail[row, size // 5 : 4 * size // 5] = True
        m = on.astype(np.uint8)
        m[on & fail] = 2
        return m

    maps, labels, lots = [], [], []
    for k, kind in enumerate(FAILURE_TYPES):
        for i in range(n_per_class):
            maps.append(make(kind))
            labels.append(k)
            lots.append(f"lot{(i % 10)}")
    return maps, np.array(labels), np.array(lots)


@pytest.fixture
def patterned_df():
    from wafer.data import FAILURE_TYPES

    maps, labels, lots = make_patterned_wafers()
    df = pd.DataFrame({"wafer_map": maps, "failure_type": np.array(FAILURE_TYPES)[labels]})
    df["lot"] = lots
    df["split"] = "Training"
    df["is_labeled"] = True
    df["n_fail"] = [int((m == 2).sum()) for m in maps]
    df["die_yield"] = [1 - (m == 2).sum() / (m > 0).sum() for m in maps]
    return df
