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
