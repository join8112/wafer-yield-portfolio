"""Loading and cleaning for the two public datasets used in this project.

WM-811K (LSWMD.pkl)
    811,457 wafer maps from real fabs. Each map is a 2-D array where
    0 = no die, 1 = passing die, 2 = failing die. About 21% of wafers carry
    one of 9 failure-pattern labels; the rest are unlabeled.

SECOM
    1,567 production runs x 590 anonymised sensor/process measurements,
    with a pass (-1) / fail (1) label and a timestamp per run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

FAILURE_TYPES = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
    "none",
]

# ---------------------------------------------------------------------------
# WM-811K
# ---------------------------------------------------------------------------


def _unwrap_label(value) -> str | None:
    """LSWMD stores labels as nested arrays like [['Center']] or an empty array."""
    arr = np.asarray(value, dtype=object).ravel()
    if arr.size == 0:
        return None
    return str(arr[0]).strip()


def clean_wm811k(raw: pd.DataFrame) -> pd.DataFrame:
    """Tidy the raw LSWMD frame into one row per wafer with derived columns.

    Adds:
        failure_type  label string, or "unlabeled"
        split         "Training" / "Test" / "unlabeled" (original authors' split)
        rows, cols    map shape
        n_dies        number of dies on the wafer (map > 0)
        n_fail        number of failing dies (map == 2)
        die_yield     1 - n_fail / n_dies; the per-wafer yield used in step 4
    """
    df = raw.rename(
        columns={
            "waferMap": "wafer_map",
            "dieSize": "die_size",
            "lotName": "lot",
            "waferIndex": "wafer_index",
            "trianTestLabel": "split",  # sic: typo is in the original dataset
            "failureType": "failure_type",
        }
    ).copy()

    df["failure_type"] = df["failure_type"].map(_unwrap_label).fillna("unlabeled")
    df["split"] = df["split"].map(_unwrap_label).fillna("unlabeled")
    df["wafer_index"] = df["wafer_index"].astype(int)

    shapes = df["wafer_map"].map(np.shape)
    df["rows"] = shapes.map(lambda s: s[0]).astype(int)
    df["cols"] = shapes.map(lambda s: s[1]).astype(int)
    df["n_dies"] = df["wafer_map"].map(lambda m: int((np.asarray(m) > 0).sum()))
    df["n_fail"] = df["wafer_map"].map(lambda m: int((np.asarray(m) == 2).sum()))
    df["die_yield"] = 1.0 - df["n_fail"] / df["n_dies"].where(df["n_dies"] > 0)

    df["is_labeled"] = df["failure_type"] != "unlabeled"
    return df.reset_index(drop=True)


def load_wm811k(path: str | Path | None = None, use_cache: bool = True) -> pd.DataFrame:
    """Load WM-811K, returning the cleaned frame from :func:`clean_wm811k`.

    The original pickle was written with pandas 0.19 and only loads on
    pandas < 3 (see requirements.txt). On first load the cleaned frame is
    cached to data/processed/wm811k_clean.pkl, which loads much faster.
    """
    cache = PROCESSED_DIR / "wm811k_clean.pkl"
    if use_cache and path is None and cache.exists():
        return pd.read_pickle(cache)

    path = Path(path) if path else RAW_DIR / "LSWMD.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py wm811k` first."
        )
    df = clean_wm811k(pd.read_pickle(path))

    if use_cache and path == RAW_DIR / "LSWMD.pkl":
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache)
    return df


# ---------------------------------------------------------------------------
# SECOM
# ---------------------------------------------------------------------------


def load_secom(raw_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load SECOM features and labels.

    Returns:
        X      (n_runs, 590) float frame, columns "f000".."f589", NaN for missing
        meta   frame with "fail" (bool), "label" (-1 pass / 1 fail) and "timestamp"
    """
    raw_dir = Path(raw_dir) if raw_dir else RAW_DIR
    data_path = raw_dir / "secom.data"
    label_path = raw_dir / "secom_labels.data"
    for p in (data_path, label_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run `python scripts/download_data.py secom` first."
            )

    X = pd.read_csv(data_path, sep=r"\s+", header=None, na_values=["NaN"])
    X.columns = [f"f{i:03d}" for i in range(X.shape[1])]

    # Each line looks like: -1 "19/07/2008 11:55:00"
    lines = [ln for ln in label_path.read_text().splitlines() if ln.strip()]
    parts = [ln.strip().split(maxsplit=1) for ln in lines]
    meta = pd.DataFrame(
        {
            "label": [int(p[0]) for p in parts],
            "timestamp": pd.to_datetime(
                [p[1].strip().strip('"') for p in parts], format="%d/%m/%Y %H:%M:%S"
            ),
        }
    )
    meta["fail"] = meta["label"] == 1

    if len(X) != len(meta):
        raise ValueError(f"SECOM row mismatch: {len(X)} feature rows vs {len(meta)} labels")
    return X, meta
