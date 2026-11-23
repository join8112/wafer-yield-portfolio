"""Build the compact modelling dataset from LSWMD.pkl.

    python scripts/prepare_wm811k.py

Writes data/processed/wm811k_labeled_64.npz: the 172,950 labeled wafers resized to
64x64, with class ids, lot-grouped train/val/test split, yield and fail counts.
This file is small enough to upload to Google Drive for CNN training on Colab.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from wafer.data import FAILURE_TYPES, load_wm811k
from wafer.preprocess import build_dataset, save_dataset


def main() -> None:
    t0 = time.time()
    df = load_wm811k()
    print(f"loaded {len(df):,} wafers in {time.time() - t0:.0f}s")

    ds = build_dataset(df)
    path = save_dataset(ds)
    print(f"saved {path} ({path.stat().st_size / 1e6:.0f} MB) in {time.time() - t0:.0f}s\n")

    counts = pd.crosstab(
        pd.Series(np.array(FAILURE_TYPES)[ds["y"]], name="class"),
        pd.Series(ds["split"], name="split"),
    )[["train", "val", "test"]]
    print(counts)
    lots = {s: set(ds["lot"][ds["split"] == s]) for s in ("train", "val", "test")}
    overlap = (lots["train"] & lots["test"]) | (lots["train"] & lots["val"])
    print(f"\nlots shared between splits: {len(overlap)}")


if __name__ == "__main__":
    main()
