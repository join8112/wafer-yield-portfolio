"""Turn raw WM-811K wafer maps into fixed-size arrays and leakage-free splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from wafer.data import FAILURE_TYPES, PROCESSED_DIR

IMG_SIZE = 64
CLASS_TO_ID = {c: i for i, c in enumerate(FAILURE_TYPES)}
DATASET_PATH = PROCESSED_DIR / "wm811k_labeled_64.npz"


def resize_nearest(wafer_map: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Nearest-neighbour resize that keeps die states discrete (0 / 1 / 2).

    Bilinear resizing would invent values like 1.4 that mean nothing for a die.
    """
    m = np.asarray(wafer_map)
    rows = (np.arange(size) * m.shape[0] / size).astype(int)
    cols = (np.arange(size) * m.shape[1] / size).astype(int)
    return m[np.ix_(rows, cols)].astype(np.uint8)


def grouped_split(
    labels: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """Assign each wafer to "train" / "val" / "test".

    Stratified by class so rare patterns appear in every split, and grouped by
    lot so wafers from the same lot never land on both sides (they share
    process conditions and look alike, which would inflate test scores).
    """
    split = np.full(len(labels), "train", dtype=object)

    n_test = max(2, round(1 / test_size))
    outer = StratifiedGroupKFold(n_splits=n_test, shuffle=True, random_state=seed)
    trainval_idx, test_idx = next(outer.split(np.zeros(len(labels)), labels, groups))
    split[test_idx] = "test"

    n_val = max(2, round((1 - test_size) / val_size))
    inner = StratifiedGroupKFold(n_splits=n_val, shuffle=True, random_state=seed)
    _, val_rel = next(
        inner.split(np.zeros(len(trainval_idx)), labels[trainval_idx], groups[trainval_idx])
    )
    split[trainval_idx[val_rel]] = "val"
    return split


def build_dataset(df: pd.DataFrame, size: int = IMG_SIZE, seed: int = 42) -> dict:
    """Labeled wafers -> arrays ready for modelling.

    Returns a dict with:
        X          (n, size, size) uint8 maps
        y          (n,) int class ids (order of FAILURE_TYPES)
        split      (n,) "train" / "val" / "test"
        lot        (n,) lot names
        die_yield  (n,) per-wafer yield from the original map
        n_fail     (n,) failing dies on the original map
        orig_split (n,) the dataset authors' Training / Test label
    """
    lab = df[df["is_labeled"]].reset_index(drop=True)
    X = np.stack([resize_nearest(m, size) for m in lab["wafer_map"]])
    y = lab["failure_type"].map(CLASS_TO_ID).to_numpy()
    lot = lab["lot"].astype(str).to_numpy()
    return {
        "X": X,
        "y": y,
        "split": grouped_split(y, lot, seed=seed),
        "lot": lot,
        "die_yield": lab["die_yield"].to_numpy(dtype=np.float32),
        "n_fail": lab["n_fail"].to_numpy(dtype=np.int32),
        "orig_split": lab["split"].to_numpy(),
    }


def save_dataset(ds: dict, path: str | Path = DATASET_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in ds.items()})
    return path


def load_dataset(path: str | Path = DATASET_PATH) -> dict:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}
