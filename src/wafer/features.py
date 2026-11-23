"""Handcrafted wafer-map features for the gradient-boosting baseline.

Each feature mirrors how an engineer describes a pattern:
    where are the fails?        -> radial rings and angular sectors
    how concentrated are they?  -> largest failing cluster: size, position, shape
    how widespread?             -> overall fail ratio
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

N_RINGS = 5
N_SECTORS = 8


def _polar_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    c = (size - 1) / 2
    yy, xx = np.mgrid[:size, :size]
    r = np.hypot(yy - c, xx - c) / (c + 0.5)  # 0 at centre, ~1 at edge
    theta = np.arctan2(yy - c, xx - c)  # -pi .. pi
    return r, theta


def zone_features(X: np.ndarray) -> pd.DataFrame:
    """Vectorised fail ratios per ring and per edge sector for a stack of maps."""
    n, size, _ = X.shape
    r, theta = _polar_grid(size)
    die = X > 0
    fail = X == 2
    n_die = die.reshape(n, -1).sum(1).clip(min=1)

    feats = {"fail_ratio": fail.reshape(n, -1).sum(1) / n_die}

    ring_edges = np.linspace(0, 1.0001, N_RINGS + 1)
    ring_edges[-1] = r.max() + 1
    for i in range(N_RINGS):
        mask = (r >= ring_edges[i]) & (r < ring_edges[i + 1])
        d = (die & mask).reshape(n, -1).sum(1)
        f = (fail & mask).reshape(n, -1).sum(1)
        feats[f"ring{i}_fail"] = np.where(d > 0, f / np.maximum(d, 1), 0.0)

    # Angular sectors of the outer ring capture Edge-Loc (one side) vs Edge-Ring (all round).
    outer = r >= ring_edges[N_RINGS - 1]
    sector_idx = ((theta + np.pi) / (2 * np.pi) * N_SECTORS).astype(int) % N_SECTORS
    sectors = []
    for s in range(N_SECTORS):
        mask = outer & (sector_idx == s)
        d = (die & mask).reshape(n, -1).sum(1)
        f = (fail & mask).reshape(n, -1).sum(1)
        sectors.append(np.where(d > 0, f / np.maximum(d, 1), 0.0))
    sectors = np.stack(sectors, 1)
    feats["edge_sector_max"] = sectors.max(1)
    feats["edge_sector_min"] = sectors.min(1)
    feats["edge_sector_std"] = sectors.std(1)
    feats["edge_uniformity"] = sectors.min(1) / np.maximum(sectors.max(1), 1e-6)
    return pd.DataFrame(feats)


def cluster_features(wafer: np.ndarray) -> dict:
    """Geometry of the largest connected cluster of failing dies on one map."""
    size = wafer.shape[0]
    fail = wafer == 2
    n_die = max(int((wafer > 0).sum()), 1)
    labels, n = ndimage.label(fail, structure=np.ones((3, 3)))
    out = {
        "n_clusters": n,
        "largest_cluster_frac": 0.0,
        "largest_cluster_dist": 0.0,
        "largest_cluster_elong": 0.0,
        "largest_cluster_fill": 0.0,
    }
    if n == 0:
        return out

    sizes = np.bincount(labels.ravel())[1:]
    k = int(sizes.argmax()) + 1
    ys, xs = np.nonzero(labels == k)
    c = (size - 1) / 2
    out["largest_cluster_frac"] = len(ys) / n_die
    out["largest_cluster_dist"] = float(np.hypot(ys.mean() - c, xs.mean() - c) / (c + 0.5))

    if len(ys) >= 3:
        # Ratio of principal axes: ~1 for a blob, large for a scratch line.
        cov = np.cov(np.stack([ys, xs]))
        ev = np.sort(np.linalg.eigvalsh(cov))
        out["largest_cluster_elong"] = float(np.sqrt(ev[1] / max(ev[0], 1e-3)))
        h, w = np.ptp(ys) + 1, np.ptp(xs) + 1
        out["largest_cluster_fill"] = len(ys) / (h * w)
    return out


def extract_features(X: np.ndarray) -> pd.DataFrame:
    zones = zone_features(X)
    clusters = pd.DataFrame([cluster_features(m) for m in X])
    return pd.concat([zones, clusters], axis=1)
