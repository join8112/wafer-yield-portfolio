"""Reusable EDA helpers so the notebooks stay short and the logic is testable."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from wafer.data import FAILURE_TYPES

# 0 = no die, 1 = pass, 2 = fail
WAFER_CMAP = ListedColormap(["#ffffff", "#9ecae1", "#d62728"])

# ---------------------------------------------------------------------------
# WM-811K
# ---------------------------------------------------------------------------


def label_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Count and share of each failure type among labeled wafers."""
    labeled = df[df["is_labeled"]]
    counts = labeled["failure_type"].value_counts().reindex(FAILURE_TYPES, fill_value=0)
    out = pd.DataFrame({"count": counts, "share": counts / counts.sum()})
    out["imbalance_vs_largest"] = out["count"].max() / out["count"].replace(0, np.nan)
    return out


def yield_by_failure_type(df: pd.DataFrame) -> pd.DataFrame:
    """Die-yield distribution per failure pattern: the defect -> yield link."""
    labeled = df[df["is_labeled"]]
    return (
        labeled.groupby("failure_type")["die_yield"]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .reindex(FAILURE_TYPES)
        .sort_values("mean")
    )


def plot_wafer_grid(
    df: pd.DataFrame, per_class: int = 4, seed: int = 0, classes: list[str] | None = None
):
    """Grid of example wafer maps, one row per failure type."""
    classes = classes or [c for c in FAILURE_TYPES if (df["failure_type"] == c).any()]
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(
        len(classes), per_class, figsize=(per_class * 1.6, len(classes) * 1.6), squeeze=False
    )
    for r, cls in enumerate(classes):
        idx = df.index[df["failure_type"] == cls].to_numpy()
        picks = rng.choice(idx, size=min(per_class, len(idx)), replace=False)
        for c in range(per_class):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            if c < len(picks):
                ax.imshow(df.at[picks[c], "wafer_map"], cmap=WAFER_CMAP, vmin=0, vmax=2)
            if c == 0:
                ax.set_ylabel(cls, rotation=0, ha="right", va="center", fontsize=9)
    fig.suptitle("WM-811K example wafer maps (red = failing die)")
    fig.tight_layout()
    return fig


def plot_yield_by_type(df: pd.DataFrame):
    labeled = df[df["is_labeled"]]
    order = yield_by_failure_type(df).index.tolist()
    fig, ax = plt.subplots(figsize=(8, 4))
    data = [labeled.loc[labeled["failure_type"] == t, "die_yield"].dropna() for t in order]
    ax.boxplot(data, tick_labels=order, showfliers=False)
    ax.set_ylabel("die yield")
    ax.set_title("Die yield by failure pattern")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# SECOM
# ---------------------------------------------------------------------------


def missing_profile(X: pd.DataFrame, thresholds=(0.0, 0.05, 0.2, 0.5, 0.8)) -> pd.Series:
    """How many features have more than each fraction of values missing."""
    rate = X.isna().mean()
    return pd.Series({f">{t:.0%}": int((rate > t).sum()) for t in thresholds}, name="n_features")


def low_information_features(X: pd.DataFrame, missing_thresh: float = 0.5) -> dict[str, list]:
    """Features that carry no signal and are safe to drop before modelling."""
    rate = X.isna().mean()
    nunique = X.nunique(dropna=True)
    return {
        "constant": nunique[nunique <= 1].index.tolist(),
        "mostly_missing": rate[rate > missing_thresh].index.tolist(),
    }


def correlated_pairs(X: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """Feature pairs with |Pearson r| above threshold (redundant sensors)."""
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack()
    pairs = pairs[pairs > threshold].sort_values(ascending=False)
    return pairs.rename("abs_r").reset_index().rename(columns={"level_0": "a", "level_1": "b"})


def fail_rate_over_time(meta: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """Weekly run count and fail rate; checks for drift / time-based leakage."""
    s = meta.set_index("timestamp")["fail"].resample(freq)
    return pd.DataFrame({"runs": s.size(), "fail_rate": s.mean()})
