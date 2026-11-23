"""Defect pattern -> yield: how many dies does each pattern cost?

A wafer map already shows its yield, so predicting yield from the map is not the
interesting question. The interesting one is *why* dies were lost. Each pattern
points to a different root cause, so attributing lost yield to patterns tells
engineers which problem to fix first.

Units: "wafer-equivalents". A wafer with 10 points less yield than a typical
`none` wafer contributes 0.10 wafers of excess loss. Every wafer counts equally,
whatever its die count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wafer.data import FAILURE_TYPES

NONE = "none"


def _names(y) -> np.ndarray:
    y = np.asarray(y)
    return np.array(FAILURE_TYPES)[y] if np.issubdtype(y.dtype, np.integer) else y


def baseline_yield(y, die_yield) -> float:
    """Mean yield of wafers with no pattern: the loss every wafer has anyway."""
    names, die_yield = _names(y), np.asarray(die_yield, dtype=float)
    return float(die_yield[names == NONE].mean())


def yield_loss_table(y, die_yield, baseline: float | None = None) -> pd.DataFrame:
    """Per pattern: frequency, severity, and total excess loss (frequency x severity).

    Columns
        wafers          number of wafers with the pattern
        mean_yield      average die yield
        excess_per_wafer  baseline - mean_yield (yield points lost *because of* the pattern)
        excess_wafers   summed excess loss, in wafer-equivalents
        share           share of all pattern-caused excess loss
        gain_if_fixed_pts  fleet-wide yield gain, in points, if this pattern's wafers were brought
                        back to baseline yield
    """
    names, die_yield = _names(y), np.asarray(die_yield, dtype=float)
    if baseline is None:
        baseline = baseline_yield(names, die_yield)
    df = pd.DataFrame({"pattern": names, "yield": die_yield})
    df["excess"] = baseline - df["yield"]
    t = df.groupby("pattern").agg(
        wafers=("yield", "size"),
        mean_yield=("yield", "mean"),
        excess_wafers=("excess", "sum"),
    )
    t["excess_per_wafer"] = baseline - t["mean_yield"]
    t = t.drop(index=NONE, errors="ignore")
    pos = t["excess_wafers"].clip(lower=0)
    t["share"] = pos / pos.sum()
    t["gain_if_fixed_pts"] = 100 * pos / len(df)
    cols = ["wafers", "mean_yield", "excess_per_wafer", "excess_wafers", "share"]
    return t[cols + ["gain_if_fixed_pts"]].sort_values("excess_wafers", ascending=False)


def attribution_error(y_true, y_pred, die_yield, baseline: float) -> pd.DataFrame:
    """Compare loss attributed with true labels vs labels predicted by a classifier.

    This is the chained use case: on new, unlabeled wafers the pattern comes from
    the model, so the question is whether the model's attribution can be trusted.
    """
    true = yield_loss_table(y_true, die_yield, baseline)["excess_wafers"]
    pred = yield_loss_table(y_pred, die_yield, baseline)["excess_wafers"]
    out = pd.concat({"true": true, "predicted": pred}, axis=1).fillna(0.0)
    out["error_pct"] = (out["predicted"] - out["true"]) / out["true"].abs().clip(lower=1e-9)
    return out.sort_values("true", ascending=False)


def plot_pareto(table: pd.DataFrame, ax=None, title: str = "Excess yield loss by pattern"):
    """Bars = total excess loss, line = cumulative share. Classic Pareto chart."""
    import matplotlib.pyplot as plt

    t = table[table["excess_wafers"] > 0]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.bar(t.index, t["excess_wafers"], color="#4C72B0")
    ax.set_ylabel("excess loss (wafer-equivalents)")
    ax.tick_params(axis="x", rotation=30)
    ax2 = ax.twinx()
    ax2.plot(t.index, t["share"].cumsum() * 100, color="#C44E52", marker="o")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("cumulative share (%)")
    ax.set_title(title)
    ax.figure.tight_layout()
    return ax
