"""Metrics and plots shared by the baseline and the CNN, so results are comparable."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from wafer.data import FAILURE_TYPES

NONE_ID = FAILURE_TYPES.index("none")


def yield_loss_capture(y_true, y_pred, n_fail) -> float:
    """Share of failing dies on patterned wafers whose pattern was identified correctly.

    Plain macro-F1 treats a missed Scratch (few lost dies) the same as a missed
    Near-full (a scrapped wafer). This weights each patterned wafer by how many
    dies it lost, which is closer to what a fab cares about.
    """
    y_true, y_pred, n_fail = map(np.asarray, (y_true, y_pred, n_fail))
    patterned = y_true != NONE_ID
    w = n_fail[patterned].astype(float)
    if w.sum() == 0:
        return float("nan")
    return float((w * (y_true[patterned] == y_pred[patterned])).sum() / w.sum())


def summarize(y_true, y_pred, n_fail=None, name: str = "model") -> pd.Series:
    out = {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    if n_fail is not None:
        out["yield_loss_capture"] = yield_loss_capture(y_true, y_pred, n_fail)
    return pd.Series(out, name=name)


def per_class_report(y_true, y_pred) -> pd.DataFrame:
    labels = list(range(len(FAILURE_TYPES)))
    rep = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=FAILURE_TYPES,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(rep).T.loc[FAILURE_TYPES]


def plot_confusion(y_true, y_pred, title: str = "Confusion matrix (row-normalised)"):
    labels = list(range(len(FAILURE_TYPES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(labels, FAILURE_TYPES, rotation=45, ha="right")
    ax.set_yticks(labels, FAILURE_TYPES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in labels:
        for j in labels:
            if cm[i, j] >= 0.01:
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if cm[i, j] > 0.5 else "black",
                )
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(title)
    fig.tight_layout()
    return fig
