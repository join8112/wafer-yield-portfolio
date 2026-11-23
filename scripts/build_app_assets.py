"""Build the small files the Streamlit demo needs, so the app runs without the full dataset.

    python scripts/build_app_assets.py

Writes to app/assets/:
    model.txt.gz   LightGBM pattern classifier (a lighter version of notebook 03's model:
                   learning rate 0.1 instead of 0.05, so about half the trees and a ~5 MB file,
                   with the same test macro-F1 to within 0.003)
    samples.npz    a few test-split wafers per pattern (never seen in training)
    model_info.json  test scores of this exact model, shown in the app
"""

from __future__ import annotations

import gzip
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np

from wafer import evaluate
from wafer.features import extract_features
from wafer.preprocess import load_dataset

ASSETS = Path(__file__).resolve().parents[1] / "app" / "assets"
PER_CLASS = 25
SEED = 42


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    ds = load_dataset()
    tr, va, te = (ds["split"] == s for s in ("train", "val", "test"))

    print("extracting features (~1 minute)...")
    F = extract_features(ds["X"])
    clf = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.1,
        num_leaves=63,
        class_weight="balanced",
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=SEED,
        verbose=-1,
    )
    warnings.filterwarnings("ignore", message=".*eval_set.*")
    clf.fit(
        F[tr],
        ds["y"][tr],
        eval_set=[(F[va], ds["y"][va])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    pred = clf.predict(F[te])
    summary = evaluate.summarize(ds["y"][te], pred, ds["n_fail"][te]).round(4).to_dict()
    print("test:", summary)

    model_str = clf.booster_.model_to_string(num_iteration=clf.best_iteration_)
    (ASSETS / "model.txt.gz").write_bytes(gzip.compress(model_str.encode(), mtime=0))
    info = {"trees_per_class": int(clf.best_iteration_), "test": summary}
    (ASSETS / "model_info.json").write_text(json.dumps(info, indent=2))

    rng = np.random.default_rng(SEED)
    test_idx = np.flatnonzero(te)
    pick = []
    for c in np.unique(ds["y"]):
        idx = test_idx[ds["y"][test_idx] == c]
        pick.extend(rng.choice(idx, size=min(PER_CLASS, len(idx)), replace=False))
    pick = np.sort(np.array(pick))
    np.savez_compressed(
        ASSETS / "samples.npz",
        X=ds["X"][pick],
        y=ds["y"][pick],
        die_yield=ds["die_yield"][pick],
    )
    for f in sorted(ASSETS.iterdir()):
        print(f"{f.name:18s} {f.stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
