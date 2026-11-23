import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from wafer import evaluate  # noqa: E402
from wafer.data import FAILURE_TYPES  # noqa: E402
from wafer.features import extract_features  # noqa: E402
from wafer.preprocess import build_dataset, load_dataset, resize_nearest, save_dataset  # noqa: E402


def test_resize_keeps_die_states_discrete():
    m = np.array([[0, 1, 2], [1, 2, 1], [0, 1, 0]])
    out = resize_nearest(m, 8)
    assert out.shape == (8, 8)
    assert set(np.unique(out)) <= {0, 1, 2}


def test_split_is_grouped_by_lot_and_covers_classes(patterned_df, tmp_path):
    ds = build_dataset(patterned_df, size=32)
    lots = {s: set(ds["lot"][ds["split"] == s]) for s in ("train", "val", "test")}
    assert not lots["train"] & lots["test"]
    assert not lots["train"] & lots["val"]
    assert not lots["val"] & lots["test"]
    assert set(ds["y"][ds["split"] == "test"]) == set(range(len(FAILURE_TYPES)))

    path = save_dataset(ds, tmp_path / "ds.npz")
    back = load_dataset(path)
    assert back["X"].shape == ds["X"].shape and back["X"].dtype == np.uint8


def test_feature_baseline_learns_patterns(patterned_df):
    from sklearn.ensemble import RandomForestClassifier

    ds = build_dataset(patterned_df, size=32)
    F = extract_features(ds["X"])
    assert not F.isna().any().any()
    tr, te = ds["split"] != "test", ds["split"] == "test"
    clf = RandomForestClassifier(n_estimators=200, random_state=0).fit(F[tr], ds["y"][tr])
    pred = clf.predict(F[te])
    s = evaluate.summarize(ds["y"][te], pred, ds["n_fail"][te])
    assert s["macro_f1"] > 0.7


def test_yield_loss_capture_weights_by_lost_dies():
    none = FAILURE_TYPES.index("none")
    y_true = np.array([0, 5, none])
    y_pred = np.array([0, 0, 0])  # gets the small one right, misses the big one
    n_fail = np.array([10, 90, 50])
    assert evaluate.yield_loss_capture(y_true, y_pred, n_fail) == pytest.approx(0.1)


def test_cnn_trains_and_grad_cam_runs(patterned_df):
    torch = pytest.importorskip("torch")
    from wafer import cnn

    ds = build_dataset(patterned_df, size=32)
    tr, va = ds["split"] == "train", ds["split"] == "val"
    model, hist = cnn.train(
        ds["X"][tr],
        ds["y"][tr],
        ds["X"][va],
        ds["y"][va],
        epochs=2,
        batch_size=32,
        device="cpu",
        verbose=False,
    )
    assert len(hist) == 2
    probs = cnn.predict(model, ds["X"][va])
    assert probs.shape == (va.sum(), len(FAILURE_TYPES))
    assert np.allclose(probs.sum(1), 1, atol=1e-5)

    cam, cls = cnn.grad_cam(model, ds["X"][0])
    assert cam.shape == (32, 32) and 0 <= cam.min() and cam.max() <= 1 + 1e-6
    assert 0 <= cls < len(FAILURE_TYPES)
    assert isinstance(torch.__version__, str)
