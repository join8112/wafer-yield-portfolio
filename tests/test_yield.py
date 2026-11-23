import numpy as np
import pytest

from wafer import yield_analysis as ya
from wafer.data import FAILURE_TYPES


@pytest.fixture
def toy():
    # 6 none wafers at 0.9, 2 Center at 0.7, 1 Near-full at 0.1, 2 Scratch at 0.92
    y = ["none"] * 6 + ["Center"] * 2 + ["Near-full"] + ["Scratch"] * 2
    yld = [0.9] * 6 + [0.7] * 2 + [0.1] + [0.92] * 2
    return np.array(y), np.array(yld)


def test_baseline_is_none_mean(toy):
    y, yld = toy
    assert ya.baseline_yield(y, yld) == pytest.approx(0.9)


def test_table_frequency_times_severity(toy):
    y, yld = toy
    t = ya.yield_loss_table(y, yld)
    assert "none" not in t.index
    assert t.loc["Center", "excess_per_wafer"] == pytest.approx(0.2)
    assert t.loc["Center", "excess_wafers"] == pytest.approx(0.4)  # 2 wafers x 0.2
    assert t.loc["Near-full", "excess_wafers"] == pytest.approx(0.8)
    # a pattern that yields better than baseline gets no share of the loss
    assert t.loc["Scratch", "share"] == 0
    assert t["share"].sum() == pytest.approx(1.0)
    assert t.index[0] == "Near-full"  # sorted by total loss
    assert t.loc["Center", "gain_if_fixed_pts"] == pytest.approx(100 * 0.4 / len(y))


def test_accepts_integer_class_ids(toy):
    y, yld = toy
    ids = np.array([FAILURE_TYPES.index(p) for p in y])
    t1, t2 = ya.yield_loss_table(y, yld), ya.yield_loss_table(ids, yld)
    assert t1["excess_wafers"].equals(t2["excess_wafers"])


def test_attribution_error_zero_for_perfect_classifier(toy):
    y, yld = toy
    out = ya.attribution_error(y, y, yld, baseline=0.9)
    assert np.allclose(out["error_pct"], 0)
    # calling a Center wafer "none" moves its loss out of Center
    pred = y.copy()
    pred[6] = "none"
    out = ya.attribution_error(y, pred, yld, baseline=0.9)
    assert out.loc["Center", "predicted"] == pytest.approx(0.2)
