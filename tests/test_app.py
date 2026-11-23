"""Smoke test: the Streamlit demo starts and classifies a wafer without errors."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "app" / "assets" / "model.txt.gz").exists():
    pytest.skip("app assets not built", allow_module_level=True)

from streamlit.testing.v1 import AppTest  # noqa: E402


def test_app_runs_and_predicts():
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Predicted pattern" in labels
    at.selectbox[0].select("Near-full").run()
    assert not at.exception
    assert at.metric[0].value == "Near-full"
