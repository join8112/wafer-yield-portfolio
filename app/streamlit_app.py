"""Wafer defect & yield explorer (Streamlit demo).

Run from the repository root:
    streamlit run app/streamlit_app.py

Needs only the small files in app/assets/ and reports/ (built by
scripts/build_app_assets.py and notebook 06), not the full dataset.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # works without `pip install -e .` (e.g. Streamlit Cloud)

from wafer.data import FAILURE_TYPES  # noqa: E402
from wafer.features import extract_features  # noqa: E402
from wafer.preprocess import IMG_SIZE, resize_nearest  # noqa: E402

ASSETS = ROOT / "app" / "assets"
REPORTS = ROOT / "reports"
WAFER_CMAP = ListedColormap(["#ffffff", "#9ecae1", "#d62728"])

# Typical suspects named in the wafer-map literature. A starting point for engineers,
# not a diagnosis: the real cause has to be confirmed with the fab's own process data.
SUSPECTS = {
    "Center": "non-uniform deposition, etch or CMP at the wafer centre",
    "Donut": "ring-shaped non-uniformity, often thermal or CMP related",
    "Edge-Loc": "a local edge problem, e.g. handling or a clamp contact point",
    "Edge-Ring": "edge effects in etch, deposition or edge-bead removal",
    "Loc": "localised contamination or particles",
    "Near-full": "a major excursion: wrong recipe or a tool failure",
    "Random": "widespread contamination or an unstable process",
    "Scratch": "mechanical damage from wafer handling or CMP",
    "none": "no systematic pattern; only the baseline loss every wafer has",
}

st.set_page_config(page_title="Wafer defect & yield explorer", page_icon="🟦", layout="wide")


@st.cache_resource
def load_model() -> lgb.Booster:
    return lgb.Booster(model_str=gzip.decompress((ASSETS / "model.txt.gz").read_bytes()).decode())


@st.cache_data
def load_samples() -> dict:
    with np.load(ASSETS / "samples.npz") as z:
        return {k: z[k] for k in z.files}


@st.cache_data
def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def predict(wafer: np.ndarray) -> pd.Series:
    feats = extract_features(wafer[None])
    proba = load_model().predict(feats)[0]
    return pd.Series(proba, index=FAILURE_TYPES).sort_values(ascending=False)


def plot_wafer(wafer: np.ndarray):
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(wafer, cmap=WAFER_CMAP, vmin=0, vmax=2, interpolation="nearest")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    return fig


def parse_upload(file) -> np.ndarray:
    grid = np.loadtxt(file, delimiter=",", dtype=float)
    if grid.ndim != 2 or not np.isin(grid, (0, 1, 2)).all():
        raise ValueError("expected a 2-D grid of 0 (no die), 1 (pass) and 2 (fail)")
    return resize_nearest(grid.astype(np.uint8), IMG_SIZE)


yield_res = load_json(REPORTS / "results_yield.json")
loss = pd.DataFrame(yield_res["table"]).set_index("pattern")
BASE = yield_res["baseline_yield"]
info = load_json(ASSETS / "model_info.json")

st.title("Wafer defect & yield explorer")
st.caption(
    "Portfolio demo on public fab data (WM-811K wafer maps, SECOM process sensors). "
    "[Code and write-up on GitHub](https://github.com/join8112/wafer-yield-portfolio)"
)

tab_cls, tab_loss, tab_secom = st.tabs(
    ["Classify a wafer", "Yield loss by pattern", "Process signals (SECOM)"]
)

# --- 1. Classify a wafer ------------------------------------------------------------------
with tab_cls:
    samples = load_samples()
    names = np.array(FAILURE_TYPES)[samples["y"]]

    left, right = st.columns([1, 2], gap="large")
    with left:
        source = st.radio("Wafer source", ["Test-set example", "Upload CSV"], horizontal=True)
        wafer, true_label, die_yield = None, None, None
        if source == "Test-set example":
            choice = st.selectbox("Pattern", ["Any"] + FAILURE_TYPES)
            pool = np.flatnonzero(names == choice) if choice != "Any" else np.arange(len(names))
            if st.button("Show another wafer") or st.session_state.get("pool") != choice:
                st.session_state["pool"] = choice
                st.session_state["idx"] = int(np.random.default_rng().choice(pool))
            i = st.session_state["idx"]
            wafer, true_label = samples["X"][i], names[i]
            die_yield = float(samples["die_yield"][i])
        else:
            up = st.file_uploader(
                "Comma-separated grid: 0 = no die, 1 = pass, 2 = fail", type=["csv", "txt"]
            )
            if up is not None:
                try:
                    wafer = parse_upload(up)
                    die = wafer > 0
                    die_yield = float(1 - (wafer[die] == 2).mean()) if die.any() else None
                except ValueError as e:
                    st.error(str(e))
        if wafer is not None:
            st.pyplot(plot_wafer(wafer), width="content")
            st.caption("Red = failing die, blue = passing die")

    with right:
        if wafer is None:
            st.info("Upload a wafer map to classify it.")
        else:
            proba = predict(wafer)
            top = proba.index[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted pattern", top)
            c1.caption(f"{proba.iloc[0]:.0%} model confidence")
            if true_label is not None:
                c2.metric(
                    "True label",
                    true_label,
                    "correct" if true_label == top else "missed",
                    delta_color="normal" if true_label == top else "inverse",
                )
            if die_yield is not None:
                c3.metric(
                    "Die yield",
                    f"{die_yield:.0%}",
                    f"{(die_yield - BASE) * 100:+.0f} pts vs pattern-free wafers",
                )
            st.bar_chart(proba.rename("probability"), horizontal=True, height=280)

            if top in loss.index:
                r = loss.loc[top]
                if r.excess_per_wafer * 100 < 1:
                    st.markdown(
                        f"**What `{top}` costs:** almost nothing in die count. These wafers "
                        "yield about the same as pattern-free ones, though the pattern can "
                        "still flag a handling problem worth fixing."
                    )
                else:
                    st.markdown(
                        f"**What `{top}` costs:** on average {r.excess_per_wafer * 100:.0f} "
                        "yield points below a pattern-free wafer. Across the dataset it accounts "
                        f"for **{r.share:.0%}** of all pattern-caused loss."
                    )
            st.markdown(f"**Typical suspects:** {SUSPECTS[top]}.")
            st.caption(
                f"LightGBM on 15 handcrafted features. Test macro-F1 "
                f"{info['test']['macro_f1']:.3f}, accuracy {info['test']['accuracy']:.3f}. "
                "Suspects are common causes from the literature, to be confirmed with "
                "process data."
            )

# --- 2. Yield loss by pattern -------------------------------------------------------------
with tab_loss:
    st.subheader("Which defect pattern costs the most yield?")
    st.markdown(
        f"Every wafer loses some dies; pattern-free wafers average **{BASE:.1%}** yield. "
        "A pattern's cost is the loss *beyond* that, counted in wafer-equivalents "
        "(a wafer 10 points below baseline = 0.1 wafers lost). "
        "Total cost = how often the pattern appears × how much each occurrence loses."
    )
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.image(str(REPORTS / "figures" / "yield_severity_frequency.png"))
    with c2:
        show = loss.assign(
            mean_yield=lambda d: (d.mean_yield * 100).round(0),
            excess_per_wafer=lambda d: (d.excess_per_wafer * 100).round(0),
            excess_wafers=lambda d: d.excess_wafers.round(0),
            share=lambda d: (d.share * 100).round(0),
        )[["wafers", "mean_yield", "excess_per_wafer", "excess_wafers", "share"]]
        show.columns = ["wafers", "mean yield %", "pts lost / wafer", "total loss", "share %"]
        st.dataframe(show, width="stretch")
    st.markdown(
        f"- **Severity is not priority.** `Near-full` wafers are nearly scrapped, but are rare; "
        f"frequent mild patterns (`Center`, `Edge-Ring`, `Edge-Loc`) cost "
        f"{loss.loc[['Center', 'Edge-Ring', 'Edge-Loc'], 'share'].sum():.0%} of pattern loss.\n"
        f"- **Patterns explain only part of the loss:** "
        f"{yield_res['pattern_share_of_all_loss']:.0%} of all lost yield is excess caused by "
        "patterns; the rest is baseline loss spread across every wafer.\n"
        "- **The classifier is good enough to rank priorities:** on the test split, the top-3 "
        "patterns by loss come out in the same order with predicted labels as with true ones."
    )

# --- 3. SECOM -------------------------------------------------------------------------------
with tab_secom:
    sec = load_json(REPORTS / "results_secom.json")
    st.subheader("Can process sensors predict failing runs?")
    st.markdown(
        "Trained on the earliest 70% of production runs, tested on the latest 30%. "
        "A random model's PR-AUC equals the fail rate."
    )
    rows = {
        m: {
            "PR-AUC": sec[m]["pr_auc"],
            "95% interval": "{:.3f} - {:.3f}".format(*sec[m]["pr_auc_95ci"]),
            "ROC-AUC": sec[m]["roc_auc"],
        }
        for m in ("Logistic (L1)", "LightGBM")
    }
    rows["Random model"] = {
        "PR-AUC": sec["LightGBM"]["base_rate"],
        "95% interval": "",
        "ROC-AUC": 0.5,
    }
    st.dataframe(pd.DataFrame(rows).T.style.format(precision=3), width="stretch")
    st.markdown(
        f"**Barely better than chance, because the process drifted.** A classifier tells "
        f"training-period runs from test-period runs with AUC {sec['adversarial_auc']:.3f}, and "
        "the most important training sensor (`f059`) has no signal afterwards. A model trained "
        "once should not run as a fail alarm; it needs drift monitoring and retraining.\n\n"
        "**Sensors to check first:** `f033`, `f102`, `f000` (predictive before *and* after the "
        "shift) and `f116`, `f542`, `f058` (drifted most)."
    )
    st.image(str(REPORTS / "figures" / "secom_sensor_drift.png"), width=640)
