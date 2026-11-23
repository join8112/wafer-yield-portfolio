"""Generate the EDA notebooks from source so they stay reviewable in git diffs.

Note: regenerating overwrites the executed notebooks and clears their outputs;
re-run them afterwards.

python scripts/make_notebooks.py
"""

from pathlib import Path

import nbformat as nbf

REPORTS = Path(__file__).resolve().parents[1] / "reports"
WM_FINDINGS = (REPORTS / "findings_01_wm811k.md").read_text(encoding="utf-8")
SECOM_FINDINGS = (REPORTS / "findings_02_secom.md").read_text(encoding="utf-8")

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"

SETUP = """\
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from wafer import eda
from wafer.data import load_secom, load_wm811k

FIG_DIR = Path("../reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
pd.set_option("display.precision", 3)"""


def md(s):
    return nbf.v4.new_markdown_cell(s)


def code(s):
    return nbf.v4.new_code_cell(s)


wm811k = [
    md(
        "# 01 · WM-811K exploratory analysis\n\n"
        "**Goal:** understand the wafer-map data before building a defect-pattern classifier, "
        "and check whether defect patterns relate to die yield (the link used in step 4).\n\n"
        "**Questions**\n"
        "1. How many wafers are labeled, and how imbalanced are the 9 classes?\n"
        "2. How much do map sizes vary? (affects CNN input design)\n"
        "3. What does each defect pattern look like?\n"
        "4. Does die yield differ by defect pattern?"
    ),
    code(SETUP),
    code("df = load_wm811k()\nprint(f'{len(df):,} wafers')\ndf.head()"),
    md("## 1. Labeled vs unlabeled, and class imbalance"),
    code(
        "print(df['is_labeled'].value_counts(normalize=True).rename('share'))\n"
        "summary = eda.label_summary(df)\nsummary"
    ),
    code(
        "ax = summary['count'].sort_values().plot.barh(figsize=(7, 4), logx=True)\n"
        "ax.set_xlabel('wafers (log scale)')\nax.set_title('Labeled wafers per failure type')\n"
        "plt.tight_layout(); plt.savefig(FIG_DIR / 'wm811k_class_counts.png', dpi=150)"
    ),
    md(
        "> **Why it matters:** with `none` dominating, accuracy is misleading. "
        "Models will be compared on **macro-F1** and per-class recall, and training will use "
        "class weights / augmentation for rare classes such as `Near-full` and `Donut`."
    ),
    md("## 2. Wafer map sizes"),
    code(
        "shape_counts = df.groupby(['rows', 'cols']).size().sort_values(ascending=False)\n"
        "print(f'{len(shape_counts)} distinct map shapes')\nshape_counts.head(10)"
    ),
    code(
        "fig, ax = plt.subplots(figsize=(6, 4))\n"
        "ax.hist(df['n_dies'], bins=60)\nax.set_xlabel('dies per wafer'); ax.set_ylabel('wafers')\n"
        "ax.set_title('Dies per wafer'); plt.tight_layout()"
    ),
    md(
        "> **Why it matters:** maps come in many shapes, so the CNN needs a fixed input "
        "(resize to e.g. 64x64 with nearest-neighbour to keep die states discrete)."
    ),
    md("## 3. What each defect pattern looks like"),
    code(
        "fig = eda.plot_wafer_grid(df[df['is_labeled']], per_class=5)\n"
        "fig.savefig(FIG_DIR / 'wm811k_examples.png', dpi=150)"
    ),
    md("## 4. Defect pattern vs die yield"),
    code("eda.yield_by_failure_type(df)"),
    code(
        "fig = eda.plot_yield_by_type(df)\n"
        "fig.savefig(FIG_DIR / 'wm811k_yield_by_type.png', dpi=150)"
    ),
    md("## 5. Train / test split provided by the dataset"),
    code("lab = df[df['is_labeled']]\npd.crosstab(lab['failure_type'], lab['split'])"),
    md(WM_FINDINGS),
]

secom = [
    md(
        "# 02 · SECOM exploratory analysis\n\n"
        "**Goal:** understand 590 anonymised process measurements before predicting pass/fail.\n\n"
        "**Questions**\n"
        "1. How imbalanced is pass vs fail?\n"
        "2. How much data is missing, and which features carry no information?\n"
        "3. How redundant are the sensors?\n"
        "4. Does the fail rate drift over time? (decides random vs time-based split)"
    ),
    code(SETUP),
    code("X, meta = load_secom()\nprint(X.shape)\nmeta['fail'].value_counts()"),
    md("## 1. Class balance"),
    code("print(f\"fail rate: {meta['fail'].mean():.1%}\")"),
    md("## 2. Missing values and uninformative features"),
    code("eda.missing_profile(X)"),
    code(
        "fig, ax = plt.subplots(figsize=(6, 4))\n"
        "ax.hist(X.isna().mean(), bins=50)\n"
        "ax.set_xlabel('fraction missing'); ax.set_ylabel('features')\n"
        "ax.set_title('Missing rate per feature'); plt.tight_layout()\n"
        "plt.savefig(FIG_DIR / 'secom_missing.png', dpi=150)"
    ),
    code("low = eda.low_information_features(X)\n{k: len(v) for k, v in low.items()}"),
    md("## 3. Redundant sensors"),
    code(
        "pairs = eda.correlated_pairs(X.drop(columns=low['constant']), threshold=0.95)\n"
        "print(f'{len(pairs)} feature pairs with |r| > 0.95')\npairs.head(10)"
    ),
    md("## 4. Fail rate over time"),
    code(
        "weekly = eda.fail_rate_over_time(meta)\n"
        "fig, ax1 = plt.subplots(figsize=(8, 4))\n"
        "ax1.bar(weekly.index, weekly['runs'], width=5, alpha=0.3, label='runs')\n"
        "ax2 = ax1.twinx(); ax2.plot(weekly.index, weekly['fail_rate'], 'r-o', label='fail rate')\n"
        "ax1.set_ylabel('runs per week'); ax2.set_ylabel('fail rate')\n"
        "ax1.set_title('SECOM runs and fail rate per week'); plt.tight_layout()\n"
        "plt.savefig(FIG_DIR / 'secom_fail_rate_time.png', dpi=150)"
    ),
    md(SECOM_FINDINGS),
]


def write(cells, name):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    nbf.write(nb, NB_DIR / name)
    print("wrote", NB_DIR / name)


if __name__ == "__main__":
    NB_DIR.mkdir(exist_ok=True)
    write(wm811k, "01_eda_wm811k.ipynb")
    write(secom, "02_eda_secom.ipynb")
