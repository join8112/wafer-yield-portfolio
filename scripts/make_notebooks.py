"""Generate the EDA notebooks from source so they stay reviewable in git diffs.

Existing notebooks are skipped so executed outputs are never lost;
pass --force to regenerate them (this clears their outputs).

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


baseline = [
    md(
        "# 03 · Baseline: handcrafted features + LightGBM\n\n"
        "**Goal:** a fast, interpretable benchmark the CNN has to beat. Each feature describes a "
        "pattern the way an engineer would: *where* the failing dies are (rings, edge sectors) "
        "and *what shape* the biggest failing cluster has (size, position, elongation).\n\n"
        "**Setup** (decided in notebook 01)\n"
        "- 172,950 labeled wafers, resized to 64x64 with nearest-neighbour\n"
        "- train / val / test split stratified by class and **grouped by lot** (no lot in two "
        "splits)\n"
        "- metrics: macro-F1 (headline), per-class recall, and *yield-loss capture*: the share of "
        "failing dies on patterned wafers whose pattern was identified correctly\n\n"
        "Run `python scripts/prepare_wm811k.py` once before this notebook."
    ),
    code(
        "%matplotlib inline\n"
        "import json\nfrom pathlib import Path\n\n"
        "import lightgbm as lgb\nimport matplotlib.pyplot as plt\nimport numpy as np\n"
        "import pandas as pd\n\n"
        "from wafer import evaluate\nfrom wafer.data import FAILURE_TYPES\n"
        "from wafer.features import extract_features\n"
        "from wafer.preprocess import load_dataset\n\n"
        "FIG_DIR = Path('../reports/figures'); FIG_DIR.mkdir(parents=True, exist_ok=True)\n"
        "pd.set_option('display.precision', 3)"
    ),
    code(
        "ds = load_dataset()\n"
        "tr, va, te = (ds['split'] == s for s in ('train', 'val', 'test'))\n"
        "pd.crosstab(np.array(FAILURE_TYPES)[ds['y']], ds['split'])[['train', 'val', 'test']]"
    ),
    md("## 1. Feature extraction (~1 minute)"),
    code("%%time\nF = extract_features(ds['X'])\nF.describe().T"),
    md("## 2. Train LightGBM with early stopping on the validation split"),
    code(
        "%%time\n"
        "clf = lgb.LGBMClassifier(\n"
        "    n_estimators=2000, learning_rate=0.05, num_leaves=63,\n"
        "    class_weight='balanced', subsample=0.8, subsample_freq=1,\n"
        "    colsample_bytree=0.8, random_state=42, verbose=-1,\n"
        ")\n"
        "clf.fit(\n"
        "    F[tr], ds['y'][tr], eval_set=[(F[va], ds['y'][va])],\n"
        "    callbacks=[lgb.early_stopping(100, verbose=False)],\n"
        ")\n"
        "print('best iteration:', clf.best_iteration_)"
    ),
    md("## 3. Test-set results"),
    code(
        "pred = clf.predict(F[te])\n"
        "summary = evaluate.summarize(ds['y'][te], pred, ds['n_fail'][te], name='LightGBM')\n"
        "summary"
    ),
    code("evaluate.per_class_report(ds['y'][te], pred)"),
    code(
        "fig = evaluate.plot_confusion(ds['y'][te], pred, 'LightGBM baseline (test, "
        "row-normalised)')\n"
        "fig.savefig(FIG_DIR / 'baseline_confusion.png', dpi=150)"
    ),
    md("## 4. Which features matter?"),
    code(
        "imp = pd.Series(clf.booster_.feature_importance('gain'), index=F.columns)\n"
        "ax = (imp / imp.sum()).sort_values().plot.barh(figsize=(6, 5))\n"
        "ax.set_xlabel('share of total gain'); ax.set_title('LightGBM feature importance')\n"
        "plt.tight_layout(); plt.savefig(FIG_DIR / 'baseline_importance.png', dpi=150)"
    ),
    md("## 5. Save results for the README"),
    code(
        "report = evaluate.per_class_report(ds['y'][te], pred)\n"
        "results = {\n"
        "    'model': 'LightGBM on 15 handcrafted features',\n"
        "    **summary.round(4).to_dict(),\n"
        "    'recall_per_class': report['recall'].round(4).to_dict(),\n"
        "}\n"
        "Path('../reports/results_baseline.json').write_text(json.dumps(results, indent=2))\n"
        "results"
    ),
    md("## Findings\n\n_Send the outputs above to Claude to write this section._"),
]

COLAB_URL = (
    "https://colab.research.google.com/github/join8112/wafer-yield-portfolio/blob/main/"
    "notebooks/04_cnn_colab.ipynb"
)

cnn_nb = [
    md(
        f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_URL})"
        "\n\n# 04 · CNN on wafer maps (train on Colab GPU)\n\n"
        "**Goal:** learn defect patterns directly from the map instead of handcrafted features, "
        "and check *why* the model decides with Grad-CAM.\n\n"
        "**Model:** 3 convolutional blocks → global average pooling → linear (~0.3M parameters). "
        "Input has two channels, *is die* and *is failing die*.\n\n"
        "**Imbalance:** class-weighted loss (inverse square-root frequency) plus random 90° "
        "rotations and flips, which are valid because a defect pattern keeps its meaning when "
        "the wafer is rotated.\n\n"
        "**Before running on Colab**\n"
        "1. *Runtime → Change runtime type → T4 GPU*\n"
        "2. Colab runs on a temporary Google machine that is wiped when the session ends, so "
        "the data has to be sent there. Default (`USE_DRIVE = True`): put "
        "`wm811k_labeled_64.npz` in Google Drive at `MyDrive/wafer/`; data and results then "
        "survive disconnects. Or set `USE_DRIVE = False` to upload the file each session."
    ),
    code(
        "import sys\nfrom pathlib import Path\n\nIN_COLAB = 'google.colab' in sys.modules\n"
        "# True: read data from Google Drive and save results there (survives disconnects).\n"
        "# False: upload the .npz from your computer each session instead.\n"
        "USE_DRIVE = True\n\n"
        "if IN_COLAB:\n"
        "    REPO_URL = 'https://github.com/join8112/wafer-yield-portfolio.git'\n"
        "    REPO = '/content/wafer-yield-portfolio'\n"
        "    if not Path(REPO).exists():\n"
        "        !git clone -q {REPO_URL} {REPO}\n"
        "    %cd {REPO}\n"
        "    # Make the repo's package importable. (`pip install -e .` is not picked up by an\n"
        "    # already-running Colab kernel, so add src/ to the path directly.)\n"
        "    sys.path.insert(0, REPO + '/src')\n"
        "    if USE_DRIVE:\n"
        "        from google.colab import drive\n"
        "        drive.mount('/content/drive')\n"
        "        DATA = Path('/content/drive/MyDrive/wafer/wm811k_labeled_64.npz')\n"
        "        OUT = '/content/drive/MyDrive/wafer/outputs'\n"
        "    else:\n"
        "        from google.colab import files\n"
        "        DATA = Path('/content/wm811k_labeled_64.npz')\n"
        "        if not DATA.exists():\n"
        "            uploaded = files.upload()  # pick data/processed/wm811k_labeled_64.npz\n"
        "            Path(next(iter(uploaded))).rename(DATA)\n"
        "        OUT = '/content/outputs'\n"
        "else:\n"
        "    DATA = '../data/processed/wm811k_labeled_64.npz'\n"
        "    OUT = '../reports'"
    ),
    code(
        "%matplotlib inline\n"
        "import json\nfrom pathlib import Path\n\n"
        "import matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd\nimport torch\n\n"
        "from wafer import cnn, evaluate\nfrom wafer.data import FAILURE_TYPES\n"
        "from wafer.eda import WAFER_CMAP\nfrom wafer.preprocess import load_dataset\n\n"
        "OUT = Path(OUT); FIG_DIR = OUT / 'figures'; FIG_DIR.mkdir(parents=True, exist_ok=True)\n"
        "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
        "print('device:', DEVICE, torch.cuda.get_device_name(0) if DEVICE == 'cuda' else '')"
    ),
    code(
        "ds = load_dataset(DATA)\n"
        "tr, va, te = (ds['split'] == s for s in ('train', 'val', 'test'))\n"
        "print({s: int(m.sum()) for s, m in zip(('train', 'val', 'test'), (tr, va, te))})"
    ),
    md("## 1. Train (~10 minutes on a T4)"),
    code(
        "%%time\n"
        "model, history = cnn.train(\n"
        "    ds['X'][tr], ds['y'][tr], ds['X'][va], ds['y'][va],\n"
        "    epochs=15, batch_size=256, lr=2e-3, device=DEVICE,\n"
        ")\n"
        "torch.save(model.state_dict(), OUT / 'wafer_cnn.pt')"
    ),
    code(
        "h = pd.DataFrame(history).set_index('epoch')\n"
        "fig, ax1 = plt.subplots(figsize=(7, 3.5))\n"
        "ax1.plot(h.index, h['train_loss'], 'b-o', label='train loss')\n"
        "ax2 = ax1.twinx(); ax2.plot(h.index, h['val_macro_f1'], 'r-o', label='val macro-F1')\n"
        "ax1.set_xlabel('epoch'); ax1.set_ylabel('train loss'); ax2.set_ylabel('val macro-F1')\n"
        "fig.legend(loc='center right'); plt.title('Training curve'); plt.tight_layout()\n"
        "plt.savefig(FIG_DIR / 'cnn_training_curve.png', dpi=150)"
    ),
    md("## 2. Test-set results"),
    code(
        "probs = cnn.predict(model, ds['X'][te], device=DEVICE)\n"
        "pred = probs.argmax(1)\n"
        "summary = evaluate.summarize(ds['y'][te], pred, ds['n_fail'][te], name='CNN')\n"
        "summary"
    ),
    code("evaluate.per_class_report(ds['y'][te], pred)"),
    code(
        "fig = evaluate.plot_confusion(ds['y'][te], pred, 'CNN (test, row-normalised)')\n"
        "fig.savefig(FIG_DIR / 'cnn_confusion.png', dpi=150)"
    ),
    md(
        "## 3. Grad-CAM: what is the model looking at?\n\n"
        "For each class, one correctly classified test wafer. Red dies are failures; the "
        "heat-map overlay shows the regions that drove the prediction."
    ),
    code(
        "test_idx = np.flatnonzero(te)\n"
        "rng = np.random.default_rng(0)\n"
        "fig, axes = plt.subplots(2, len(FAILURE_TYPES), figsize=(2 * len(FAILURE_TYPES), 4.4))\n"
        "for k, name in enumerate(FAILURE_TYPES):\n"
        "    hits = test_idx[(ds['y'][te] == k) & (pred == k)]\n"
        "    for ax in axes[:, k]: ax.set_xticks([]); ax.set_yticks([])\n"
        "    axes[0, k].set_title(name, fontsize=9)\n"
        "    if len(hits) == 0: continue\n"
        "    i = rng.choice(hits)\n"
        "    cam, _ = cnn.grad_cam(model, ds['X'][i], target=k)\n"
        "    axes[0, k].imshow(ds['X'][i], cmap=WAFER_CMAP, vmin=0, vmax=2)\n"
        "    axes[1, k].imshow(ds['X'][i] > 0, cmap='gray', vmin=0, vmax=1.5)\n"
        "    axes[1, k].imshow(cam, cmap='jet', alpha=0.5)\n"
        "axes[0, 0].set_ylabel('wafer map'); axes[1, 0].set_ylabel('Grad-CAM')\n"
        "plt.tight_layout(); plt.savefig(FIG_DIR / 'cnn_gradcam.png', dpi=150)"
    ),
    md("## 4. Save results and copy them into the repo"),
    code(
        "report = evaluate.per_class_report(ds['y'][te], pred)\n"
        "results = {\n"
        "    'model': 'CNN (3 conv blocks, 2-channel 64x64 input)',\n"
        "    **summary.round(4).to_dict(),\n"
        "    'recall_per_class': report['recall'].round(4).to_dict(),\n"
        "    'epochs': len(history),\n"
        "}\n"
        "(OUT / 'results_cnn.json').write_text(json.dumps(results, indent=2))\n"
        "results"
    ),
    code(
        "if IN_COLAB:\n"
        "    # Downloads results_cnn.json, the figures and the model weights as one zip.\n"
        "    import shutil\n"
        "    from google.colab import files\n"
        "    shutil.make_archive('/content/cnn_outputs', 'zip', OUT)\n"
        "    files.download('/content/cnn_outputs.zip')"
    ),
    md(
        "Unzip `cnn_outputs.zip` into the repo's `reports/` folder (`results_cnn.json` → "
        "`reports/`, PNGs → `reports/figures/`). Then *File → Download → Download .ipynb* and "
        "replace `notebooks/04_cnn_colab.ipynb`, so the outputs show on GitHub."
    ),
    md("## Findings\n\n_Send the outputs above to Claude to write this section._"),
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


NOTEBOOKS = {
    "01_eda_wm811k.ipynb": wm811k,
    "02_eda_secom.ipynb": secom,
    "03_baseline_features.ipynb": baseline,
    "04_cnn_colab.ipynb": cnn_nb,
}

if __name__ == "__main__":
    import sys

    force = "--force" in sys.argv
    NB_DIR.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        if (NB_DIR / name).exists() and not force:
            print("skip (exists, use --force to overwrite):", name)
            continue
        write(cells, name)
