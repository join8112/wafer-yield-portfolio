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
_f5 = REPORTS / "findings_05_secom_model.md"
SECOM_MODEL_FINDINGS = (
    _f5.read_text(encoding="utf-8") if _f5.exists() else "## Findings\n\n_To be written._"
)

_f6 = REPORTS / "findings_06_yield.md"
YIELD_FINDINGS = (
    _f6.read_text(encoding="utf-8") if _f6.exists() else "## Findings\n\n_To be written._"
)

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

secom_model = [
    md(
        "# 05 · SECOM: predicting failing runs from process sensors\n\n"
        "**Goal:** flag failing production runs from 590 anonymised sensor readings, and find "
        "which sensors drive the prediction.\n\n"
        "**Setup** (decided in notebook 02)\n"
        "- **Time-ordered split:** train on the earliest 70% of runs, test on the latest 30%. "
        "This is how a fab would deploy the model, and notebook 02 showed the fail rate drifts.\n"
        "- **No leakage:** sensor dropping, imputation and scaling are fitted on training runs "
        "only, inside a scikit-learn `Pipeline`.\n"
        "- **Metrics:** PR-AUC (compare with the fail rate, which is what a random model scores), "
        "ROC-AUC, and *recall at 5% false alarms*: the share of failing runs caught if engineers "
        "inspect 5% of good runs. With only ~26 failing test runs, every number gets a bootstrap "
        "95% interval.\n"
        "- **Models:** L1 logistic regression (sparse, interpretable) and LightGBM, with fixed "
        "settings chosen before looking at the test period."
    ),
    code(
        "%matplotlib inline\n"
        "import json, warnings\nfrom pathlib import Path\n\n"
        "import matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd\n"
        "import lightgbm as lgb\n"
        "from sklearn.metrics import PrecisionRecallDisplay, roc_auc_score\n"
        "from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, "
        "cross_val_predict\n\n"
        "from wafer import secom as S\nfrom wafer.data import load_secom\n\n"
        "warnings.filterwarnings('ignore', category=UserWarning)\n"
        "FIG_DIR = Path('../reports/figures'); FIG_DIR.mkdir(parents=True, exist_ok=True)\n"
        "pd.set_option('display.precision', 3)"
    ),
    code(
        "X, meta = load_secom()\n"
        "y = meta['fail'].astype(int).to_numpy()\n"
        "tr, te = S.time_split(meta['timestamp'])\n"
        "pd.DataFrame({\n"
        "    'runs': [len(tr), len(te)],\n"
        "    'fails': [y[tr].sum(), y[te].sum()],\n"
        "    'fail_rate': [y[tr].mean(), y[te].mean()],\n"
        "    'from': [meta['timestamp'].iloc[tr].min(), meta['timestamp'].iloc[te].min()],\n"
        "    'to': [meta['timestamp'].iloc[tr].max(), meta['timestamp'].iloc[te].max()],\n"
        "}, index=['train', 'test'])"
    ),
    md("## 1. Sensor cleaning (fitted on training runs only)"),
    code(
        "cleaner = S.SensorCleaner().fit(X.iloc[tr])\n"
        "print(f'{cleaner.n_input_} sensors -> {len(cleaner.columns_)} after dropping '\n"
        "      'mostly-missing, constant and near-duplicate (|r| > 0.95) sensors')"
    ),
    md(
        "## 2. Random cross-validation vs. time-ordered test\n\n"
        "Many published SECOM results use random cross-validation. Comparing it with the "
        "time-ordered test shows how much that overstates performance."
    ),
    code(
        "%%time\n"
        "MODELS = {'Logistic (L1)': lambda: S.make_logreg(0.05), 'LightGBM': S.make_lgbm}\n"
        "rows, test_scores = [], {}\n"
        "for name, make in MODELS.items():\n"
        "    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)\n"
        "    cv_pr = [S.score_all(y[b], make().fit(X.iloc[a], y[a]).predict_proba(X.iloc[b])"
        "[:, 1])['pr_auc']\n"
        "             for a, b in cv.split(X, y)]\n"
        "    model = make().fit(X.iloc[tr], y[tr])\n"
        "    p = model.predict_proba(X.iloc[te])[:, 1]\n"
        "    test_scores[name] = (model, p)\n"
        "    r = S.score_all(y[te], p)\n"
        "    lo, hi = S.bootstrap_ci(y[te], p)\n"
        "    rlo, rhi = S.bootstrap_ci(y[te], p, roc_auc_score)\n"
        "    rows.append({\n"
        "        'model': name,\n"
        "        'random-CV PR-AUC': f'{np.mean(cv_pr):.3f} ± {np.std(cv_pr):.3f}',\n"
        "        'CV fail rate': f'{y.mean():.3f}',\n"
        "        'time-split PR-AUC': f\"{r['pr_auc']:.3f} [{lo:.3f}, {hi:.3f}]\",\n"
        "        'test fail rate': f\"{r['base_rate']:.3f}\",\n"
        "        'time-split ROC-AUC': f\"{r['roc_auc']:.3f} [{rlo:.3f}, {rhi:.3f}]\",\n"
        "        'recall @ 5% false alarms': f\"{r['recall_at_5pct_fpr']:.3f}\",\n"
        "    })\n"
        "results = pd.DataFrame(rows).set_index('model')\nresults"
    ),
    code(
        "fig, ax = plt.subplots(figsize=(6, 4.5))\n"
        "for name, (_, p) in test_scores.items():\n"
        "    PrecisionRecallDisplay.from_predictions(y[te], p, name=name, ax=ax)\n"
        "ax.axhline(y[te].mean(), ls='--', c='gray', label=f'random model ({y[te].mean():.3f})')\n"
        "ax.set_title('Precision-recall on the time-ordered test period'); ax.legend()\n"
        "plt.tight_layout(); plt.savefig(FIG_DIR / 'secom_pr_curve.png', dpi=150)"
    ),
    md(
        "## 3. Why is the time-ordered test so much harder? Checking for drift\n\n"
        "*Adversarial validation:* train a classifier to tell training-period runs from "
        "test-period runs. If it can (AUC near 1), the sensor readings themselves changed over "
        "time."
    ),
    code(
        "period = np.zeros(len(y)); period[te] = 1\n"
        "Xc = S.SensorCleaner().fit(X).transform(X)\n"
        "adv = lgb.LGBMClassifier(n_estimators=200, num_leaves=15, verbose=-1)\n"
        "adv_p = cross_val_predict(adv, Xc, period, method='predict_proba',\n"
        "                          cv=StratifiedKFold(5, shuffle=True, random_state=0))[:, 1]\n"
        "adv_auc = roc_auc_score(period, adv_p)\n"
        "imp = pd.Series(adv.fit(Xc, period).booster_.feature_importance('gain'), "
        "index=Xc.columns)\n"
        "print(f'adversarial AUC (train period vs test period): {adv_auc:.3f}')\n"
        "(imp / imp.sum()).sort_values(ascending=False).head(5).rename('share of gain')"
    ),
    md("## 4. Which sensors drive the prediction? (SHAP on LightGBM, coefficients for L1)"),
    code(
        "gbm, _ = test_scores['LightGBM']\n"
        "shap = S.lgbm_shap(gbm, X.iloc[tr])\n"
        "top = shap.abs().mean().sort_values(ascending=False)\n"
        "lr, _ = test_scores['Logistic (L1)']\n"
        "coef = pd.Series(lr.named_steps['clf'].coef_[0], index=lr.named_steps['clean'].columns_)\n"
        "nonzero = coef[coef != 0].abs().sort_values(ascending=False)\n"
        "print(f'L1 logistic keeps {len(nonzero)} sensors; '\n"
        "      f'{len(set(top.index[:20]) & set(nonzero.index[:20]))} of the top 20 overlap with "
        "SHAP')\n"
        "ax = top.head(15).sort_values().plot.barh(figsize=(6, 5))\n"
        "ax.set_xlabel('mean |SHAP value| (training runs)')\n"
        "ax.set_title('Top sensors for LightGBM')\n"
        "plt.tight_layout(); plt.savefig(FIG_DIR / 'secom_shap_top.png', dpi=150)"
    ),
    md(
        "## 5. Do the top sensors keep their signal over time?\n\n"
        "Single-sensor ROC-AUC in the training period vs the test period (0.5 = no signal)."
    ),
    code(
        "def single_auc(col, idx):\n"
        "    v = X[col].fillna(X[col].iloc[tr].median()).iloc[idx]\n"
        "    return roc_auc_score(y[idx], v)\n\n"
        "stab = pd.DataFrame({\n"
        "    'train AUC': [single_auc(c, tr) for c in top.index[:10]],\n"
        "    'test AUC': [single_auc(c, te) for c in top.index[:10]],\n"
        "}, index=top.index[:10])\n"
        "# Direction-free strength: 0.5 = none, 1.0 = perfect\n"
        "stab['train strength'] = (stab['train AUC'] - 0.5).abs() + 0.5\n"
        "stab['test strength'] = (stab['test AUC'] - 0.5).abs() + 0.5\n"
        "stab"
    ),
    code(
        "ax = stab[['train strength', 'test strength']].plot.bar(figsize=(7, 3.5))\n"
        "ax.axhline(0.5, c='gray', ls='--'); ax.set_ylim(0.45, 0.8)\n"
        "ax.set_ylabel('single-sensor AUC strength'); ax.set_title('Top-10 SHAP sensors: "
        "signal before vs after the split')\n"
        "plt.tight_layout(); plt.savefig(FIG_DIR / 'secom_sensor_drift.png', dpi=150)"
    ),
    md("## 6. Save results for the README"),
    code(
        "out = {name: S.score_all(y[te], p) for name, (_, p) in test_scores.items()}\n"
        "for name, (_, p) in test_scores.items():\n"
        "    out[name]['pr_auc_95ci'] = S.bootstrap_ci(y[te], p)\n"
        "out['adversarial_auc'] = adv_auc\n"
        "out['top_sensors_shap'] = top.head(10).round(4).to_dict()\n"
        "Path('../reports/results_secom.json').write_text(json.dumps(out, indent=2))\n"
        "out"
    ),
    md(SECOM_MODEL_FINDINGS),
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


yield_nb = [
    md(
        "# 06 · Defect pattern → yield: which pattern costs the most?\n\n"
        "**Question:** a wafer map already shows its yield, so the useful question is not *how "
        "much* yield was lost but *why*. Each defect pattern points to a different root cause "
        "(tool, process step, handling), so attributing lost yield to patterns tells engineers "
        "which problem to fix first.\n\n"
        "**Method**\n"
        "- *Baseline loss*: the mean yield of `none` wafers (training split). Every wafer loses "
        "this much anyway; a pattern's cost is the loss **beyond** it.\n"
        "- *Excess loss* per pattern = frequency × severity, in **wafer-equivalents** "
        "(a wafer 10 points below baseline counts as 0.10 wafers lost).\n"
        "- *Chained use*: on new wafers there is no label, so the pattern comes from the "
        "classifier. Section 4 checks whether loss attributed with **predicted** patterns "
        "matches the loss attributed with true labels on the test split."
    ),
    code(
        "%matplotlib inline\n"
        "import json\nimport warnings\nfrom pathlib import Path\n\n"
        "import lightgbm as lgb\nimport matplotlib.pyplot as plt\nimport numpy as np\n"
        "import pandas as pd\n\n"
        "from wafer import evaluate\nfrom wafer import yield_analysis as ya\n"
        "from wafer.data import FAILURE_TYPES\nfrom wafer.features import extract_features\n"
        "from wafer.preprocess import load_dataset\n\n"
        "FIG_DIR = Path('../reports/figures'); FIG_DIR.mkdir(parents=True, exist_ok=True)\n"
        "pd.set_option('display.precision', 3)\n\n"
        "ds = load_dataset()\n"
        "names = np.array(FAILURE_TYPES)[ds['y']]\n"
        "tr, va, te = (ds['split'] == s for s in ('train', 'val', 'test'))\n"
        "BASE = ya.baseline_yield(ds['y'][tr], ds['die_yield'][tr])\n"
        "print(f'baseline (none) yield: {BASE:.3f}')"
    ),
    md("## 1. Severity vs frequency"),
    code(
        "table = ya.yield_loss_table(ds['y'], ds['die_yield'], BASE)\n"
        "table"
    ),
    code(
        "order = table.index.tolist() + ['none']\n"
        "fig, ax = plt.subplots(figsize=(8, 4))\n"
        "ax.boxplot([ds['die_yield'][names == p] for p in order], showfliers=False)\n"
        "ax.set_xticks(range(1, len(order) + 1), order, rotation=30)\n"
        "ax.axhline(BASE, color='grey', ls='--', lw=1, label='baseline (none)')\n"
        "ax.set_ylabel('die yield'); ax.legend(); ax.set_title('Die yield by pattern')\n"
        "plt.tight_layout(); plt.savefig(FIG_DIR / 'yield_box_by_pattern.png', dpi=150)"
    ),
    code(
        "t = table[table['excess_wafers'] > 0]\n"
        "fig, ax = plt.subplots(figsize=(7, 5))\n"
        "ax.scatter(t['wafers'], t['excess_per_wafer'] * 100, s=t['excess_wafers'] * 3,\n"
        "           alpha=0.5, color='#4C72B0', edgecolor='k')\n"
        "# label offsets (points) so the crowded bubbles on the right stay readable\n"
        "offset = {'Center': (-10, 38), 'Edge-Loc': (25, -38), 'Loc': (-45, -12),\n"
        "          'Edge-Ring': (30, 32)}\n"
        "for p, r in t.iterrows():\n"
        "    dx, dy = offset.get(p, (0, 0))\n"
        "    ax.annotate(f\"{p}\\n{r['excess_wafers']:.0f}\",\n"
        "                (r['wafers'], r['excess_per_wafer'] * 100),\n"
        "                xytext=(dx, dy), textcoords='offset points',\n"
        "                ha='center', va='center', fontsize=8,\n"
        "                arrowprops=dict(arrowstyle='-', lw=0.6) if dx or dy else None)\n"
        "ax.set_xscale('log'); ax.set_xlabel('wafers with this pattern (log scale)')\n"
        "ax.set_ylabel('yield points lost per wafer, beyond baseline')\n"
        "ax.set_title('Severity x frequency (bubble = total excess loss, wafer-equivalents)')\n"
        "ax.margins(0.2); plt.tight_layout()\n"
        "plt.savefig(FIG_DIR / 'yield_severity_frequency.png', dpi=150)"
    ),
    md("## 2. Pareto: where does pattern-caused loss go?"),
    code(
        "ax = ya.plot_pareto(table)\n"
        "plt.savefig(FIG_DIR / 'yield_pareto.png', dpi=150)"
    ),
    md("## 3. How much of all lost yield do patterns explain?"),
    code(
        "total_loss = (1 - ds['die_yield']).sum()\n"
        "pattern_excess = table['excess_wafers'].clip(lower=0).sum()\n"
        "print(f'total loss: {total_loss:,.0f} wafer-equivalents')\n"
        "print(f'excess loss caused by patterns: {pattern_excess:,.0f} '\n"
        "      f'({pattern_excess / total_loss:.1%} of all loss)')"
    ),
    md(
        "## 4. Chained: attribute loss with the classifier's predictions\n\n"
        "Same LightGBM baseline as notebook 03, trained on the training split. On the test "
        "split, compare the loss attributed to each pattern using predicted vs true labels."
    ),
    code(
        "%%time\n"
        "F = extract_features(ds['X'])\n"
        "clf = lgb.LGBMClassifier(\n"
        "    n_estimators=2000, learning_rate=0.05, num_leaves=63,\n"
        "    class_weight='balanced', subsample=0.8, subsample_freq=1,\n"
        "    colsample_bytree=0.8, random_state=42, verbose=-1,\n"
        ")\n"
        "# eval_set works on every LightGBM 4.x; newer versions only warn that it is renamed\n"
        "warnings.filterwarnings('ignore', message='.*eval_set.*')\n"
        "clf.fit(F[tr], ds['y'][tr], eval_set=[(F[va], ds['y'][va])],\n"
        "        callbacks=[lgb.early_stopping(100, verbose=False)])\n"
        "pred = clf.predict(F[te])\n"
        "evaluate.summarize(ds['y'][te], pred, ds['n_fail'][te], name='LightGBM')"
    ),
    code(
        "attr = ya.attribution_error(ds['y'][te], pred, ds['die_yield'][te], BASE)\n"
        "attr"
    ),
    code(
        "a = attr[attr['true'] > 0]\n"
        "x = np.arange(len(a)); w = 0.4\n"
        "fig, ax = plt.subplots(figsize=(8, 4))\n"
        "ax.bar(x - w / 2, a['true'], w, label='true labels')\n"
        "ax.bar(x + w / 2, a['predicted'], w, label='predicted by classifier')\n"
        "ax.set_xticks(x, a.index, rotation=30)\n"
        "ax.set_ylabel('excess loss (wafer-equivalents)')\n"
        "ax.set_title('Test split: loss attributed with predicted vs true patterns')\n"
        "ax.legend(); plt.tight_layout()\n"
        "plt.savefig(FIG_DIR / 'yield_attribution_test.png', dpi=150)"
    ),
    code(
        "same_rank = list(attr.index[:3]) == list(attr.sort_values('predicted', "
        "ascending=False).index[:3])\n"
        "print('top-3 priority order identical with predicted labels:', same_rank)"
    ),
    md("## 5. Save results for the README and the app"),
    code(
        "results = {\n"
        "    'baseline_yield': round(BASE, 4),\n"
        "    'pattern_share_of_all_loss': round(float(pattern_excess / total_loss), 4),\n"
        "    'table': table.round(4).reset_index().to_dict(orient='records'),\n"
        "    'test_attribution': attr.round(3).reset_index().to_dict(orient='records'),\n"
        "}\n"
        "Path('../reports/results_yield.json').write_text(json.dumps(results, indent=2))\n"
        "table.round(3)"
    ),
    md(YIELD_FINDINGS),
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
    "05_secom_model.ipynb": secom_model,
    "06_yield_by_pattern.ipynb": yield_nb,
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
