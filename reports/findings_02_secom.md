## Findings

**Data**
- **1,567 runs x 590 sensors, 104 fails (6.6%).** A model predicting "pass" every time is 93.4% accurate, so evaluation uses **PR-AUC and recall at a fixed false-alarm rate**, not accuracy.
- **Missing data is widespread but concentrated:** 538 of 590 features have some missing values, but only 52 are missing more than 5% of the time. 28 features are more than half empty and 8 are more than 80% empty.
- **116 features are constant** (one value or all missing) and carry no information.
- **Heavy redundancy:** 329 feature pairs have |r| > 0.95, with groups of sensors that are perfect copies of each other (e.g. f074, f206, f209, f342, f347, f478 all |r| = 1.0).
- After dropping constant and mostly-empty features, roughly **450 usable features** remain, and many of those are duplicates.

**Drift over time**
- The weekly fail rate falls from **~10–23% in July–August to mostly 1–6% from September (one 9% week in early October)**, while run volume rises. Early weeks have few runs, so part of this is noise, but the trend is clear.
- A random train/test split would mix early high-fail runs into both sets and **overstate model performance**. A time-ordered split is closer to how the model would be used in a fab.

**Decisions for step 3**
1. **Drop** the 116 constant and 28 mostly-empty features; impute the rest with the median fitted on training data only.
2. **Reduce redundancy** by keeping one feature from each highly correlated group, then compare against PCA.
3. **Split by time:** train on the earliest ~70% of runs and test on the latest ~30%. Also report stratified cross-validation for comparison.
4. **Models:** logistic regression baseline, then LightGBM/XGBoost with class weights; explain with SHAP.
