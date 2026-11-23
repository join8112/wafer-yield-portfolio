## Findings

**Headline: on a realistic time-ordered test, process sensors barely predict failures, and the reason is drift, not the model.**

| | Logistic (L1) | LightGBM |
|---|---|---|
| Random-CV PR-AUC (fail rate 0.066) | 0.192 ± 0.040 | 0.196 ± 0.052 |
| **Time-split PR-AUC** (fail rate 0.055) | 0.082 [0.044, 0.172] | 0.086 [0.054, 0.143] |
| Time-split ROC-AUC | 0.539 [0.420, 0.664] | 0.662 [0.570, 0.749] |
| Recall at 5% false alarms | 0.15 (4 of 26) | 0.04 (1 of 26) |

*Brackets: bootstrap 95% intervals. Test period: 471 runs, 26 failures.*

**1. Random cross-validation overstates performance about 2x.** Shuffled CV gives PR-AUC ~0.19, about 3x the fail rate. On the later, unseen time period it drops to ~0.08, only 1.5x the fail rate, and the confidence interval reaches down to random. Many published SECOM results use shuffled CV; this is why they look better than a model would perform in a fab.

**2. The sensor readings themselves shift over time.** A classifier can tell training-period runs from test-period runs with **AUC 0.996** (adversarial validation). A handful of sensors (f116, f542, f058) account for most of the difference, which looks like recalibration, maintenance or a process change around late September.

**3. The sensor–failure relationship changes too.** `f059` is by far the most important sensor in training (top SHAP value, and the largest L1 coefficient), and on its own separates failures with AUC 0.76. In the test period it carries no signal (0.47). Of the 10 most important sensors, 7 lose most of their signal after the split; only `f033`, `f102` and `f000` keep theirs. This is *concept drift*: the model learned real patterns, but they stopped holding.

**4. The two models agree on what mattered.** L1 logistic regression keeps 63 of 274 sensors; 7 of its top 20 overlap with LightGBM's top 20 SHAP sensors, and both rank `f059` first. So the signal in the training period is real, not a modelling artefact.

**What this means for a fab**
- A static model trained once is **not** ready to run as a fail alarm: at a 5% false-alarm budget it catches at most 4 of 26 failures.
- It *is* useful as a **sensor shortlist** for engineers: `f033`, `f102` and `f000` predicted failures both before and after the shift (the most trustworthy leads), `f059` was the strongest signal until late September, and `f116`, `f542`, `f058` are the sensors that changed most. All three lists are worth a process-engineering review.
- A next model should favour sensors whose signal is **stable over time**, not just strong in the training window.
- Deployment would need **drift monitoring** (e.g. the adversarial check above run weekly) and **regular retraining** on recent runs.

**Limitations:** only ~3 months of data and 104 failures in total, so every test number has a wide interval. Sensor names are anonymised, so domain interpretation stops at "which sensor", not "which tool".
