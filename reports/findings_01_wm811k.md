## Findings

**Data**
- **811,457 wafers, only 21.3% labeled** (172,950). The 638k unlabeled maps are a candidate for semi-supervised learning later, but step 2 trains on labeled wafers only.
- **Extreme class imbalance:** `none` is 85.2% of labeled wafers. The rarest class, `Near-full`, has just **149 examples, ~990x fewer** than `none`. Accuracy is meaningless here: always predicting `none` already scores 85%.
- **632 distinct map shapes** (most common: 32x29, 25x27, 49x39).

**Defect pattern → yield**

| Pattern | Mean die yield | vs `none` (89.4%) |
|---|---|---|
| Near-full | 12.3% | −77 pts |
| Random | 51.9% | −38 pts |
| Donut | 72.3% | −17 pts |
| Center | 76.8% | −13 pts |
| Edge-Loc | 81.7% | −8 pts |
| Edge-Ring | 84.9% | −5 pts |
| Loc | 85.0% | −4 pts |
| Scratch | 89.9% | ≈ 0 |

- Pattern alone explains a large share of yield loss: `Near-full` and `Random` wafers are effectively scrap, while `Scratch` costs almost nothing in die count. For an engineer, **detecting Random / Donut / Center early matters more than catching every Scratch**, so per-class recall should be weighted by yield impact, not treated equally.
- Even `none` wafers range from 53% to 100% yield, so pattern is not the only driver of loss (motivates the process-side model in step 3).

**The dataset's own train/test split is unusual**
- `Edge-Ring` is 88% training, but `Edge-Loc`, `Loc`, `Near-full` and `Scratch` have *more* test than training wafers, and `none` is 75% test. Class mix differs sharply between the two halves.

**Decisions for step 2**
1. **Metric:** macro-F1 and per-class recall; also report a yield-weighted recall.
2. **Split:** re-split the labeled wafers myself, stratified by class and **grouped by lot**, so wafers from the same lot never sit in both train and test (avoids leakage from lot-level similarity). Keep the original split only as a secondary comparison.
3. **Input:** resize maps to 64x64 with nearest-neighbour interpolation so die states stay discrete (0/1/2).
4. **Imbalance:** class-weighted loss plus rotation/flip augmentation for rare classes (wafer patterns are rotation-invariant, so this is physically valid). Consider downsampling `none`.
