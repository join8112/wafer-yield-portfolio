## Findings

**Headline (test set, 34,585 wafers from lots never seen in training)**

| Metric | Score | Note |
|---|---|---|
| Macro-F1 | **0.861** | headline metric; every class counts equally |
| Yield-loss capture | **0.921** | 92% of failing dies on patterned wafers got the right pattern |
| Accuracy | 0.967 | looks high, but predicting `none` for everything already gives 0.852 |

**Where it works**
- `Edge-Ring` (F1 0.98), `Center` (0.90), `Random` (0.90) and `Near-full` (recall 1.00) are separated well by where the fails sit (rings) and how large the biggest failing cluster is. `Near-full` has only 30 test wafers, so its score is less certain.

**Where it struggles**
- **`Scratch` (recall 0.70) and `Loc` (0.76) are the weak classes.** The most common mistake is calling them `none` (17% of scratches, 12% of Loc wafers): they are *missed*, not confused with another pattern. Both are small, local features that ring averages dilute, and a thin diagonal scratch is hard to describe with hand-built shape features.
- `Loc` and `Edge-Loc` are also *over*-predicted (precision 0.67 and 0.73). Balanced class weights push the model to favour recall on rare classes over precision.

**Yield-loss capture is higher than macro-F1**, because the mistakes fall mostly on low-impact classes: `Scratch` wafers lose almost no extra dies (EDA: 89.9% yield vs 89.4% for `none`). From a fab's point of view, the baseline already catches most of the costly patterns.

**Which features matter:** the outer-middle ring fail rate (`ring3_fail`) and the shape of the largest failing cluster (fill, distance from centre, size) carry most of the signal. The edge-sector features matter less than expected, since `ring4_fail` and cluster distance already encode "near the edge".

**What the CNN has to beat:** macro-F1 0.861, with the clearest room for improvement on `Scratch` and `Loc`, the spatially local patterns where learning directly from the image should help most.
