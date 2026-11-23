## Findings

**Headline: the CNN edges out the feature baseline overall, and the two models make different mistakes.**

| Metric (test, 34,585 wafers) | LightGBM baseline | CNN | Change |
|---|---|---|---|
| Macro-F1 | 0.861 | **0.868** | +0.007 |
| Yield-loss capture | 0.921 | 0.921 | ≈ 0 |
| Accuracy | 0.967 | 0.973 | +0.006 |

| Recall by class | LightGBM | CNN |
|---|---|---|
| Scratch | 0.70 | **0.83** |
| Center | 0.90 | **0.93** |
| Donut | 0.82 | **0.86** |
| Random | 0.87 | **0.90** |
| Edge-Ring | 0.98 | 0.98 |
| Edge-Loc | **0.87** | 0.85 |
| Near-full | **1.00** | 0.97 (1 of 30 missed) |
| Loc | **0.76** | 0.68 |

**What the CNN fixed:** `Scratch` recall rose from 0.70 to 0.83. A scratch is a thin line, which ring averages and cluster statistics describe poorly but convolutions pick up directly. This was the baseline's biggest weakness.

**What got worse:** `Loc` recall fell from 0.76 to 0.68. The CNN now confuses `Loc` with `Center` (9%) and `Edge-Loc` (6%): a local cluster's *position* decides between these three classes, and the handcrafted "distance of the largest cluster from the centre" feature encodes position more explicitly than a CNN with global average pooling.

**How confident is the +0.007?** Not very. This is a single training run with one random seed, and validation macro-F1 still moved between 0.76 and 0.85 from epoch to epoch. The honest reading is that **the two models are roughly on par overall but complementary**: the CNN is better on shape (Scratch), the features are better on position (Loc). That makes an ensemble the natural next step.

**Grad-CAM sanity check:** the model looks at the right dies. Heat sits on the wafer edge for `Edge-Ring`, on the line for `Scratch`, on the local blob for `Loc`, and on the ring for `Donut`. This matters for a fab: a model that scored well by keying on artefacts (map size, die count) would not transfer to new products.

**Training:** 15 epochs, 16 minutes on a free Colab T4. Validation macro-F1 was still noisy and rising at the end, so more epochs, a lower learning rate or averaging several seeds would likely help.

**Next steps for this model**
1. Average the CNN's and LightGBM's predicted probabilities (cheap, uses existing models).
2. Add the wafer's polar coordinates as extra input channels, so the CNN can use position the way the features do.
3. Report the mean and spread over 3–5 seeds instead of a single run.
