## Findings

**Question.** A wafer map already shows its yield, so the useful question is *why* the dies were
lost. Each defect pattern points to a different root cause, so attributing lost yield to patterns
tells engineers which problem to fix first.

**Method.** Pattern-free (`none`) wafers average **89.4%** die yield (training split). That is
the loss every wafer has anyway. A pattern's cost is the loss *beyond* it, summed over its wafers
in wafer-equivalents (a wafer 10 points below baseline counts as 0.10 wafers lost).

| Pattern | Wafers | Mean yield | Points lost per wafer | Total excess loss | Share |
|---|---|---|---|---|---|
| Center | 4,294 | 77% | 13 | 544 | 26% |
| Edge-Ring | 9,680 | 85% | 5 | 441 | 21% |
| Edge-Loc | 5,189 | 82% | 8 | 400 | 19% |
| Random | 866 | 52% | 37 | 325 | 16% |
| Loc | 3,593 | 85% | 4 | 159 | 8% |
| Near-full | 149 | 12% | 77 | 115 | 6% |
| Donut | 555 | 72% | 17 | 95 | 5% |
| Scratch | 1,193 | 90% | −1 | −6 | 0% |

1. **Severity is not priority.** `Near-full` is the most severe pattern (wafers are nearly
   scrapped) but ranks 6th by total loss because it is rare. Three frequent, mild patterns
   (`Center`, `Edge-Ring`, `Edge-Loc`) cause **67%** of pattern-caused loss. Ranking defects by
   how bad one wafer looks would send engineers to the wrong problem.
2. **`Center` is the first problem to fix.** It ranks first because it is both fairly common
   and fairly severe (13 points per wafer). Bringing those wafers back to baseline would raise
   the yield of the whole labeled set by about 0.3 points.
3. **`Scratch` costs no measurable yield.** Its wafers yield as well as pattern-free ones.
   A scratch can still flag a handling problem, but not a yield one.
4. **Patterns explain only about 10% of all lost yield.** The other 90% is the baseline loss
   spread across every wafer. Pattern classification is a tool for the systematic part.
   The random part needs process data, as in the SECOM analysis.
5. **The classifier is good enough for this use.** On the test split, loss attributed with the
   LightGBM model's *predicted* patterns is within 8% of the true attribution for every pattern
   that costs yield, except `Loc` (+16%: 205 test wafers labeled `none` are called `Loc`, and
   they carry some extra loss). The top-3 priority order is identical, so the chain also works
   on new, unlabeled wafers.

**Limitations.** Only the labeled 21% of WM-811K is used, and labeled wafers may not reflect the
fab's real production mix. Wafer-equivalents weight every wafer equally, whatever its die
count. The suspected causes shown in the demo are common explanations from the literature, not a
diagnosis.
