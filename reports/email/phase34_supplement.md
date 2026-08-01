# Phase 3 & 4 — Supporting Results

Supplement to the update email. Every number was produced by running the current
codebase; nothing is quoted from an earlier generation.

**Sample sizes, stated up front.** Phase 3 uses 5 seeds; at n=5 the minimum
attainable two-sided Wilcoxon *p* is 0.0625, so the significance tests below are
one-sample t-tests against a deterministic comparator. Phase 4 is one run per
cell — the filter and baseline arms are deterministic, so they have no seed
variance to report. The criterion bake-off in section 4 uses 5 independent candidate pools.
Two datasets throughout. These are strong directional results, not a 30-seed
protocol.

---

## 1. Adaptation hurts on real data

Same filter rule, same k, one refitted on every fold's training window and one
frozen on the first window and never updated.

![adaptation](figs/A_adaptation.png){width=full}

| | k=5 | k=10 | k=20 | frozen wins |
|---|---|---|---|---|
| sector ETF · LogReg | −0.0141 | −0.0054 | −0.0044 | 3/3 |
| sector ETF · LightGBM | −0.0108 | −0.0162 | −0.0020 | 3/3 |
| v2 synthetic · LogReg | +0.0839 | +0.0708 | +0.0306 | 0/3 |
| v2 synthetic · LightGBM | +0.0742 | +0.0454 | +0.0294 | 0/3 |

The sign flips cleanly, 6/6 real cells against 6/6 synthetic cells.

**Mechanism.** The synthetic benchmark *defines* regime shift as the informative
feature set rotating: `s0–s2` are predictive only before the switch and `s3–s4`
only after (measured on v2 — pre-switch AUC(s0–s2)=0.938 vs AUC(s3–s4)=0.514;
post-switch 0.545 vs 0.943). Adaptation is therefore rewarded by construction.
Real financial regime shift moves the distribution without changing *which*
features are informative — sector volatility and momentum stay informative
through 2008 and 2020 — so re-ranking each fold chases sampling noise in the
most recent window and discards a stable, correct answer.

**Caveat for framing.** One real dataset supports this. Before it headlines the
paper it should be reproduced on Fama-French and ideally one further market.

---

## 2. Phase 3 — the existential comparison

![phase3](figs/B_phase3.png){width=full}

sector ETF, LogReg final model, mean AUC over 8 folds, ± std across 5 seeds:

| arm | AUC | vs baseline | n_sel |
|---|---|---|---|
| all features | 0.7955 ± 0.0000 | — | 50 |
| OrPSOC, current fitness | 0.8157 ± 0.0185 | +0.0202 | 13.0 |
| OrPSOC, fixed fitness (`min_k`) | 0.8283 ± 0.0090 | +0.0328 | 14.2 |
| **FROZEN filter, k=5** | **0.8504 ± 0.0000** | **+0.0549** | **5** |
| test-fold filter (NON-causal) | 0.8554 ± 0.0000 | +0.0599 | 10 |

- frozen filter vs OrPSOC(`min_k`): **+0.0220**, one-sample t = −4.89, **p = 0.008**
- frozen filter vs OrPSOC(current): **+0.0346**, t = −3.73, **p = 0.020**
- fixing the compass recovered **36%** of the gap (+0.0126, paired p = 0.29)

The frozen filter is deterministic — no seed variance — and lands within 0.005
of a filter that was allowed to see the test fold.

**Note on a corrected label.** Earlier experiments in this project called the
frozen filter an "oracle". It is not: it is causal, uses only the first training
window, and is fully achievable. That mislabelling understated the threat, by
presenting an achievable method as an unreachable bound. The genuinely
non-causal arm is the last row, and even that is only an upper bound on
*univariate filtering*, not on selection in general.

---

## 3. Phase 4 — collinearity, not capacity

![capacity](figs/C_capacity.png){width=full}

Frozen filter minus all-features, per classifier:

| classifier | sector ETF (real) | v2 synthetic |
|---|---|---|
| LogReg | **+0.0549** | −0.0309 |
| LinearSVM | **+0.0739** | −0.0258 |
| kNN(25) | −0.0255 | **+0.0468** |
| RandomForest | −0.0102 | −0.0155 |
| LightGBM | −0.0148 | −0.0313 |

kNN has no built-in feature selection at all, so the capacity hypothesis
predicts a large gain — and it **loses** 0.026 on real data. The variable that
does sort the table is **collinearity sensitivity**: sector ETF has 58 nominal
features spanning ~13 effective dimensions, which inflates coefficient variance
in linear models but leaves trees (splitting) and kNN (distance aggregation)
unaffected. Pruning to 5 features stabilises the linear fit and throws
information away for the others.

v2 shows the mirror image because it has almost no collinearity but 45 pure-noise
dimensions, which wreck a distance metric — hence kNN is the only gainer there.

---

## 4. The criterion bake-off

![criteria](figs/D_criteria_seeded.png){width=full}

Criterion isolated from search: **5 independent pools** of 100 candidate subsets
per fold, scored causally, then "what would a *perfect* search maximising this
land on?" Ranked by percentage of the achievable gap captured (0% = no better
than random picking), mean ± std across pools.

| criterion | sector ETF (real) | v2 synthetic |
|---|---|---|
| **median_k** | **33% ± 10.4** | **43% ± 12.3** |
| mean_sd | 32% ± 10.9 | 40% ± 7.4 |
| min_k | 30% ± 10.5 | 23% ± 11.6 |
| mean_k | 26% ± 5.7 | 38% ± 13.1 |
| mb_perf | 24% ± 16.9 | 43% ± 12.8 |
| current | 23% ± 13.4 | 28% ± 8.9 |
| **pooled** | 16% ± 21.3 | 28% ± 4.4 |
| **mb_stability** | **−36% ± 22.5** | −16% ± 14.8 |
| **mb_thresh** | **−61% ± 13.5** | −18% ± 15.3 |
| *all-features baseline* | *44%* | *55%* |

> **Correction.** An earlier version of this figure used a **single** candidate
> pool per fold and was circulated before replication. Its rankings do not hold
> up: it showed `min_k` best on real data (40%, actually 30%), `median_k` at 21%
> (actually 33%), and `pooled` at −25% (actually **+16%**, straddling zero). The
> per-pool spread is 5–22 percentage points, so single-draw rankings were not
> reliable. This table supersedes it.

**`median_k` is the best criterion on both datasets** — the only one that ranks
first in both panels. But the margins are not significant at 5 pools: +9.5pp
over `current` on sector ETF (paired p = 0.247, 4/5 pools) and +15.3pp on v2
(p = 0.094, 4/5). Against the runner-up `mean_sd` it is +0.6pp and +3.4pp —
effectively a tie.

**No criterion reaches the all-features baseline on either dataset** (best 33%
vs 44%; 43% vs 55%). Fixing the compass narrows the deficit; it does not close it.

![why](figs/E_why_failed.png){width=full}

**Why MB stability fails — and this result is robust.** `mb_thresh` is −61% ±
13.5 on real data with a *maximum across pools of −38%*: negative in every pool.
Correlation with test AUC is −0.019 (mean form) and −0.454 (classical threshold
form) — actively anti-correlated. With 58 features in ~13 effective dimensions,
selection frequency measures which member of a collinear cluster arbitrarily won
a bootstrap resample, not which features are useful; thresholding then
concentrates on the in-sample-strongest, least diverse set. This is not an
implementation artefact: the first version scored by *mean* frequency and was
size-blind (its argmax picked k ≈ 5.5 against ~16 for every size-neutral
criterion); rebuilding it in the classical threshold form made the result
**worse**, which rules that explanation out.

**Pooled validation is too noisy to rank, not reliably negative.** At 16% ± 21.3
on sector ETF it straddles zero. The mechanism identified earlier still holds and
is worth reporting: the HMM regime state is **0.392 correlated with the target**
there (y-mean 0.32 vs 0.738 across states), because the target is a
rolling-volatility threshold and the regime is a rolling-volatility state, so
conditioning on regime strips out the label variation. On v2, where that
correlation is 0.010, pooled is stable (28% ± 4.4). The idea is sound; it is
incompatible with *this dataset's* target definition.

## 5. Continuous feature weighting is mathematically inert

![weighting](figs/F_weighting.png){width=full}

Max |ΔAUC| over three random weight vectors in [0.05, 1], **with no scaler in the
pipeline**:

| model | max change |
|---|---|
| LightGBM | **0.000000** |
| RandomForest | 0.001082 |
| LogReg (L2) | 0.013806 |
| kNN (k=25) | 0.039387 |

For LightGBM the change is exactly zero: trees split on thresholds, and scaling a
column is a monotone transform, so no reachable split point changes.

The stronger result is that the pipeline standardises:

```
StandardScaler(X)  vs  StandardScaler(X · w):   max abs difference = 2.2e-15

z = (w·x − mean(w·x)) / std(w·x) = w(x − mean x) / (w · std x) = (x − mean x) / std x
```

**The weight cancels exactly, for every model.** A continuous-weight OrPSOC on
this pipeline would search a space in which the objective is literally constant.

This does not rule out the *hyperparameter* variant, which changes the model in
ways weighting cannot and tunes the embedded selector rather than competing with
it. That direction inherits the compass problem, since per-regime tuning relies
on the same trailing-window validation that §1 shows is counterproductive.

---

## 6. What is not yet done

- The full L1–L4 ablation has **not** been re-run on the v2 benchmark.
- No default is locked, pending the ruling in the email. On the replicated
  evidence `median_k` is the defensible pick (best on both datasets), not
  `min_k` as an earlier single-pool run suggested.
- The LogReg Paradox narrative is **unwritten**, pending the same ruling.
- Three pending changes each alter reported numbers and should go into one
  regeneration run: v2 data, `fold_recall_active`, and θ = 0.5.

### On `fold_recall_active`

A metric defect was found and fixed while auditing L4. Union recall over all five
signal features can reach 1.0 only by keeping the two or three features that have
gone inert after the switch; a selector that correctly drops them scores at most
3/5 = 0.60 pre-switch and 2/5 = 0.40 post-switch. **The union denominator pays
for dead weight and penalises correct adaptation** — the exact behaviour the
study exists to measure. `fold_recall_active` scores against the features
actually predictive in each fold's regime, so the attainable maximum is 1.0
everywhere; straddling folds return NaN rather than a number that would silently
mean two different things. Since §3.6 leans on recall to make the Jaccard result
conclusive, that section needs regenerating and the direction of the effect could
change.
