# General Findings

Everything established **before** the Tier A / Tier C analyses. Covers the
engineering state of the pipeline, the corrected benchmark, and the four
scientific results that came out of the 2026-08-07 full paper run.

Companion documents: [`TIERA_FINDINGS.md`](TIERA_FINDINGS.md),
[`TIERC_FINDINGS.md`](TIERC_FINDINGS.md).

---

## 1. Provenance of the numbers in these documents

Every number below comes from the **full paper run** completed 2026-08-07 on a
`c6i.8xlarge`, not from a fast-mode read.

| | step7 (synthetic) | step9 (real markets) |
|---|---|---|
| `fast_mode` | `False` | `False` |
| seeds | 30 | 20 |
| max_iter / particles | 60 / 20 | 60 / 20 |
| folds (`n_splits`) | 8 | 8 |
| θ | 0.5 | 0.5 |
| benchmark | `v2` | — |
| provenance hash | `4e388f80e5ac` | `c4e9cd394b46` |

`apsoll_patience=1`, `apsoll_rearm_after=None` — i.e. the paper run executed
**legacy APSOLL**; the patience/re-arm fix was implemented but switched off.

Datasets: 5 synthetic v2 levels (null, white noise, AR(1), drift, regime
switch), 4 real (sector ETFs, Fama-French, bonds, commodities). All 14 datasets
passed the integrity gate at **100% label reproduction from their own base**.

**Seed-count decision (work order C6):** real-market runs use **20** seeds,
synthetic **30**. Manuscript §2.4 currently claims 30 everywhere and must be
amended to state both. The executed config is the source of truth.

---

## 2. Headline scientific results

### 2.1 The baseline is not beaten on the synthetic benchmark

Corrected v2 benchmark, 30 seeds, mean AUC:

| Level | Baseline | OrPSOC | +APSOLL | Full Hybrid |
|---|---|---|---|---|
| L1 White Noise | **0.9122** | 0.8841 | 0.8074 | 0.8879 |
| L2 AR(1) | **0.9103** | 0.8622 | 0.7879 | 0.8477 |
| L3 Drift | **0.8225** | 0.8160 | 0.7832 | 0.7884 |
| L4 Regime Switch | **0.7356** | 0.6994 | 0.6702 | 0.6687 |

Wilcoxon vs Baseline is `p=1.0000` on all four rows under
`alternative="greater"` — Full Hybrid is *worse*, not merely tied.

Baseline's `± 0.0000` is expected, not a bug: with no PSO it is deterministic
across all 30 seeds.

### 2.2 The compass is broken, and near its ceiling

Split-half reliability on `v2_regime_switch` gives a **Spearman-Brown corrected
ceiling of +0.718** — no causal criterion can correlate better than that with
test AUC.

| criterion | corr w/ test | % of ceiling |
|---|---|---|
| current | +0.201 | 28% |
| `median_k` | +0.244 | 34% |
| `univ_thresh` | +0.347 | 48% |
| `univ_sum` | +0.390 | 54% |
| `univ_sqrt` | +0.395 | **55%** |

`median_k` — the criterion locked by supervisor ruling — captures 34% of what
is achievable. `univ_sqrt` reaches 55%.

> **Caveat carried forward:** `univ_sum` was *retracted* as a recommendation
> earlier — it correlates **+0.978 with subset size**, so its argmax simply
> picks the largest allowed subset (k≈29). `univ_sqrt` has not been given the
> same scrutiny and must be checked for the same defect before use.

### 2.3 Selection is far more stable than chance

Ratio-to-null Jaccard (closed-form null, per-seed Wilcoxon):

| Level | condition | mean k | null J | obs J | **ratio** |
|---|---|---|---|---|---|
| L3 Drift | full_hybrid | 8.3 | 0.086 | 0.625 | **9.29×** |
| L4 Regime | full_hybrid | 9.4 | 0.093 | 0.412 | **5.45×** |
| L3 · LogReg/`median_k` | full_hybrid | 8.6 | 0.091 | 0.732 | **9.67×** |
| L4 · LogReg/`median_k` | full_hybrid | 9.0 | 0.096 | 0.726 | **8.94×** |

`standard_orpsoc` sits at 1.5–2.0×, so most of the stability comes from the
hybrid machinery rather than PSO alone.

### 2.4 Adaptation hurts on real data — replicated across asset classes

Frozen vs refitting univariate filter, identical rule and k, no search
(`experiments/ff_adaptation.py`, 4 classifiers × 3 subset sizes):

| market | frozen wins |
|---|---|
| sector_etf | 12/12 |
| fama_french | 11/12 |
| **bonds** | **12/12** |
| **commodities** | **11/12** |
| v2_regime_switch (synthetic) | **0/12** |

**46/48 real cells favour the frozen filter; 0/12 synthetic do.** The sign flips
exactly at the real/synthetic boundary, across three asset classes.

### 2.5 The APSOLL degeneracy is real — and repairing it changes nothing

The trigger is `c = (m/T)^(2/3) + 1` firing at `c < 1.05`, i.e. only when
`m < 0.05^1.5·T`. At T=60 that means **only m=0** — literally zero improvement.

Observed directly (`experiments/apsoll_sweep.py`, 360 units, 30 seeds):

```
rearm=None   re-arms/fold=[0,0,0]   triggers=[[6],        [6],        [8]]
rearm=10     re-arms/fold=[2,2,2]   triggers=[[6,32,58],  [6,32,58],  [8,34]]
```

Under the legacy setting the trigger fires **once at iteration 6 and never
again**. That is evidence, not algebra.

Repairing it does **not** help. The legacy control reproduced the paper run to
four decimals (harness valid), the re-arm engaged (1–2 per fold), and across
**20 comparisons**:

```
raw p < 0.05        1        expected by chance   1.0
survives Bonferroni 0        survives BH q<0.05   0
median |delta|      0.0032
```

Signs are inconsistent (`patience=3` helps drift, hurts regime switch). The
conclusion is a clean null: **the APSOLL trigger is not the binding
constraint.**

---

## 3. Engineering defects found and fixed

| # | Defect | Consequence | Status |
|---|---|---|---|
| 1 | `pyarrow` undeclared | pandas 3.0 backs `str` dtype with PyArrow; cached pickles unreadable on a clean box | fixed + pinned |
| 2 | `ModuleNotFoundError` swallowed as `SKIP` | a broken env reported as "no datasets (check network)" four stages later | fixed, now fatal |
| 3 | `.deps_installed` marker not content-keyed | fixing a dependency and re-running skipped the install and failed identically | keyed on `sha1(requirements.txt)` |
| 4 | step8 hardcoded **v1** level keys | `get_seed_aucs` returns `[]` on a miss, so tables printed all-`N/A` then died with `KeyError` | prefix derived from data + fail-fast guard |
| 5 | `ff_adaptation.py` hardcoded macOS path | killed stage 14 on EC2 | resolved from `__file__` |
| 6 | `stage ... \|\| true` did **not** protect | `fail()` calls `exit 1`; `exit` in a function kills the script, so `\|\| true` is never reached — stage 15 and the results tarball were lost | real `optional_stage()` |
| 7 | `n_jobs=6` hardcoded in 3 experiment scripts | ignored `ORPSOC_N_JOBS`; ran at 6 of 32 vCPUs (~5× slowdown, paid for on every EC2 run) | `default_workers()` |
| 8 | **No `__main__` guard** on step0–step8 | `import step7_ablation` executes the entire 4-hour ablation as a side effect | import guard added (exec-head idiom still works) |
| 9 | step9 discarded per-seed AUCs | blocked every paired per-seed test on real data | now persists `seed_aucs`/`seed_fold_aucs`; historical values recovered from checkpoints |
| 10 | Jaccard pre/post split used **hardcoded midpoint** | same defect `classify_folds()` fixed for folds; correct only by coincidence at n_splits=8 | `switch_pair` parameter, self-labelling output |

**Deliberately not fixed:** `orpsoc_utils.py` prints on import (cosmetic log
noise). It feeds *all three* provenance hashes, so editing it orphans 611
checkpoint units for a purely aesthetic gain.

### Leakage audit — clean

`FoldEvalContext` fits `SimpleImputer` + `StandardScaler` on `X_p` (the internal
PSO training split) only; `X_v` is transform-only. Per-column statistics make
fit-then-slice exactly equivalent to fit-on-subset, and the code asserts rather
than assumes on all-NaN columns. No leakage path found.

### Did fix #10 require a re-run? No.

Recomputed pre/post/drop from the saved `fold_selected` both ways:

```
condition             saved drop    midpoint  regime-aware   match?
standard_orpsoc          +0.0813     +0.0813       +0.0813      YES
apsoll                   +0.1170     +0.1170       +0.1170      YES
full_hybrid              +0.1085     +0.1085       +0.1085      YES
full_hybrid_noimp        +0.0525     +0.0525       +0.0525      YES
```

At n_splits=8 with a mid-series break, the midpoint lands on the true switch
pair. Published numbers stand; the fix is defensive for other configurations.

---

## 4. Elastic subset sizing (`experiments/pareto_knee.py`)

Tests whether k should be chosen from the data rather than fixed by θ. Per fold:
split the **training window** 75/25 (test fold never touched), rank on
inner-train, build the AUC-vs-k curve on inner-val, choose k by a knee rule,
then refit on the full training window and score the test fold.

**The classifier dominates the sizing rule:**

| dataset | LogReg: best arm vs all | LightGBM: best arm vs all |
|---|---|---|
| sector_etf | +0.0506 | −0.0075 (all wins) |
| fama_french | **+0.0807** (p=0.039) | −0.0240 (all wins) |
| bonds | +0.0444 | +0.0016 |
| commodities | **+0.0777** (p=0.016) | −0.0212 (all wins) |

Under LogReg every selection arm beats all-features on every real dataset.
Under LightGBM all-features wins nearly everywhere, and the knee rule becomes
the *worst* arm on commodities (−0.0794, p=0.016). Same data, same folds,
opposite conclusion — LightGBM is an embedded selector, so an external wrapper
competes with machinery the model already has.

### Retraction: the "Goldilocks / effective rank" explanation

An earlier hypothesis held that Fama-French failed because the algorithms cut to
5–9 features when ~21 (the effective rank) were needed. **Tested directly, this
is wrong:**

```
fama_french — LogReg
  effrank        k=17.5  ->  0.8088   (+0.0054, barely above baseline)
  fixed5         k=5.0   ->  0.8712
  knee_marginal  k=3.4   ->  0.8841   <- best arm in the table
```

On Fama-French **k≈3 wins and k≈18 does not**. "Too few features" was never the
problem. Effective rank is computed from the correlation matrix of `X` alone and
**never looks at `y`** — an unsupervised guess at how many features are needed
to predict a target it has not seen. The original claim generalised from a
Spearman of +0.62 on 8 points and did not replicate.

The pattern *does* hold on synthetic (`v2_regime_switch` has effective rank 42.6
of 50, and the `effrank` arm at k≈39 is the only one matching baseline).

---

## 5. Open items

- **Tier D not started.** D1 (external comparators: standard binary PSO, mRMR,
  LASSO/RFE) is flagged *required for journal submission*. D2 (a
  high-dimensional dataset) is the strongest principled route to a positive
  headline, since the central negative result is explicitly attributed to
  dimensionality.
- `univ_sqrt` must be checked for the size-proxy defect that disqualified
  `univ_sum`.
- Manuscript §4 items (M1–M7) are assigned to the author personally.
