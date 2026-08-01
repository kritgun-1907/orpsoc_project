# OrPSOC — Architecture, Workflow & Dead-Code Inventory

Companion to `ORPSOC_WORK_ORDER.md`. That document says *what still needs
doing*; this one says *how the thing actually works* — the execution model, the
data flow, what every file contributes, and what is not contributing anything.

Everything marked **[verified]** was checked by running or instrumenting the
code. Everything marked **[from comments]** is the authors' own claim, repeated
here but not independently re-derived.

---

## 1. What this project is

A feature-selection method (OrPSOC: **Or**thogonal-init **PSO** with
**C**rossover) evaluated under regime shift, using walk-forward cross-validation
on four synthetic datasets of increasing difficulty plus two real financial
datasets. The experiment is an **ablation**: five conditions, from a
no-selection baseline up to the full hybrid, measured with the same protocol so
each mechanism's marginal contribution is isolated.

---

## 2. How to run it

```bash
python step0_setup.py          # dependency check (once)
python step1_generate_data.py  # writes data/{white_noise,ar1,drift,regime_switch}.pkl
python step7_ablation.py       # THE synthetic experiment  -> results/step7_ablation.json
python step8_results.py        # figures + statistics       -> plots/, results/step8_*.json
python step9_real_data.py      # real-data experiment       -> results/step9_real_data.json
```

`step2`–`step6` are narrative/legacy and are **not** required for any reported
number (see §8).

### Execution knobs

| Knob | Where | Effect |
|---|---|---|
| `FAST_MODE` | top of step7 / step9 | reduced seeds/iters/particles/splits. **Changes results** — a FAST run is not comparable to a FULL run. |
| `N_JOBS` | top of step7 / step9 | worker processes. Parallelises **seeds only**. Does not change results. |
| `ORPSOC_N_JOBS` | environment | overrides `N_JOBS` without editing code |
| `USE_CHECKPOINTS` | top of step7 / step9 | resume completed seeds. Does not change results. |
| `PSO_FAST_EVAL` | `orpsoc_utils.py:72` | fitness model size (40 trees vs 100). **Changes results.** Currently `True`. |

```bash
ORPSOC_N_JOBS=32 python step7_ablation.py   # on a bigger machine
```

---

## 3. The execution model

```
step7_ablation.py
│
└── for level in [white_noise, ar1, drift, regime_switch]      SEQUENTIAL
    │   (loads data/<level>.pkl — a hard reset; nothing carries over)
    │
    └── for seed in 0..N_SEEDS-1                               PARALLEL (N_JOBS procs)
        │   (fresh AdaptiveRegimeThreshold + fresh warm-start chains)
        │
        └── for fold in 1..N_SPLITS                            SEQUENTIAL, ALWAYS
            │   (walk-forward: warm-start and detector history flow forward)
            │
            └── 5 conditions
                ├── baseline            all features, LightGBM
                ├── standard_orpsoc     run_standard_orpsoc()
                ├── apsoll              run_hybrid_orpsoc(hmm_trigger=False)
                ├── full_hybrid         run_hybrid_orpsoc(hmm_trigger=<HMM>)
                └── full_hybrid_noimp   as above, use_importance_reinit=False
```

**The unit of parallelism is one seed = one complete walk-forward sequence.**
Folds are never parallelised and never reordered.

### Why parallelising seeds is safe **[verified]**

1. Seeds were already independent: `hmm_threshold`, `warm_start_fh`, and
   `warm_start_fh_noimp` are constructed at the *top of the seed body*, so
   nothing crosses a seed boundary.
2. There is no global `np.random` use anywhere in the pipeline. Every draw comes
   from a local `np.random.RandomState(seed + fold_idx * 1000)` — a pure
   function of `(seed, fold)`. Execution **order cannot change any result**.
3. Confirmed empirically: 24 seed-jobs through a worker pool reproduced the
   serial results exactly; and Level 3 run *after* Levels 1–2 vs. run *alone*
   produced bit-identical fold AUCs, subsets, and detector diagnostics —
   including `p_trans` to all 16 significant digits.

---

## 4. The data, and exactly what is visible when

### Datasets

| File | Shape | Origin | Notes |
|---|---|---|---|
| `data/white_noise.pkl` | 1000 × 50 | step1 | Level 1 — no temporal structure |
| `data/ar1.pkl` | 1000 × 50 | step1 | Level 2 — stationary AR(1) |
| `data/drift.pkl` | 1000 × 50 | step1 | Level 3 — slowly shifting mean |
| `data/regime_switch.pkl` | 1000 × 50 | step1 | Level 4 — abrupt switch at t=500 |
| `data/sector_etf.pkl` | 6586 × 58 | step9 | 9 SPDR sector ETFs, daily from 2000 |
| `data/fama_french.pkl` | 6556 × 50 | step9 | FF 5-factor daily |

All six have **zero NaNs and zero all-NaN columns** **[verified]**.

Synthetic levels carry ground-truth signal columns: `signal_0..2` predictive in
regime 1, `signal_3..4` predictive in regime 2. That is what makes recall
measurable at all.

### Two nested causal splits

`walk_forward_folds(X, y, n_splits, gap=5, min_train)` yields
`(X_train, y_train, X_test, y_test, train_end)` where train is always a prefix
`[0, train_end)` and test starts `gap` rows later. Actual structure on
`regime_switch` **[verified]**:

```
fold   X_tr rows      inner X_p      inner X_v        X_te rows   gap
  1   [   0, 149]   [   0, 111]   [ 112, 149]   [ 155, 259]     5
  2   [   0, 254]   [   0, 190]   [ 191, 254]   [ 260, 364]     5
  3   [   0, 359]   [   0, 269]   [ 270, 359]   [ 365, 469]     5
  4   [   0, 464]   [   0, 347]   [ 348, 464]   [ 470, 574]     5
  5   [   0, 569]   [   0, 426]   [ 427, 569]   [ 575, 679]     5
  6   [   0, 674]   [   0, 505]   [ 506, 674]   [ 680, 784]     5
  7   [   0, 779]   [   0, 584]   [ 585, 779]   [ 785, 889]     5
  8   [   0, 884]   [   0, 662]   [ 663, 884]   [ 890, 994]     5
```

Inside each runner there is a **second** split of the *training window*:

```python
cut = int(len(X_tr) * 0.75)
X_p, y_p = X_tr.iloc[:cut], y_tr.iloc[:cut]   # PSO inner-train
X_v, y_v = X_tr.iloc[cut:], y_tr.iloc[cut:]   # PSO fitness target
```

**PSO fitness is AUC on `X_v`, which lives entirely inside the training
window.** The swarm never evaluates against `X_te`.

`X_te` is touched at exactly one place per run, after PSO finishes:

```python
pipe.fit(X_tr[sel], y_tr)                                    # refit on FULL train window
auc = roc_auc_score(y_te, pipe.predict_proba(X_te[sel])[:,1])  # score only
```

The `gap=5` exists because features are rolling-window statistics; without it
the last training rows and the first test rows share input observations.

On `sector_etf` (`min_train=500`, 8 splits) the training windows are
`train_end ∈ {500, 1260, 2020, 2780, 3540, 4300, 5060, 5820}` **[verified]**,
so the inner training split grows from 375 to 4365 rows across the run — which
is why later folds cost ~3× more than early ones.

### The full leakage surface

Only three things consume rows, and all three are confined to `[0, train_end)`:

| Consumer | Rows it sees |
|---|---|
| imputer + scaler (`FoldEvalContext`) | `X_p` only |
| PSO fitness (`evaluate_ctx`) | fits `X_p`, scores `X_v` |
| HMM detector + `windowed_feature_importance` | `X_tr` (trailing window for importance) |

Cross-fold state is causal too: `warm_start_fh` carries fold *k*'s `gbest_pos`
into fold *k+1*, and fold *k*'s training window is a strict subset of fold
*k+1*'s. `AdaptiveRegimeThreshold` computes its percentile from `prior_window`
— history *before* appending the current observation.

---

## 5. The five conditions

| Key | Label | What it adds |
|---|---|---|
| `baseline` | Baseline (all features) | No selection. LightGBM on all 50/58 features. Deterministic given the seed. |
| `standard_orpsoc` | Standard OrPSOC | Orthogonal init + two-point crossover, fixed `cr`, linear `w` decay. |
| `apsoll` | +APSOLL (no HMM) | Adds adaptive-`c`, three-leader GWO velocity, three-phase schedule. Self-triggered by stagnation. |
| `full_hybrid` | Full Hybrid | Adds the HMM regime detector, warm-start/elite preservation, `p_trans`-scaled burst, importance-guided reinit. |
| `full_hybrid_noimp` | Full Hybrid (no imp-reinit) | Identical to Full Hybrid **except** fresh particles are reinitialised orthogonally rather than from recent-window importances. The `full_hybrid − full_hybrid_noimp` delta is the isolated contribution of importance-guided reinit. |

Note: `full_hybrid` and `full_hybrid_noimp` are bit-identical on every fold up
to and including the first trigger, because `use_importance_reinit` is gated
behind `if hmm_trigger and use_importance_reinit`. On `ar1`, `drift`, and
`regime_switch` the first trigger is fold 5, so folds 1–4 are duplicate work
**[verified]**. This is left in place deliberately — see §9.

---

## 6. The engine (`orpsoc_utils.py`)

### Fitness

```
fitness = θ·AUC + (1-θ)·(1 - n_selected/n_features)      θ = 0.7
```

Scale-invariant, always in `[0,1]`. The compactness term is what drives subsets
below 50 features.

### PSO loop (`run_hybrid_orpsoc`)

Per iteration: update `c_t` → pick phase → compute `cr_t`/`w_t` → velocity
update → position sample via `sigmoid` → crossover → evaluate + update bests.

**Three phases:**

- **Phase 1 — exploration.** `cr=cr_low`, linear `w` decay, standard PSO
  velocity. Exits to Phase 2 on either the APSOLL stagnation trigger
  (`it > 5 and c_t < 1.05`) or the delayed HMM trigger (`it >= hmm_trigger_delay`).
- **Phase 2 — burst.** `cr`/`w` ramp up to `p_trans`-scaled targets over
  `ramp_iters`, GWO three-leader velocity. Runs `N_explore` iterations.
- **Phase 3 — decay.** Exponential blend back toward standard PSO.

Phases are **one-way** (`1 → 2 → 3`); there is no path back to Phase 1.

### Warm start / elite preservation

- `hmm_trigger=False` → `particles[0]` seeded from the previous fold's `gbest`
  (continuation).
- `hmm_trigger=True` → `elite_frac` of the swarm seeded from the carried
  `gbest` (one exact copy, the rest bit-flipped), the remainder reinitialised —
  orthogonally, or from `windowed_feature_importance` when
  `use_importance_reinit=True`.

### Detector

`SimpleHMM` (2-state Gaussian, Baum-Welch) is fit per fold on the rolling std
of the first feature; `p_trans = gamma[-1, 1]` feeds
`AdaptiveRegimeThreshold`, which fires when `p_trans` exceeds the 85th
percentile of **prior** history, subject to a warm-up guard
(`warmup_min_obs=150`) and a cooldown.

⚠ `SimpleHMM` and `get_hmm_trigger` exist as **two independent copies**, one
inlined in `step7_ablation.py` and one in `step9_real_data.py`. They are not
imported from a shared module. **Any detector change must be applied to both.**

### `FoldEvalContext` — the hoisted preprocessing

Added as part of the performance work (work-order 2.4.b). `evaluate()` rebuilt
`SimpleImputer` + `StandardScaler` inside a fresh sklearn `Pipeline` on *every*
particle evaluation — ~1700× per PSO run — always producing the same statistics,
because they only ever depend on `X_p`. `FoldEvalContext` computes them once per
fold; `evaluate_ctx()` then slices columns from the pre-transformed matrices.

Exactly equivalent because both transformers compute **per-column** statistics,
so fitting on all columns and slicing afterwards equals fitting on the subset
**[verified]**: `sc_all.mean_[cols] == sc_sub.mean_` and
`sc_all.scale_[cols] == sc_sub.scale_`, exactly. The fit stays on `X_p` only, so
the no-leakage requirement in 2.4.b holds, and `X_te` never enters the object.

---

## 7. File map

### Canonical — produces every reported number

| File | Role |
|---|---|
| `orpsoc_utils.py` | The engine. `evaluate`/`evaluate_ctx`, `FoldEvalContext`, `run_standard_orpsoc`, `run_hybrid_orpsoc`, `APSOLLAdaptiveC`, `AdaptiveRegimeThreshold`, `walk_forward_folds`, `feature_stability_ratio`. |
| `orpsoc_runner.py` | Execution infrastructure only: thread pinning, worker count, provenance hashing, `CheckpointStore`. Contains no modelling logic. |
| `step7_ablation.py` | Synthetic ablation, 4 levels × 5 conditions × N seeds × N folds. Own inlined `SimpleHMM` + `get_hmm_trigger`. |
| `step8_results.py` | Reads `results/step7_ablation.json` → figures + statistics. Runs no experiments. |
| `step9_real_data.py` | Real-data ablation. Downloads/builds datasets, then the same 5 conditions. **Own separate** inlined `SimpleHMM` + `get_hmm_trigger`. |

### Support

| File | Role |
|---|---|
| `step0_setup.py` | Dependency check/install. Run once. |
| `step1_generate_data.py` | Generates the four **v1** synthetic levels. Untouched — every number in the current manuscript draft comes from these. |
| `make_level0_null.py` | Generates `data/null.pkl` — Level 0, a true null (`y` independent of all features). The suite had no such control; L1 is a stationarity control, not a null. |
| `make_benchmark_v2.py` | Generates `data/v2_*.pkl` — corrected L0–L4. Fixes signal redundancy, signal/noise distinguishability, L3's non-drift, and L4's regime sets. **Does not overwrite v1.** v1 and v2 numbers are not comparable. |
| `test_equivalence.py` | Regression guard: 18 checks over walk-forward causality, `evaluate_ctx` equivalence, leakage, reproducibility, fold partitioning, and benchmark/objective invariants. |

### Legacy / pedagogical — **not** the pipeline

| File | Status |
|---|---|
| `step2_baseline.py` | Narrative step: establishes the all-features baseline story. |
| `step3_orpsoc_stationary.py` | Narrative step: first OrPSOC run, Level 2 only. |
| `step4_adaptive_crossover.py` | Standalone origin of the three-phase idea. Carries a canonical-engine banner. |
| `step5_hmm_detector.py` | Standalone HMM detector narrative + threshold sensitivity table. |
| `step6_apsoll_velocity.py` | Standalone hybrid implementation. Carries a canonical-engine banner. |

These import from `orpsoc_utils` but re-implement their own PSO loops. **Never
"fix" a reported number by editing one of these** — reported numbers come from
`orpsoc_utils` + `step7` + `step9` only (work-order G5).

---

## 8. Dead code inventory

Literal dead code, verified by call-graph search **[verified]**:

| Item | Location | Status |
|---|---|---|
| `partial_reinit()` | `orpsoc_utils.py:104` | **Dead.** Zero call sites anywhere. Still *imported* by `step7_ablation.py:49`, which makes it look live. Superseded by the elite-preservation logic inlined in `run_hybrid_orpsoc`. Note it is the only code in the repo that touches the global RNG (`np.random.randint`) — harmless while uncalled, but it would break order-independence if ever wired in. |
| `AdaptiveRegimeThreshold.calibrate_from_baseline()` | `orpsoc_utils.py:488` | **Dead.** Zero call sites. Only reachable if `method="cusum"`, which the canonical pipeline never selects. |
| `AdaptiveRegimeThreshold` CUSUM branch | `orpsoc_utils.py`, `update()` | **Dead in the canonical pipeline.** `step7` and `step9` both construct with `method="percentile"`. Only `step5_hmm_detector.py:312` uses `"cusum"`, and step5 is legacy. |
| `PSO_FAST_EVAL = False` branch | `orpsoc_utils.py:195`, `:293` | **Dead as configured.** The flag is `True` and nothing flips it. The 100-tree branch never executes during PSO. |
| `evaluate()` | `orpsoc_utils.py:179` | **No longer used by the canonical pipeline** — `run_standard_orpsoc`/`run_hybrid_orpsoc` now call `evaluate_ctx()`. Deliberately kept: `step3`/`step4`/`step6` still call it (22 call sites), and it is the reference implementation the equivalence test checks `evaluate_ctx` against. Not dead, but no longer on the hot path. |

Effectively dead (runs, but cannot influence any reported number):

| Item | Why |
|---|---|
| Everything in `step2`–`step6` | Legacy per G5. They regenerate their own plots and JSON, none of which feed `step7`/`step8`/`step9`. |
| `step5_hmm_detector.py` threshold-sensitivity sweep | Informative for the paper narrative; the canonical detector config (`percentile_k=85`) is hard-coded in step7/step9 and not read from step5's output. |

Not dead but worth knowing:

- The `(m/T)^(2/3)` curve in `APSOLLAdaptiveC` is **arithmetically inert** at
  the configured `max_iter`: `c < 1.05` requires `m < 0.0112·T`, which at
  `T=60` means only `m=0` can ever fire. The formula collapses to the boolean
  "did gbest fail to improve this iteration?" This is work-order item **2.1**
  and is *predicted from code reading*, **not yet empirically confirmed** —
  do not "fix" it before measuring it (guardrail G4).
- `full_hybrid_noimp` duplicates `full_hybrid` exactly until the first trigger
  (§5). Real but deliberate — see §9.

---

## 9. Performance

### Where the time goes **[verified]**

Essentially all of it is `evaluate_ctx()` → one LightGBM fit per particle
position. ~1650–1800 unique fits per PSO run (the position cache absorbs ~26%).
HMM, Jaccard, crossover, and bookkeeping are together under 1%.

### What was changed

| Change | Gain | Equivalence |
|---|---|---|
| `n_jobs=1` on all 9 `LGBMClassifier` sites | **4.3×** | LightGBM spawned one OpenMP thread per core for fits on a few hundred rows; sync overhead dominated. Threading parameter only — cannot affect output. |
| Seed-level process parallelism | **3.8×** | See §3. |
| `FoldEvalContext` (hoisted imputer/scaler) | **1.16×** | See §6. |

Measured end-to-end on `regime_switch` fold 7, one full-config hybrid run:
35.62 s → 8.31 s (`n_jobs`) → 7.19 s (hoist), AUC `0.9493575208` throughout.

Expected: `step7` ~25 h → **~5 h**; `step9` ~18 h → **~4 h**.

### Worker count

Reference machine is a **fanless MacBook Air M4** (4 performance + 6 efficiency
cores, 16 GB). Sustained throughput measured on the real workload **[verified]**:

| Workers | Throughput |
|---|---|
| 4 | 2.80× |
| 6 | **3.78×** |
| 8 | 2.60× |

Past 6 workers the efficiency cores and thermal throttling give the gain back,
which is why `default_workers()` caps at 6. On a fanned or server machine, raise
it with `ORPSOC_N_JOBS`.

### Deliberately *not* done

- **Shared cross-condition evaluation cache** (work-order 2.4.c). Measured on a
  full-config fold: 6833 evaluations summed across the four PSO conditions,
  6562 distinct — a shared cache saves **4.0%** **[verified]** and costs the
  G6 per-fold-scoping risk. Not worth it. Recommend closing 2.4.c as measured
  and rejected.
- **Skipping the duplicated `full_hybrid_noimp` runs** before the first trigger
  (~10%). Provably correct, but a subtly wrong guard would silently corrupt the
  importance-reinit ablation delta — the exact quantity condition 5 exists to
  measure — and the corruption would be invisible in the output. Bad trade.

---

## 10. Checkpointing & provenance

`orpsoc_runner.CheckpointStore` stores one file per **completed seed** under:

```
results/checkpoints/step7/<provenance-hash>/<level>_seed<NNN>.pkl
results/checkpoints/step9/<provenance-hash>/<dataset>_seed<NNN>.pkl
```

The provenance hash covers the run config (`fast_mode`, `n_seeds`, `max_iter`,
`n_particles`, `n_splits`) **and** the SHA-1 of `orpsoc_utils.py` plus the
runner script. Changing either produces a **new directory**, so stale results
can never be silently reused; old directories are kept, so re-running at a
previous config still resumes. The provenance record is also written inside
every checkpoint and re-verified on load, and writes are atomic
(`tmp` + `os.replace`) so a kill mid-write cannot leave a half-written file
that loads as valid.

Resuming is safe because a seed is a *complete, self-contained* walk-forward
sequence with order-independent RNG (§3). Nothing partial is ever stored.

**This replaced a real bug.** The previous scheme in `step9_real_data.py` keyed
on `data/checkpoint_{key}_{fast|full}_v3.pkl` — `FAST_MODE` and nothing else.
Changing `MAX_ITER`, `N_PARTICLES`, `N_SEEDS`, `N_SPLITS`, or any pipeline code
would have silently reloaded the old results and reported them as new: a direct
violation of guardrail **G3**, with nothing in the output to reveal it.

Output JSONs now carry a `provenance` block alongside `config`, so two result
files can be checked for comparability without guesswork.

---

## 11. Live issues you should know about

- **The saved `results/step7_ablation.json` is a FAST_MODE artifact.** Its
  config block reads `{fast_mode: True, n_seeds: 10, max_iter: 20,
  n_particles: 10, n_splits: 6}` **[verified]**, and its L1 baseline is 0.9927 —
  matching the manuscript's Table 2 value of 0.993. The manuscript's stated
  protocol is 30 seeds × 60 iterations × 20 particles × 8 folds. Every Tier A
  analysis in the work order is currently computed against FAST_MODE data.
  This resolves work-order **2.2.a** and points at **2.2.c** being a config
  difference rather than a code-correction effect.
- **Uncommitted behavioural change in the working tree:**
  `AdaptiveRegimeThreshold` default `cooldown` was changed `2 → 1`. This
  changes detector behaviour and therefore results. Left as-is (it looks
  deliberate) but it must be recorded before comparing against any earlier run.
- **`step5_hmm_detector.py` working tree reverts commit `7e37d46`**, restoring
  the fixed-window HMM fit instead of the per-fold refit. Legacy file per G5, so
  no reported number is affected, but the revert appears unintentional.
- **Two independent `SimpleHMM` copies** (step7, step9). Any detector change
  must be applied to both.
- **APSOLL trigger degeneracy** (work-order 2.1) is now **measured and
  confirmed** — this closes 2.1.b. At the paper config (60 iterations, 20
  particles), over 40 runs on `regime_switch`: first-fire histogram
  `{6: 26, 7: 10, 8: 4}`, never-fired `0/40`, max `c_t` = 1.1908 against a
  theoretical bound of 2.0, and the trigger condition true on 45–50 of 60
  iterations. The arithmetic is `c < 1.05 ⟺ m < 0.05^1.5·T`, so at `T=60` only
  `m=0` can fire. `+APSOLL` is therefore **standard OrPSOC plus one
  15-iteration burst at iteration ~7**, then ~98% standard PSO for the
  remainder (Phase 3 decays by `exp(-0.1·dt)`).

  **Manuscript impact (2.1.d):** O2 claims the mechanism needs "no manually
  tuned threshold". That is **false as implemented** — 1.05 is a hand-tuned
  constant tuned into degeneracy.

  The patience + re-arm fix is implemented behind
  `APSOLL_PATIENCE` / `APSOLL_REARM_AFTER`, **defaulting to legacy** so nothing
  changes until a seeded sweep justifies a value (G4, and 2.1.c requires the
  choice not be silent). A 5-seed read at `(patience=8, rearm=15)` recovered
  +0.031 AUC on `regime_switch` and reached parity with standard OrPSOC, but at
  n=5 the minimum attainable two-sided Wilcoxon p is 0.0625 and the observed p
  was 0.1875 — directional only, not a result.

- **Fold partitioning** now uses `orpsoc_utils.classify_folds()` against the
  actual break index instead of `fold_idx < N_SPLITS // 2`. At `n_splits=8`
  fold 4 is a **straddle** (test window `[470, 574]`, 71% post-switch) and was
  previously pooled into *pre*. Fold 4's training window is also entirely
  pre-switch, so **fold 5 is the earliest fold at which any method could
  adapt** — any recovery analysis counting fold 4 as a failure to adapt is
  measuring an impossibility. `fold_is_pre` is retained for step8 but its
  meaning changed (straddle now counts as post); new analysis should use
  `fold_phase` and report or drop the straddle fold.

- **Benchmark defects** (v1, all four levels). Signal features are five noisy
  copies of one latent (`sig_i = base + 0.3·randn`), so on L2 the best 3-of-5
  subset scores AUC 1.0000 and the remaining two add +0.0000. Recall-of-5 is
  therefore unachievable *and* undesirable, and it conflicts directly with the
  fitness function, whose optimum sits at k=3. `make_benchmark_v2.py` fixes
  this; measured v1 vs v2 marginal gain of the full signal set over the best
  3-subset: **+0.0003 (v1) → +0.0683 (v2)**, with the fitness argmax moving
  from k=3 to k=5. L3 v1 does not stress the model at all (baseline AUC
  *improves* 0.987 → 0.997 across folds); v2 L3 degrades 0.844 → 0.634 as
  intended.

- **On L4 specifically, recall-of-all-5 remains the wrong target even in v2**,
  because the signal set rotates: `s0–s2` predict regime 1 and `s3–s4` predict
  regime 2, so only 2–3 features are genuinely predictive in any given window.
  Use the regime-appropriate `fold_r1_hits` / `fold_r2_hits`, which the harness
  already records, rather than `fold_recall` (which unions all five).
