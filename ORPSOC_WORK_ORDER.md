# OrPSOC — Work Order & Verification Checklist

**Purpose.** This document is a hand-off spec. It lists everything that must be
verified, fixed, measured, or built in the OrPSOC codebase so the manuscript
(`Adaptive Feature Selection under Regime Shift`) can be completed. It is written
to be actionable without prior conversation context.

**Read `§0 GUARDRAILS` first and treat it as binding.** Several tasks below look
like "just fix it" but are actually "measure it, then decide." Getting that order
wrong destroys findings and wastes multi-hour compute runs.

---

## §0 GUARDRAILS (non-negotiable)

- [ ] **G1 — Never fabricate a number.** If a run has not completed, write
  `PENDING`. Do not interpolate, extrapolate, or "estimate" any AUC, p-value,
  or delta. Partial runs are not results.
- [ ] **G2 — Archive before overwriting.** `results/step7_ablation.json`,
  `results/step9_real_data.json`, and everything in `plots/` are overwritten
  in place with no versioning. Copy to `results/archive_<date>/` before any
  re-run.
- [ ] **G3 — Check config provenance before comparing any two numbers.**
  `step7_ablation.json` stores a `config` block
  (`fast_mode`, `n_seeds`, `max_iter`, `n_particles`, `n_splits`).
  Two numbers produced under different configs are **not comparable**. This has
  already caused one confirmed discrepancy (see T1.1).
- [ ] **G4 — Measure before mechanism changes.** Do not alter the APSOLL trigger
  (§2.1) or any detector logic before its current behaviour has been logged and
  quantified. The measurement is the finding; a silent fix erases it.
- [ ] **G5 — Legacy files are not the pipeline.** `step4_adaptive_crossover.py`,
  `step5_hmm_detector.py`, `step6_apsoll_velocity.py` are standalone /
  pedagogical. Reported results come from `orpsoc_utils.py` +
  `step7_ablation.py` + `step9_real_data.py`. Never "fix" a reported number by
  editing a legacy file.
- [ ] **G6 — Cache scoping.** If an evaluation cache is ever shared across
  conditions/seeds, it **must be scoped per-fold**. `evaluate()` depends on
  `X_p`/`X_v`, which change every walk-forward fold. A cache keyed only on the
  feature mask and shared across folds returns stale, wrong values silently.

---

## §1 REPO MAP

### Canonical pipeline (produces all reported numbers)
| File | Role |
|---|---|
| `orpsoc_utils.py` | Shared engine: `evaluate`, `run_standard_orpsoc`, `run_hybrid_orpsoc`, `APSOLLAdaptiveC`, `AdaptiveRegimeThreshold`, `walk_forward_folds` |
| `step7_ablation.py` | Synthetic ablation, 4 levels × 5 conditions. Has its own inlined `SimpleHMM` + `get_hmm_trigger` |
| `step8_results.py` | Reads step7 JSON → figures + statistical summary |
| `step9_real_data.py` | Real-data ablation (sector ETF + Fama-French). Has its **own separate** inlined `SimpleHMM` + `get_hmm_trigger` |

> ⚠️ `step7_ablation.py` and `step9_real_data.py` each contain an **independent
> copy** of `SimpleHMM` and `get_hmm_trigger`. Any detector change must be applied
> to **both**. They are not imported from a shared module.

### Support / legacy
| File | Role |
|---|---|
| `step1_generate_data.py` | Generates 4 synthetic levels. Untouched, believed correct. |
| `step2_baseline.py`, `step3_orpsoc_stationary.py` | Early narrative steps |
| `step4_*`, `step5_*`, `step6_*` | **Legacy/pedagogical.** Carry banners saying so. |

---

## §2 KNOWN DEFECTS — INVESTIGATE & RESOLVE

### 2.1 APSOLL stagnation trigger appears degenerate — **HIGH PRIORITY**

**Status:** Predicted from code reading + arithmetic. **NOT yet empirically confirmed.**

**Location:** `orpsoc_utils.py` — class `APSOLLAdaptiveC` (~line 314), trigger
consumed at ~line 899 inside `run_hybrid_orpsoc`.

**The mechanism as written:**
```python
# APSOLLAdaptiveC.update()
if self.prev_fit is not None and current_fit > self.prev_fit:
    self.m += 1
else:
    self.m = 0                      # full reset, not decrement
c = (self.m / max(self.max_iter, 1)) ** (2.0/3.0) + 1.0

# consumed in run_hybrid_orpsoc:
c_t = adap_c.update(gbest_fit)
apsoll_trigger = it > 5 and c_t < 1.05
```

**The arithmetic:**
```
c < 1.05  ⟺  (m/T)^(2/3) < 0.05  ⟺  m/T < 0.05^1.5 = 0.01118  ⟺  m < 0.01118·T
T=60 → m < 0.67 → only m=0 fires
T=20 → m < 0.22 → only m=0 fires
```
Ladder at T=60: `m=0 → c=1.000 (FIRES)`, `m=1 → 1.065`, `m=2 → 1.104`,
`m=10 → 1.303`, `m=30 → 1.630`. You would need `T ≥ 90` for `m=1` to be
*capable* of firing.

**Consequence A — the formula collapses to a boolean.** The entire
`(m/T)^(2/3)` curve and its `[1,2]` range are inert. The trigger is exactly
"did gbest fail to improve on this single iteration?" The code's own docstring
(~line 764) already states this.

**Consequence B (the serious one) — one-shot, wrong-time firing.**
- `c_t` is fed `gbest_fit`, which is **monotone non-decreasing** in PSO
  (improves or stays flat).
- Phases are **one-way**: `1 → 2 → 3`, never back to 1
  (verified: no `phase = 1` reassignment exists after init).
- Therefore the trigger fires **at most once per PSO run**, at the *first flat
  gbest iteration at or after `it=6`*.
- A single flat gbest iteration early in a run is routine. So this likely fires
  around iteration 6–9 in nearly every run, then can never fire again — including
  at iteration ~40 when the swarm has genuinely converged and *is* stagnant.

**Supporting circumstantial evidence:** `+APSOLL` is worst or joint-worst in
every reported table (synthetic L1 0.968, L2 0.979, L3 0.976, L4 0.827;
sector-ETF 0.790). Consistent with an unmotivated early diversity burst.

**Tasks:**
- [ ] **2.1.a — Instrument (do this first).** In `orpsoc_utils.run_hybrid_orpsoc`,
  record the iteration index at which `apsoll_trigger` first becomes true, plus
  whether it ever fires at all. Return it in the result dict (e.g.
  `apsoll_trigger_iter`, `None` if never). Propagate through `step7_ablation.py`
  into the per-seed records for **both** the `apsoll` and `full_hybrid`
  conditions.
  > Note: currently only `full_hybrid` logs trigger diagnostics. The `apsoll`
  > condition logs none — that gap is the whole reason this is unmeasured.
- [ ] **2.1.b — Measure.** Run `FAST_MODE = True` (cheap) and histogram
  `apsoll_trigger_iter` across all seeds/folds/levels.
  - Tightly clustered at ~6–9 → **degeneracy confirmed**, proceed to 2.1.c.
  - Widely spread across iterations → prediction wrong, close this item and
    record that it was checked.
- [ ] **2.1.c — Only if confirmed: fix + document.** Minimal correct fix is a
  **patience counter** — require `m == 0` for `k` consecutive iterations
  (analogous to early-stopping patience) instead of a single flat step —
  **and** allow `Phase 3 → Phase 1` re-arming so genuine late-run stagnation can
  be detected. Alternatives: decay `m` instead of resetting it; raise the
  threshold so non-zero `m` can fire. Do not pick one silently — report the
  before/after trigger-iteration distribution.
- [ ] **2.1.d — Manuscript impact.** O2 currently claims the mechanism requires
  "no manually tuned threshold." If 2.1.b confirms degeneracy, that claim is
  **false as implemented** (1.05 is a hand-tuned constant tuned into
  degeneracy) and must be corrected. The finding itself is publishable —
  same shape as the already-documented HMM defects.

### 2.2 Config provenance mismatch — **BLOCKING for Table 2**

- [ ] **2.2.a** Open the `config` block in the current `results/step7_ablation.json`
  and record: `fast_mode`, `n_seeds`, `max_iter`, `n_particles`, `n_splits`.
- [ ] **2.2.b** Compare against the manuscript's stated protocol (§2.4):
  *"8-fold walk-forward; 30 seeds × 60 iterations × 20 particles."*
- [ ] **2.2.c** Known discrepancy to resolve: manuscript Table 2 reports
  L1 baseline = **0.993**, but an observed in-progress run reported
  L1 baseline ≈ **0.950**. Determine whether this is (i) a config difference
  (different `n_splits` → different folds → different baseline) or
  (ii) a genuine effect of the code corrections. **Table 2 cannot be finalised
  until this is resolved.** If the configs differ, the two number sets were never
  comparable.
- [ ] **2.2.d** If the executed config differs from §2.4, either re-run at the
  stated config **or** amend §2.4 to state the actual configuration. Do not
  leave a table whose caption implies a protocol that was not run.

### 2.3 Detector — already fixed, verify still present

Three corrections were previously applied. Confirm all are present in **both**
`step7_ablation.py` and `step9_real_data.py`:
- [ ] **2.3.a** `SimpleHMM._emission_prob` uses `sig * np.sqrt(2*np.pi)`
  (multiplicative), **not** `sig + np.sqrt(2*np.pi)`.
- [ ] **2.3.b** `SimpleHMM.fit` floors sigma at `max(1e-4, 0.1*np.std(x))`,
  not a bare `+1e-4`.
- [ ] **2.3.c** `AdaptiveRegimeThreshold.update` (in `orpsoc_utils.py`)
  (i) does **not** `.pop()` the triggering observation, (ii) computes the
  percentile threshold from **prior** history only (excluding the current
  observation), (iii) exposes `last_raw_fire`, (iv) has a `cooldown` parameter.
- [ ] **2.3.d** `get_hmm_trigger` has a `warmup_min_obs` guard (default 150) and
  returns a 3-tuple `(triggered, p_trans, is_warmup)`.

### 2.4 Performance — partially applied

- [ ] **2.4.a** Verify `n_jobs=1` is present on **all 9** `LGBMClassifier`
  instantiations (5 in `orpsoc_utils.py`, 1 each in `step7_ablation.py`,
  `step9_real_data.py`, `step2_baseline.py`, `step3_orpsoc_stationary.py`).
  This was verified bit-identical in output — it changes threading only.
- [ ] **2.4.b — NOT applied, optional.** `evaluate()` rebuilds
  `SimpleImputer` + `StandardScaler` on every call. Extracting these to once
  per fold is a legitimate speedup. **If implemented:** fit the imputer and
  scaler on the internal training split **only** (`X_tr.iloc[:cut]`), then
  transform both halves. Fitting on the whole fold before splitting leaks
  validation statistics into the PSO's internal fitness signal.
- [ ] **2.4.c — NOT applied, optional.** Shared evaluation cache across
  conditions. See **G6** — must be per-fold-scoped. Also note
  `run_hybrid_orpsoc` is called 3× per fold (apsoll, full_hybrid,
  full_hybrid_noimp), each with its own empty cache today.

### 2.5 Legacy-file issues (low priority, cosmetic/consistency)

- [ ] **2.5.a** `step5_hmm_detector.py` fits the HMM **once** on `obs[:600]` and
  reuses fixed parameters across folds. Its own docstring already flags this.
  The canonical pipeline refits per fold and is unaffected. Fix for
  consistency, or add an explicit note.
- [ ] **2.5.b** `step6_apsoll_velocity.py` computes recall from a single
  end-of-run `gbest_pos` snapshot against both `signal_r1` and `signal_r2`.
  Since those two sets are predictive in *different* regimes, one snapshot
  cannot score well on both — this is an evaluation artifact, not an algorithm
  failure. The canonical pipeline tracks `fold_r1_hits` / `fold_r2_hits` /
  `fold_is_pre` per fold and handles it correctly.

---

## §3 ANALYSIS TASKS FROM THE MANUSCRIPT

Tiered by cost. **Tier A requires no new runs** — the data is already in
`results/step7_ablation.json` under `full_results`.

### Available fields (per level → conditions → per seed)
```
fold_aucs, fold_selected, runtimes, fold_recall,
fold_r1_hits, fold_r2_hits, fold_is_pre, jaccard
full_hybrid only: fold_triggered, fold_p_trans, fold_raw_fire, fold_is_warmup
Top level: summary{...seed_aucs, mean_recall_pre, mean_recall_post,
           mean_r1_hits_per_fold, mean_r2_hits_per_fold},
           importance_ablation, trigger_log, config
```

---

### TIER A — analysis only, zero re-runs

#### A1. Pre/post-switch recovery split (§3.5) — *"the critical missing piece"*
- [ ] **A1.a** Define the recovery metric **before** looking at results.
  Options: (i) folds-to-recovery within ε of pre-switch mean; (ii) area under
  the post-switch per-fold AUC curve. Write the definition down first.
- [ ] **A1.b** Split L4 into pre-switch (folds 1–3) and post-switch (folds 4–8)
  using `fold_is_pre`. Report mean AUC per group per condition.
- [ ] **A1.c** Paired per-seed Wilcoxon within each group.
- [ ] **A1.d** State the claim at exactly the supported strength. Current
  evidence: **baseline recovers fastest** (≈0.76 @ fold 5, ≈0.87 @ fold 6 vs
  variants ≈0.53–0.57 @ fold 5). If that holds, the correct claim is *"the
  adaptive mechanisms recover as fast as standard OrPSOC while selecting
  adaptively correct subsets"* — **not** "faster recovery than baseline."

#### A2. Recall of post-switch signal features (§3.6) — *turns Jaccard from suggestive to conclusive*
- [ ] **A2.a** Plot recall of `s3–s4` per fold per condition using
  `fold_r2_hits` / `mean_r2_hits_per_fold`.
- [ ] **A2.b** Test the specific claim: **recall of `s3–s4` rises from the
  switch fold onward for Full Hybrid but not for Standard OrPSOC.**
  Jaccard shows the subset *changed*; recall shows it changed to the *right*
  features. Without this, a reviewer attributes the Jaccard dip to noise.
- [ ] **A2.c** Also plot `fold_r1_hits` (pre-switch signals) — the mirror
  image should show `s0–s2` recall *falling* after the switch.

#### A3. Jaccard statistics (§3.6)
- [ ] **A3.a** Quantify the dip-and-recovery signature statistically: paired
  comparison of Full Hybrid Jaccard at the switch fold-pair vs adjacent
  fold-pairs, across seeds. The mean trace alone is insufficient given the
  wide variance band.
- [ ] **A3.b** Address L2 volatility: Full Hybrid Jaccard on *stationary* L2
  swings 0.37 → 1.0 → 0.41 → 0.75. Use `trigger_log` for L2 to determine
  whether this is spurious detections (a detector problem) or PSO stochasticity
  (a selector property). These have very different implications.
- [ ] **A3.c** Open question from the supervisor: can the wide variance band be
  reduced? Consider more seeds, median instead of mean, or CI bands.

#### A4. Compactness & interpretability (§3.7)
- [ ] **A4.a** Mean selected-subset size per condition per level, from
  `len(fold_selected[i])`. Baseline = 50 by definition.
- [ ] **A4.b** Explicit trade-off sentence, e.g. *"Full Hybrid attains X% of
  baseline AUC using Y% of the features."*
- [ ] **A4.c** Runtime per condition from `runtimes`.
- [ ] **A4.d** Produce a table and/or plot.

#### A5. Figure enhancements (§3.1, §3.3)
- [ ] **A5.a** Full pairwise significance matrix for Figure 1, not just the
  single asterisk. Currently only one significant result exists across the
  entire synthetic table: **Full Hybrid > +APSOLL on L4, p = 0.017** — and it
  is *variant-vs-variant*, not variant-vs-baseline. This must be stated
  explicitly in the caption/text so a reviewer does not infer it independently.
- [ ] **A5.b** Add mean-subset-size as a companion panel or secondary axis to
  Figure 1 — converts it from a pure loss report into the O4 trade-off statement.
- [ ] **A5.c** Overlay per-fold detector trigger state (from `trigger_log`)
  beneath the AUC panels in Figures 2 and 3, so cause (detection) and effect
  (behaviour change) are visible together.
- [ ] **A5.d** Optional: replace ±1 std with confidence intervals.

#### A6. Sector-ETF fold-2 observation (§3.3) — *potentially the paper's best real-data finding*
- [ ] **A6.a** **Pre-register the definition of "break-adjacent" before looking
  at results.** Proposed: `break_folds` and `break_folds + 1`. Writing this down
  first is what separates a finding from post-hoc fold selection.
- [ ] **A6.b** Paired per-seed comparisons of each variant vs baseline,
  restricted to break-adjacent folds only.
- [ ] **A6.c** Context: at fold 2 (first documented break) baseline drops to
  ≈0.73 while every selection variant holds ≈0.78–0.80. **This is the only
  point in the entire study where selection visibly beats the baseline**, and
  it occurs exactly where the method is theorised to help. If significant, this
  becomes the empirical centrepiece of the reframed O4 claim.
- [ ] **A6.d** Report per-fold selected-subset size alongside AUC.
- [ ] **A6.e** Add text explaining why the second break appears at fold 6 rather
  than the 2008 calendar date — the walk-forward causality argument (a break can
  only be detected once it has entered the *training* window, i.e. one fold
  after the fold whose *test* window contains it).

---

### TIER B — needs new instrumentation + one cheap `FAST_MODE` run
- [ ] **B1** APSOLL trigger-iteration histogram (§2.1.a–b above).
- [ ] **B2** L2 trigger log for A3.b, if not already in `trigger_log`.

---

### TIER C — needs the full re-run
- [ ] **C1** Regenerate Table 2 + Figure 1 from the corrected codebase with
  **all five conditions**, so table and figure report from a single code version.
- [ ] **C2** Complete the Fama-French run; replace the in-progress exhibit with
  a proper five-condition table + per-fold recovery figure matching §3.3's format.
- [ ] **C3** Report the Fama-French importance-reinit delta + paired Wilcoxon.
- [ ] **C4** Verify Fama-French detector trigger folds against the documented
  2008 / 2020 dates, as was done for sector ETFs (reconstruct calendar dates
  from raw price data and confirm each fire lands at the earliest causally
  possible fold).
- [ ] **C5** Report per-seed variance explicitly (std and range) for
  Fama-French and comment on selector stability. Observed spread ≈0.07–0.08 AUC
  across seeds, vs baseline pinned at 0.839 (deterministic). If this holds it
  strengthens the argument that **paired per-seed tests, not mean comparisons,
  are the correct statistical lens.**
- [ ] **C6** Decide and state: 20 seeds or 30 for Fama-French? Manuscript §2.4
  claims 30. Consistency matters.
- [ ] **C7** Regenerate Figure 3 (L4 recovery) with the fifth condition and
  annotate trigger state at folds 4–5.
- [ ] **C8** Regenerate Figure 4 (Jaccard) with the fifth condition.

---

### TIER D — new experiments

#### D1. External comparators (§3.8) — **required for journal submission**
Current ablation compares only against all-features and internal variants.
Reviewers will require external comparators under the *identical* walk-forward
protocol:
- [ ] **D1.a** Standard binary PSO — isolates whether the orthogonal-init +
  crossover machinery is worth anything.
- [ ] **D1.b** One filter method (e.g. mRMR).
- [ ] **D1.c** One embedded/wrapper method (e.g. LASSO, or RFE with LightGBM).
- [ ] **D1.d** All reported with the same metrics: AUC, recall, subset size.

#### D2. Statistical scope (§3.9) — **strategically the most important item**
Literature standard is ~30 seeds across ≥5 datasets. Current: 4 synthetic + 2 real.
- [ ] **D2.a** Option (a): add 1–2 further real datasets with a documented break.
- [ ] **D2.b** Option (b): add one genuinely **high-dimensional** dataset
  (hundreds to thousands of features).
  > **Why this matters most:** the paper's central negative result — that
  > external selection does not outperform the all-features baseline — is
  > explicitly attributed to dimensionality: a gradient-boosted classifier given
  > 50 features *internally* reweights feature importance across regime changes,
  > absorbing the selection problem. At 1000+ features this stops being true.
  > This is the strongest principled route to a positive headline result, and it
  > reframes the contribution as *"we identify the dimensionality regime where
  > external feature selection stops being redundant"* — a stronger claim than
  > the current one. **This is not p-hacking; it is a stated, testable hypothesis.**
- [ ] **D2.c** Decide explicitly between (a) and (b) and record the decision.

---

## §4 MANUSCRIPT / LaTeX ISSUES (not code)

- [ ] **M1** `\journal{Nuclear Physics B}` is the elsarticle template default.
  Change to the real target journal.
- [ ] **M2** Abstract, keywords, authors, affiliations, and graphical abstract
  are all **empty**.
- [ ] **M3** Everything after the **first** `\end{document}` is not compiled —
  this currently includes `\appendix` and the entire bibliography. References
  are effectively absent from the built PDF.
- [ ] **M4** Two sections are numbered **3.2**:
  `\subsection*{3.2 Real-data detector verification}` (starred, manually
  numbered, doesn't increment the counter) followed by
  `\subsection{Sector–ETF ablation}` which auto-numbers to 3.2. All downstream
  cross-references are shifted.
- [ ] **M5** Figure titles burned into the images contradict the LaTeX captions
  (image in Figure 3 reads "Figure 2 — AUC per Walk-Forward Fold"; image in
  Figure 4 reads "Figure 3 — Feature Stability"). Regenerate plots without
  internal figure numbers.
- [ ] **M6** O4 cross-references "section 3.5–3.7" but the relevant content
  renders at 3.6–3.8. Fix after M4.
- [ ] **M7** §3.1 text says *"The completed full-scale run..."* while Table 2's
  caption says *"Provisional ... (to be re-run on corrected code)."*
  Contradiction — resolve after §2.2.

---

## §5 SUGGESTED EXECUTION ORDER

1. **§0 guardrails** — archive existing results.
2. **2.2** — resolve config provenance. Nothing else is trustworthy until the
   provenance of the current numbers is known.
3. **2.3, 2.4.a** — verify prior fixes are still present.
4. **Tier A** — all of it. No re-runs needed; this is the bulk of the
   supervisor's requests and can be done immediately against existing JSON.
5. **2.1.a–b / Tier B** — instrument and measure the APSOLL trigger
   (cheap `FAST_MODE` run).
6. **2.1.c** — fix the trigger *only if* 2.1.b confirms degeneracy.
7. **Tier C** — the full re-run, once, after all mechanism changes are settled.
   Do not start this until steps 2–6 are closed.
8. **Tier D** — new experiments.
9. **§4** — manuscript cleanup.

---

## §6 REPORTING STANDARD

Every completed item should record:
- What was measured/changed, and **where** (file + line).
- The **config** under which any number was produced.
- Whether a claim is **verified empirically** or **predicted from code reading**
  (these are different and must not be conflated).
- If a prediction was tested and found **wrong**, say so explicitly and close
  the item. Negative verification results are valuable and must not be silently
  dropped.
