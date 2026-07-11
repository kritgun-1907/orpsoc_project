# OrPSOC — Fix Log (what changed and what you must re-run)

Scope: addressed the professor's outstanding technical gap (#2) plus the
internal inconsistencies flagged in review. Every change was syntax-checked;
`orpsoc_utils.py` and the new code paths were functionally smoke-tested.

> ⚠ `step9_real_data__1_.py` was NOT on disk — the upload only contained
> `step_real_data.py`. All real-data edits below were applied to
> `step_real_data.py`. If your step9 is a genuinely different file, send it and
> re-apply these same edits there, or these fixes will not be in your step9.

---

## 1. Importance-guided reinit — the professor's suggestion #2 (NEW)
**File: `orpsoc_utils.py`**

- Added `windowed_feature_importance(X_tr, y_tr, feat_names, window_frac=0.4)`:
  fits a fast LightGBM on the **most recent** `window_frac` of the training
  window and returns a normalised importance vector (uniform fallback on any
  failure / single-class window).
- Added `build_importance_guided_positions(...)`: samples fresh particle
  positions biased by that importance vector (min_f guaranteed).
- `run_hybrid_orpsoc(...)` now, when `hmm_trigger=True` and
  `use_importance_reinit=True` (default), seeds the **non-elite** particles
  from those importances instead of the blind orthogonal draw. Elites are still
  preserved (partial restart / population memory is untouched).
- New params: `use_importance_reinit=True`, `importance_window_frac=0.4`.
- Works both with a warm start (step7 full-hybrid) and without (step_real_data).

This is the item you should NOT claim was done before — now it is.

## 2. Phase-2 ramp consistency (step4 & step6 no longer contradict the engine)
- **`step6_apsoll_velocity.py`**: Phase-2 now ramps cr/w up over `ramp_iters`
  (new param, default 5) instead of the instant `cr_low→cr_high` jump.
- **`step4_adaptive_crossover.py`**: `AdaptiveCRW.step()` transition phase now
  ramps over `ramp_iters` (new param). Verified trajectory:
  `0.3→0.42→0.54→0.66→0.78→0.9` then exponential decay.
- Both files got a **CANONICAL-ENGINE banner** stating they are standalone /
  pedagogical and that reported results come from `orpsoc_utils`.

## 3. `requirements.txt`
- Added `yfinance` (unpinned, with a comment). `step_real_data.py` hard-imports
  it and `SystemExit`s if missing — the old file silently omitted it.

## 4. Model-family consistency (all scoring now LightGBM)
Previously mixed: XGBoost in step2/step3-final/step4/step_real_data-baseline,
LightGBM everywhere else. Now standardised on LightGBM:
- **`step2_baseline.py`**: baseline model → `LGBMClassifier(100, num_leaves=31)`
  (identical to step7's baseline, so the numbers are comparable across tables).
- **`step3_orpsoc_stationary.py`**: final `quick_auc()` → LightGBM. (PSO fitness
  already used `orpsoc_utils.evaluate` = LightGBM.)
- **`step4_adaptive_crossover.py`**: local `evaluate()` model → LightGBM
  (fitness is still the legacy flat penalty — noted in the banner).
- **`step_real_data.py`**: `run_baseline()` → LightGBM (matches step7).
- Stale "XGBoost" comment in step7 header corrected to "LightGBM".

---

## YOU MUST RE-RUN (numbers will change)
- `step2_baseline.py`  → baseline AUC/degradation numbers change (XGB→LGBM).
- `step3_orpsoc_stationary.py` → the 3 final AUCs shift slightly.
- `step7_ablation.py` → full-hybrid now uses importance-guided reinit; the
  Level-4 post-switch recall / recovery numbers should change (ideally improve).
- `step8_results.py` → regenerate from the new step7 JSON.
- `step_real_data.py` → real-data baseline row changes (XGB→LGBM).

Run order unchanged: `step1 → step2 → step3 → step4 → step5 → step6 → step7 →
step8`, then `step_real_data.py`.

## STILL NOT DONE (be honest with the professor)
- **Fama-French 5-factor** dataset: not added. `step_real_data.py` still pulls a
  single ticker (SPY). Decide explicitly whether you're adding it.
- **Real-data breadth**: still ~10 engineered features on one asset, 2 of the 3
  breaks (no 2001). Consider adding SPDR sector ETFs to raise feature count so
  the *selection* story is demonstrable on real data.
- **Paper reframing** (feature selection under regime shift): a manuscript
  change, not a code change — nothing to verify here.

---

# ADDENDUM (after receiving step9_real_data.py + your result files)

## Correction to the review above
`step9_real_data.py` supersedes `step_real_data.py`. It ALREADY does Fama-French
AND the nine SPDR sector ETFs, builds 50+ feature matrices, uses all three
breaks (2001/2008/2020), and scores with LightGBM. So the three "STILL NOT DONE"
data items above were in fact already handled in step9. The old single-SPY
`step_real_data.py` has been REMOVED from this bundle to avoid two-versions
confusion; `step9_real_data.py` is the canonical real-data file.

## Importance-reinit ablation instrumentation (NEW — the point of this round)
To measure the ISOLATED contribution of importance-guided reinit, a fifth
condition `full_hybrid_noimp` was added — identical to Full Hybrid but with
`use_importance_reinit=False`. The Full-Hybrid − no-imp delta is the mechanism's
marginal effect.

- **`step7_ablation.py`**: added `full_hybrid_noimp` (own warm-start chain);
  per-fold HMM trigger logging (`fold_triggered`, `fold_p_trans`); an
  "IMPORTANCE-REINIT ABLATION" table (Δ AUC + paired one-sided Wilcoxon, overall
  and post-switch); an "HMM TRIGGER READOUT" (per-fold fire-rate + P(trans));
  both saved to the JSON as `importance_ablation` and `trigger_log`.
- **`step8_results.py`**: reads those fields; prints the ablation table; writes
  **Figure 5** (`step8_fig5_importance_reinit.png`) = L4 per-fold recovery for
  Baseline vs Full-Hybrid vs no-imp; prints trigger fire-rate. Degrades
  gracefully if run against an OLD step7 JSON (prints a "re-run step7" note
  instead of crashing). Existing Figures 1-4 are unchanged.
- **`step9_real_data.py`**: same fifth condition + trigger logging;
  per-dataset `importance_ablation` (incl. post-break-fold delta) and
  `trigger_fire_rate` saved to `step9_real_data.json`; recovery plot gains the
  purple no-imp line. Checkpoint filename bumped to `_v2` so your existing
  4-condition checkpoints are ignored (delete old `checkpoint_*.pkl` is not
  needed — the new name sidesteps them).

## How to read the new output
1. **Trigger fire-rate first.** If it doesn't spike near the switch/break folds
   — or if it fires on *every* fold — the detector is the problem, and no
   reinit change matters until that's fixed. (In a tiny smoke test the detector
   fired on all folds; verify this isn't happening on your real run.)
2. **Then the Δ AUC / Wilcoxon.** Positive + significant on regime_switch
   (especially post-switch) = importance-reinit earns its place. Null or
   negative = it doesn't help on this data; report that honestly.
3. **Do NOT expect Full Hybrid to beat Baseline.** Nothing here targets that.
   The all-features baseline still wins on AUC; the paper's case is
   compactness + stability (+ recovery, only if the ablation supports it).

## Re-run order
`step7 → step8` (regenerates the synthetic ablation plus Fig 5), and `step9` (real data). Old
`step7_ablation.json` / `step8_*` will be regenerated with the 5th condition.

---

# ADDENDUM 2 — the always-on HMM trigger bug (found from real-data checkpoints)

Your real-data checkpoints showed the HMM trigger firing on 7-8 of 8 folds on
BOTH datasets, every seed, with P(Transition) pinned at exactly 1.0 (fama_french)
or 1.0 with zero variance across all 20 seeds (sector_etf — p_trans doesn't
depend on the PSO seed at all, only on the data). I did not just patch this and
ship it — I tested three hypotheses in order, and the first one was wrong.
Recording that here on purpose: the honest version of debugging looks like this,
not "found it, fixed it" on the first guess.

**Attempt 1 (insufficient on its own):** floored `SimpleHMM`'s sigma at
`max(1e-4, 0.1*std(x))` instead of a bare `+1e-4`, reasoning that the low-vol
state's variance was collapsing on quiet historical stretches. Tested against a
realistic quiet/crisis/quiet/crisis series: **P(Transition) still read exactly
1.0 on every single fold**, including folds hundreds of days after a crisis had
clearly ended. This alone did not fix anything — kept in the final code as
reasonable general regularization, but it was NOT the cause.

**Attempt 2 (also insufficient alone):** Dirichlet-smoothed the transition
matrix M-step (`xi.sum(0) + prior`), on the theory that the high-vol state was
becoming near-absorbing (`A[1,0] ≈ 0`) because the data contains very few actual
"high→low" transition timesteps to estimate that entry from. Tested: `A[1,1]`
still converged to ~0.999-1.0 even with a prior — the data likelihood swamped a
sane pseudo-count. Not used in the final fix; not the root cause either,
though a real secondary risk worth knowing about.

**Actual root cause, confirmed by direct test:** `SimpleHMM._emission_prob()`
computed the Gaussian normalizing constant as `sigma + sqrt(2*pi)` (ADDITIVE)
instead of the correct `sigma * sqrt(2*pi)` (MULTIPLICATIVE). For a small
sigma — i.e. exactly the tight, confident "we clearly know this is quiet" state
— `sqrt(2*pi)≈2.51` dominates the sum and the denominator barely moves with
sigma. This silently destroys the core property that makes a narrow Gaussian
informative: it should assign sharply higher density to points near its own
mean than a fat Gaussian does. With the additive bug, the model's "this is
quiet" emission evidence was never strong enough to overcome the transition
prior once the "high-vol" state had ever been visited, so the posterior stuck
near 1.0 indefinitely.

**Fix, applied identically in `step7_ablation.py` and `step9_real_data.py`**
(their `SimpleHMM` classes are separate inlined copies, not shared — both
needed the same fix):
```python
# was:  probs[:, k] = np.exp(-0.5*(diff/sig)**2) / (sig + np.sqrt(2*np.pi))
probs[:, k] = np.exp(-0.5*(diff/sig)**2) / (sig * np.sqrt(2*np.pi))   # fixed
```

**Verified end-to-end** (not just unit-tested in isolation) with the actual
delivered `SimpleHMM` + `AdaptiveRegimeThreshold` classes, a fresh random seed,
and two full crisis windows inside a 2400-point walk-forward-style series:
fires exactly at the two folds touching a real crisis, reads 0.0 (no trigger)
on all 10 genuinely-quiet folds, `max_consecutive_raw_triggers = 1`. That is
the behavior a regime detector is supposed to have; the previous run had none
of it.

## Also fixed: `AdaptiveRegimeThreshold.update()` — the pop-on-trigger bug
Independent of the HMM bug above, `orpsoc_utils.py`'s `AdaptiveRegimeThreshold`
used to `.pop()` the triggering P(Transition) value out of its own history the
moment it fired — meant to stop one confirmed spike from raising the bar for
future folds, but if P(Transition) is chronically elevated (which the HMM bug
above was causing), every fold's trigger deletes its own evidence before the
percentile threshold can ever calibrate against it, guaranteeing perpetual
firing. Fixed: history is never discarded; a new `cooldown` parameter
(default 2) suppresses re-triggering for `cooldown` calls after a confirmed
detection instead. Also added `consecutive_raw_triggers` tracking with a
console WARNING at 4-in-a-row, specifically so this class of bug surfaces in a
2-minute FAST_MODE run instead of after a 16-hour FULL run next time.

Both fixes are needed together: the emission-formula fix stops P(Transition)
from being wrong; the cooldown fix stops the threshold logic from amplifying
any future glitch into a permanent lock-on. Neither alone fully explains what
you saw — I tested that, not assumed it.

## What to do next
1. `FAST_MODE = True` in `step9_real_data.py` first. Check the trigger
   fire-rate print — it should now spike near `break_folds` and stay near 0
   elsewhere, not print 1.0 on 7-8 of 8 folds.
2. Only after that looks sane, re-run `FAST_MODE = False` (the full run) and
   re-run `step7`/`step8` for the synthetic side too (same bug existed there,
   just less visible since Level 4's single sharp synthetic switch doesn't
   expose "does it correctly revert to quiet after" the way real multi-year
   data does).
3. Re-read the importance-reinit ablation (Δ AUC, Wilcoxon p) only after
   confirming the trigger is behaving — the numbers you got before are
   confounded by this bug and should not be trusted or written up.

---

# ADDENDUM 3 — observability gap + cold-start fix (from your first post-fix FAST_MODE run)

Your first FAST_MODE run after the emission-formula fix produced:
`break_folds=[1,4]  trigger fire-rate/fold=1.00 0.00 0.00 0.00 0.00 0.00`

The always-on pathology was gone (good — that part of the fix held). But the
one fold that fired (fold 1, array position 0) was NOT a real break — the
real breaks sit at array positions 1 and 4. This is a second, distinct
problem, and the previous logging couldn't tell you whether the real breaks
were being missed or just masked. Fixed both issues:

**1. Cold-start instability at fold 1.** Confirmed directly: with only 90
rolling-vol observations and ZERO actual regime change in them, the HMM still
produced `p_trans=1.000` — a 2-state model doesn't have enough data that early
to separate states reliably, and can read pure noise as "high confidence
crisis." `get_hmm_trigger()` (in both `step7_ablation.py` and
`step9_real_data.py`) now takes a `warmup_min_obs` parameter (default 150).
Below that many observations, `threshold_obj.update()` is never called at
all — p_trans is still computed and returned for logging, but the detector
cannot act on it, and critically it can never consume cooldown or pollute the
threshold's history. Verified: 90-obs pure-noise case now returns
`is_warmup=True, triggered=False`, with `cooldown_remaining=0` and
`history` untouched.

**2. You couldn't tell "no signal" from "signal masked by cooldown."**
`AdaptiveRegimeThreshold` now exposes `self.last_raw_fire` — the pre-cooldown
decision — after every `update()` call. Both files' per-fold logging and
console printouts now show FOUR rows instead of one: `p_trans` (the raw
model output), `raw_fire` (did it cross the bar before gating, `W` if it was
a warm-up fold and wasn't evaluated), and `final trig` (what actually ran).
Read all three together — if `raw_fire` is high but `final trig` is low
outside of a `W`, that's cooldown suppression, not absence of signal, and
means the underlying detection was real.

**Why 150 is a starting point, not a proven number:** I did not derive 150
from your actual data's break dynamics — I chose it as a defensible floor
based on the instability I could reproduce (300 obs was borderline-unstable
at ~0.76 in earlier testing; 90 was clearly unreliable at 1.000). Watch the
`p_trans` row on your next FAST_MODE run for any fold flagged `is_warmup` —
if it's still swinging wildly even above 150, raise the floor. This is
something to verify against your own data, not take on faith.

## Re-run now
`FAST_MODE = True` again. Look at all four printed rows (`p_trans`,
`raw_fire`, `final trig`, and which folds are `W`), and check specifically
whether `raw_fire` (not just `final trig`) lines up with `break_folds` this
time. If `raw_fire` is still not spiking at the real breaks, that's a new,
different problem from anything fixed so far and needs its own diagnosis —
tell me exactly what the four rows show, don't just say "still not working."

---

# ADDENDUM 4 — self-referential threshold bug (found from the actual checkpoint)

You ran the diagnostic-upgraded file (has `fold_raw_fire`/`fold_is_warmup`,
proof: those keys are in the checkpoint) but not yet the version-bumped one
(`_v2.pkl`, not `_v3.pkl` — grab the latest file before your next run; the
`_v2` one is now orphaned, delete it, it won't be reloaded by the new
filename anyway but there's no reason to keep it).

The run itself surfaced a THIRD bug, found by loading your actual checkpoint
and replaying the exact recorded numbers, not by reading the terminal:

```
fold_p_trans  : [0.995, 1.000, 1.000, 0.0028, 0.0010, 0.99998]
fold_raw_fire : [True,  True,  True,  False,  False,  False]
```

Fold 6's raw p_trans is 0.99998 — essentially certain — yet `raw_fire` was
`False`. Reason, confirmed by direct computation on your exact numbers:
`AdaptiveRegimeThreshold`'s percentile branch included the CURRENT observation
in the very same window it computed the threshold from. With several earlier
folds also near 1.0 already in history, the 85th percentile of a window that
includes today's own reading works out to ~0.9999823 — and today's reading
(0.9999764) missed it by 0.0000059. A real detection was thrown away by a
margin smaller than floating-point noise, because the reading was being
compared partly against itself.

**Fix:** compute the percentile threshold from PRIOR history only (before
appending the current point), exactly the way the CUSUM branch already did
it (`mu_ref = mean(window[:-1])`) — the percentile branch just hadn't
mirrored that. Verified on your exact recorded sequence: fold 6 now correctly
returns `raw_fire=True`. This most likely corresponds to your actual second
documented break — walk-forward detection of an event can only ever happen
starting the fold AFTER it enters the training window, so firing one fold
after `break_folds[1]` rather than exactly on it is expected, honest
behavior, not a miss.

## What is still NOT resolved — and I'm not going to pretend I fixed it

Fold 1 (position 0) still fires and its cooldown still suppresses fold 2
(position 1 — your first documented break). I did not "fix" this, because I
cannot tell from the data alone whether it should be fixed:

- If fold 1's reading is a genuine early warning of the SAME approaching
  event dated at `break_folds[0]` — real markets do show volatility rising
  before an official crisis date — then one detection consuming the
  cooldown before the "official" date is correct behavior, not a bug.
- If fold 1's reading is unrelated noise coinciding with a real break one
  fold later, then cooldown is masking a genuinely separate detection.

I can't tell these apart without knowing the actual calendar dates involved.
**Your job, not mine:** check what date range fold 1's training window
covers, and how many trading days separate it from `break_folds[0]`'s actual
documented date. Close together (weeks) → treat current behavior as correct.
Far apart (months) → the cooldown is masking a real separate event, and the
fix is to tighten what counts as "early" (e.g. require the raw signal to
sustain across 2 consecutive folds before triggering, rather than firing on
one). Don't ask me to just pick a number here — this is a judgment call
about your specific data that I don't have enough information to make for
you honestly.

## Re-run order
1. Confirm you're on this file (checkpoint should now save as `_v3.pkl`).
2. `FAST_MODE = True`, re-run, check the four-row printout again.
3. Look up the actual dates for fold 1's window vs `break_folds[0]`'s date
   before deciding anything about the remaining cooldown question above.
4. Only once both real breaks show `raw_fire=True` at or immediately after
   their documented fold, move to `FAST_MODE = False`.