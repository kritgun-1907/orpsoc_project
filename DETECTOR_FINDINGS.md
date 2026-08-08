# Detector Findings

The regime-change detector, investigated end to end. This is the newest strand of
work and the one with the strongest negative result, because it is the only claim
here established across **20 independent benchmark draws** rather than one.

Companion documents: [`GENERAL_FINDINGS.md`](GENERAL_FINDINGS.md),
[`TIERA_FINDINGS.md`](TIERA_FINDINGS.md), [`TIERC_FINDINGS.md`](TIERC_FINDINGS.md).

Scripts: `experiments/detector_statistics.py`, `detector_threshold_sweep.py`,
`detector_tail.py`, `detector_alternatives.py`, `detector_multidraw.py`.
Data: `results/detector_multidraw.json`.

---

## 1. The headline result

> Across **20 independent benchmark draws** of `v2_regime_switch`, the regime
> detector as implemented fires **13.2 ± 1.4 times before the break** and
> **11.3 ± 1.5 times on pure noise features** per draw, while detecting the
> genuine break at the earliest causally possible fold on only **1.40 of 5**
> signal features. It achieved a false-positive-free run on **0 of 20 draws**.

Full comparison, one gate (`percentile_k=85`, `cooldown=1`) for every statistic
so differences are attributable to the statistic, not the gating:

| statistic | hit@4 /5 | false positives | NULL fires | frac FP=0 |
|---|---|---|---|---|
| `hmm_level` **(current)** | 1.40 ± 0.97 [0,3] | **13.2 ± 1.4** [11,16] | 11.3 ± 1.5 [8,14] | **0.00** |
| `hmm_occupancy` | 0.00 ± 0.00 [0,0] | **1.3 ± 1.1** [0,4] | 2.3 ± 1.4 [0,5] | 0.25 |
| `ks_twin` | 0.40 ± 0.49 [0,1] | 5.0 ± 1.7 [1,8] | 5.7 ± 1.1 [4,8] | 0.00 |
| `bocpd` | 3.95 ± 0.80 [3,5] | 19.9 ± 0.2 [19,20] | 17.6 ± 0.9 [16,19] | 0.00 |

*hit@4* = fires at the earliest **causally possible** fold (a walk-forward learner
cannot react until the break enters its TRAINING window). *FP* = any fire before
that fold. *NULL* = any fire on a `noise_*` feature, which has no relationship to
`y` and no regime structure at all.

---

## 2. How the detector works, and why it fails

```python
obs     = rolling_std(X_train[feat_names[0]], window=20)   # ONE feature
hmm.fit(obs)                                                # 2-state Gaussian HMM
p_trans = gamma[-1, 1]                                      # <-- the defect
triggered = AdaptiveRegimeThreshold(percentile_k=85).update(p_trans)
```

`gamma[-1, 1]` is the posterior probability that **the last observation of the
training window sits in the high-volatility state**. That is a **level**
statistic, not a **change** statistic: it asks *"is it turbulent right now?"*,
never *"did something change?"*. Any volatility cluster landing at the end of a
window fires it, and autocorrelated series produce such clusters by construction.

Three consequences, all measured:

1. **It fires on pure noise.** `noise_0/1/2/3/4` trigger as readily as the real
   signals, at pre-break folds. Whatever it is detecting is not regime change.
2. **False alarms are more confident than real breaks.** On the seed-42 draw the
   spurious fold-2 fire scored `p_trans = 0.9356`; the genuine break at fold 4
   scored `0.6902`. No threshold separates those in the right direction.
3. **The posteriors are saturated.** Roughly three quarters of all `gamma` values
   are pinned at 0.000 or 1.000 — the condition `AdaptiveRegimeThreshold` warns
   about itself: *"the underlying P(Transition) signal is saturated ... Check HMM
   state separation."* Every statistic downstream is a summary of those
   posteriors and cannot recover what they destroyed.

A further oddity: `feat_names[0]` on this benchmark is `signal_0`, a **pre-switch**
signal. After the break it stops predicting `y` entirely, yet the detector keeps
watching it. The monitored variable is the one guaranteed to become irrelevant.

---

## 3. What was tried, and what each attempt ruled out

| attempt | result | what it eliminated |
|---|---|---|
| Swap the observable (8 features incl. noise) | noise fires as readily as signal | **not** the input feature |
| `deviation` = `gamma[-1,1] − mean(gamma[:,1])` | FP 15→8, still fires at the critical fold 2, never hits fold 4 | baseline subtraction rescales a single-point estimate; it does not denoise it |
| `occupancy` = recent-half minus earlier-half mean | FP → 0 on seed 42, but **never** detects at fold 4 | half-window averaging dilutes a break that has only just entered |
| Threshold sweep, 16 settings `k∈[50,85] × cooldown∈{0,1}` | `hit@4 = 0/5` at **every** setting | not a gating problem — the statistic has no fold-4 response |
| Recent-tail, `T ∈ {10..60} × k` | no setting reaches FP=0 | the responsiveness/robustness trade-off is in the estimator |
| Twin-window KS | FP 13.2 → 5.0, hit@4 0.40 | distribution-free helps, but twin windows dilute the same way |
| BOCPD (hazard 1/250, `P(r≤50)`) | `FP = 19.9 ± 0.2` of a maximum 20 | fires on everything — see caveat below |

**The axis that explains all of it:** how many observations the estimate rests
on. One point (`level`, `deviation`) → responsive but fires on any cluster. Half
a window (`occupancy`) → clean but structurally one fold late. No point on that
axis is both.

---

## 4. Corrections on record

Three claims made during this investigation did not survive scrutiny, and are
retracted here rather than quietly dropped.

- **"`occupancy` never false-fires."** True on the seed-42 draw, and the entire
  basis for preferring it. On a held-out draw (seed 1234) it fired 3 times before
  the break, and across 20 draws it is FP-free on only **25%**. The correct claim
  is that it false-fires **far less** (1.3 vs 13.2), not that it doesn't.
- **"The differences between detectors may be smaller than the variance between
  draws."** Too pessimistic. The `level` vs `occupancy` FP gap is 11.9 against a
  pooled sd of 1.7, with non-overlapping ranges ([11,16] vs [0,4]). The *ranking*
  is stable; only the absolute FP=0 claim was draw-specific.
- **"A recent-tail statistic should respond at fold 4 while staying clean."**
  Stated as a principled prediction. Tested: no `(T, k)` reaches FP=0.

**Methodological note.** Four experiments selected operating points from the same
seed-42 draw before anyone re-ran on fresh data. The held-out draw overturned the
headline conclusion immediately. Any operating point chosen from a single draw
here is tuning on the test — which is why §1 is reported over 20 draws and the
earlier single-draw tables are not the reportable result.

---

## 5. Caveats

- **BOCPD was tested at one parameterisation** (constant hazard 1/250, signal
  `P(run length ≤ 50)`, applied to the rolling-volatility series, standardised
  input). None of it was tuned. This is a verdict on that configuration, not on
  BOCPD as a method; a different hazard, or running on the raw series rather than
  rolling volatility, could behave very differently.
- All results are on **synthetic** `v2_regime_switch` with a single engineered
  break at row 500. Real markets have several breaks and no ground truth.
- The 20 draws vary the **data**; they do not vary `n_splits`, the rolling
  window, or the break position.

---

## 6. Why this matters beyond the detector

The same defect explains three findings that would otherwise need separate
explanations:

1. **Fama-French dormancy** (`TIERC_FINDINGS.md` §C4) — the detector fires once
   in eight folds there, so Full Hybrid runs as "+APSOLL with a warm start" for
   seven of eight folds. Those rows measure a **dormant mechanism**, not a
   failure of adaptive selection.
2. **The gated-memory churn regression** (`TIERA_FINDINGS.md`) — state-dependent
   memory recovered cold-start adaptation speed and the best AUC in the study
   (+0.0213, p=0.0001), but lost the stability benefit because a **pre-break false
   positive** forced a spurious quarantine. Its performance is bounded by detector
   precision, not by the gating logic.
3. **Full Hybrid's adaptation lag** — resolved as the warm start on non-trigger
   folds, but the detector's false alarms are what make the trigger-gated design
   fragile in the first place.

**Recommended framing:** the adaptive machinery is gated on a detector that, on
this benchmark, cannot distinguish a regime change from ordinary volatility
clustering. Any conclusion about adaptive feature selection drawn while that gate
is broken is a conclusion about the gate.
