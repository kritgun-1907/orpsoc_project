# Tier A Findings

Work-order §3 Tier A (A1–A6) — **analysis only, no re-runs**. Source:
`results/step7_ablation_v2.json` (30 seeds, 8 folds, θ=0.5, provenance
`4e388f80e5ac`) plus the step9 checkpoints for A6. Reproduce with
`python experiments/tier_a.py`; machine-readable output in `results/tier_a.json`.

Companion documents: [`GENERAL_FINDINGS.md`](GENERAL_FINDINGS.md),
[`TIERC_FINDINGS.md`](TIERC_FINDINGS.md).

---

## Pre-registered definitions

Fixed **before** any result was inspected, because choosing a metric after
seeing the numbers is how a post-hoc story becomes an apparent finding.

- **Recovery (A1.a)**, two metrics, both reported:
  (i) `folds_to_recovery` — post-switch folds before per-fold AUC first returns
  within **ε = 0.02** of the pre-switch mean; right-censored at `n_post` with the
  censored count reported. (ii) `post_mean` — mean per-fold AUC over post-switch
  folds; no threshold, so it cannot be gamed by ε.
- **Fold grouping (A1.b)** — from `fold_phase`, **not** `fold_is_pre`. The
  latter pools the straddling fold into POST, which is wrong; a straddle fold is
  neither. The straddle fold is reported separately and excluded from both groups.
- **Break-adjacent (A6.a)** — `break_folds ∪ (break_folds + 1)`. The `+1` is
  causal: a walk-forward learner cannot react to a break until it has entered the
  **training** window, one fold after the fold whose *test* window contains it.
- **Statistics** — paired per-seed Wilcoxon, two-tailed, with Benjamini-Hochberg
  q alongside every raw p.

Fold phases for `v2_regime_switch`: `[pre, pre, pre, straddle, post, post, post, post]`.

---

## A1 — Pre/post-switch recovery

| condition | pre | straddle | post | drop | folds→recovery | censored |
|---|---|---|---|---|---|---|
| Baseline | 0.8699 | 0.5393 | **0.6840** | 0.1859 | 4.00 | 30/30 |
| OrPSOC | 0.8654 | 0.5498 | 0.6122 | 0.2532 | 3.90 | 27/30 |
| +APSOLL | 0.8187 | 0.5445 | 0.5903 | 0.2284 | 3.90 | 23/30 |
| Full Hybrid | 0.8454 | 0.5406 | 0.5682 | 0.2772 | 4.00 | 30/30 |
| FH no-imp | 0.8320 | 0.5375 | 0.5685 | 0.2635 | 4.00 | 28/30 |

Paired Wilcoxon vs Baseline:

| group | condition | Δ vs base | p | BH q |
|---|---|---|---|---|
| pre | OrPSOC | −0.0044 | 0.7151 | 0.715 |
| pre | Full Hybrid | −0.0245 | 0.0006 | 0.001 |
| **post** | OrPSOC | **−0.0717** | 0.0000 | 0.000 |
| **post** | +APSOLL | **−0.0937** | 0.0000 | 0.000 |
| **post** | Full Hybrid | **−0.1158** | 0.0000 | 0.000 |
| **post** | FH no-imp | **−0.1155** | 0.0000 | 0.000 |

### A1.d — claim at supported strength

Baseline's post-switch mean (0.6840) **exceeds** Full Hybrid's (0.5682), and
every variant is significantly worse post-switch. **"Faster recovery than
baseline" is not supported** — exactly the outcome A1.d anticipated.

Note also that **no condition recovers** to within ε of its pre-switch mean
inside the 4 post folds; baseline is censored 30/30. The defensible claim is
about adaptively-correct subsets at a compactness trade-off (A4), not recovery
speed.

---

## A2 — Signal recall across the switch

`r2` = post-switch signals (`signal_3/4`, should rise); `r1` = pre-switch
signals (`signal_0/1/2`, mirror image, should fall).

| condition | r2 pre | r2 post | r2 rise | q | r1 fall | q |
|---|---|---|---|---|---|---|
| OrPSOC | 0.444 | 1.058 | **+0.614** | 0.000 | +1.831 | 0.000 |
| +APSOLL | 0.189 | 0.700 | +0.511 | 0.000 | +1.775 | 0.000 |
| Full Hybrid | 0.378 | 0.575 | +0.197 | 0.002 | +1.789 | 0.000 |
| FH no-imp | 0.378 | 0.542 | +0.164 | 0.013 | +1.744 | 0.000 |

### A2.b — the manuscript's specific claim is refuted

§3.6 states recall of `s3–s4` rises from the switch fold onward **for Full
Hybrid but not for standard OrPSOC**. It rises for *both*, and **more than three
times as much for OrPSOC** (+0.614 vs +0.197). The mirror image holds cleanly —
`r1` falls ≈1.8 for every variant — so adaptation is real; Full Hybrid simply
is not distinctive at it. **This sentence must be rewritten.**

---

## A3 — Jaccard dip-and-recovery, tested

`per_fold_jaccard[i]` compares fold `i` with `i+1` (verified in
`orpsoc_utils.feature_stability_ratio`). The switch pair is index **3**
(straddle → first post): the earliest pair across which the subset can
*causally* change.

| condition | J@switch | J@adjacent | dip | p | BH q |
|---|---|---|---|---|---|
| OrPSOC | 0.213 | 0.170 | −0.043 | 0.0106 | 0.053 |
| +APSOLL | 0.143 | 0.145 | +0.002 | 0.6408 | 1.000 |
| **Full Hybrid** | 0.480 | 0.485 | **+0.005** | 0.8448 | 1.000 |
| FH no-imp | 0.449 | 0.423 | −0.026 | 0.7112 | 1.000 |

**No significant dip for Full Hybrid at the causally correct pair.**

Full trace (mean across seeds):

```
pair          0-1    1-2    2-3    3-4    4-5    5-6    6-7
Baseline     1.000  1.000  1.000  1.000  1.000  1.000  1.000
OrPSOC       0.229  0.232  0.245  0.213  0.095  0.132  0.175
+APSOLL      0.187  0.225  0.229  0.143  0.060  0.073  0.111
Full Hybrid  0.295  0.207  0.920  0.480  0.050  0.712  0.220
FH no-imp    0.295  0.271  0.786  0.449  0.060  0.791  0.292
                                  ^SWITCH
```

### Exploratory — flagged, not tested

Full Hybrid's Jaccard **minimum is at pair 4-5 (0.050), not the pre-registered
pair 3-4 (0.480)**. The dip-and-recovery shape exists (0.480 → 0.050 → 0.712)
but arrives one pair later than causal reasoning predicts. Moving the test index
to match the observed minimum would be post-hoc fold selection — precisely what
A6.a warns against. **Treat as a hypothesis to pre-register and test on a fresh
benchmark draw.**

> **Correction on record:** the first version of this analysis used the
> pre→straddle pair (index 2), which is *before* adaptation is causally possible.
> It reported a spurious "spike" (J=0.920). Corrected to index 3.

---

## A4 — Compactness and runtime: the defensible claim

| Level | Baseline AUC | FH AUC | FH mean k | % of N | % of baseline AUC |
|---|---|---|---|---|---|
| v2_null | 0.5239 | 0.4992 | 7.3 | 15% | 95.3% |
| v2_white_noise | 0.9122 | 0.8879 | 7.1 | 14% | **97.3%** |
| v2_ar1 | 0.9103 | 0.8477 | 9.4 | 19% | 93.1% |
| v2_drift | 0.8225 | 0.7884 | 8.3 | 17% | 95.8% |
| v2_regime_switch | 0.7356 | 0.6687 | 9.4 | 19% | 90.9% |

**Full Hybrid attains 90.9–97.3% of baseline AUC using 14–19% of the features.**
This is the strongest supportable O4 statement in the synthetic study.

Runtime: ~42 s per fold for the search conditions vs **0.22 s** for baseline —
roughly a 190× cost for the compactness gain.

### Subset size is systematically smaller for Full Hybrid

| level | OrPSOC k | Full Hybrid k |
|---|---|---|
| v2_null | 10.9 | 7.3 |
| v2_white_noise | 11.4 | 7.1 |
| v2_ar1 | 11.4 | 9.4 |
| v2_drift | 10.6 | 8.3 |
| v2_regime_switch | 11.1 | 9.4 |

Smaller in **every** level, never once the reverse, at identical θ. Attributed
to the warm start compounding compression: each fold inherits the previous
fold's gbest and the size penalty only ever pushes downward, with no mechanism
to re-add a dropped feature. OrPSOC re-searches cold each fold and re-discovers
the fuller subset.

---

## A6 — Break-adjacent folds on real data

### sector_etf — the one place selection beats baseline

`break_folds=[1,5]`, break-adjacent `[1,2,5,6]`, 20 seeds:

| condition | AUC@adjacent | vs base | BH q | mean k |
|---|---|---|---|---|
| Baseline | 0.8203 | — | — | 58.0 |
| **OrPSOC** | **0.8287** | **+0.0084** | **0.024** | **12.6** |
| +APSOLL | 0.7982 | −0.0221 | 0.000 | 5.9 |
| Full Hybrid | 0.7863 | −0.0340 | 0.000 | 7.1 |
| FH no-imp | 0.7856 | −0.0347 | 0.000 | 6.4 |

**Standard OrPSOC beats baseline on break-adjacent folds using 12.6 features
against 58**, surviving BH correction. Full Hybrid does not.

### fama_french — nothing wins

| condition | AUC@adjacent | vs base | BH q | mean k |
|---|---|---|---|---|
| Baseline | 0.8732 | — | — | 50.0 |
| OrPSOC | 0.8162 | −0.0570 | 0.000 | 9.6 |
| Full Hybrid | 0.7965 | −0.0767 | 0.000 | 9.6 |

A6.c's hypothesis therefore **half-survives**: selection does visibly beat the
baseline exactly where the method is theorised to help, but it is the *plain
wrapper*, not the adaptive machinery, and on one dataset only.

### A6.e — why a break appears one fold "late"

Walk-forward causality. A learner cannot react to a break until it has entered
the **training** window, which is one fold after the fold whose *test* window
first contains it. A detector fire at `break_fold + 1` is the earliest causally
possible response, not a lag requiring explanation.

---

## Cross-cutting mechanism: warm-start memory

`warm_start_pos` is passed **only** to `full_hybrid` and `full_hybrid_noimp`
(`step7_ablation.py:604,646`). OrPSOC and +APSOLL start every fold cold. Three
Tier A puzzles share this single cause.

```
mean Jaccard   WARM conditions = 0.416    COLD = 0.168    ratio 2.48x
```

**Per-fold r2 recall — Full Hybrid lags OrPSOC by one fold:**

```
                    f0    f1    f2    f3    f4    f5    f6    f7
OrPSOC (cold)     0.33  0.60  0.40  0.23  0.33  0.37  1.63  1.90
FullHybrid (WARM) 0.53  0.43  0.17  0.23  0.17  0.07  0.37  1.70
                                    ^break
```

OrPSOC picks up the new signals at fold 6, Full Hybrid at fold 7.

**And the detector is not at fault:**

```
Full Hybrid detector:   f0    f1    f2    f3    f4    f5    f6    f7
phase                  pre   pre   pre  strad  post  post  post  post
TRIGGERED             0.00  0.00  1.00  0.00  1.00  0.00  0.00  0.00
```

The HMM fires at **fold 4 — the earliest causally possible fold — in 30/30
seeds**. Detection is perfect; the *response* is not. The swarm does not change
its selection until fold 7, three folds later.

### Why: the elite fraction inverts at the trigger

With `n_particles=20`, `elite_frac=0.2`:

| branch | particles seeded from old-regime gbest |
|---|---|
| `hmm_trigger=False` | `particles[0]` only → **1 of 20** |
| `hmm_trigger=True` | `round(0.2×20)` → **4 of 20** |

**Detecting a regime change quadruples the old-regime material in the swarm**,
each elite installed with `best_fit` evaluated so it acts as a strong attractor.
The 4 elites pull against the 16 importance-reinitialised particles.

> This is **not** a contradiction of the supervisor's guidance. Both suggestions
> — importance-guided reinit (#2) and elite preservation / partial restart (#3)
> — are already implemented (`orpsoc_utils.py:1136-1180`, the former explicitly
> labelled *"Professor suggestion #2"*). What is unspecified is *how many*
> elites. `experiments/elite_frac_sweep.py` sweeps `elite_frac ∈ {0.2, 0.1,
> 0.05, 0.0}` with importance-reinit held on; `0.0` is the endpoint of the
> sweep, not a rejection of population memory.

### RESULT: the elites are exonerated

`experiments/elite_frac_sweep.py`, 240/240 units, 30 seeds
(`results/elite_frac_sweep.json`). On `v2_regime_switch`:

| elite_frac | condition | AUC | Δ vs 0.2 | p | k | **adapt fold** |
|---|---|---|---|---|---|---|
| 0.2 | full_hybrid | 0.6687 | (control) | — | 9.4 | **7** |
| 0.1 | full_hybrid | 0.6703 | +0.0016 | 0.657 | 9.4 | **7** |
| 0.05 | full_hybrid | 0.6677 | −0.0010 | 1.000 | 9.2 | **7** |
| 0.1 | FH no-imp | 0.6708 | +0.0074 | 0.100 | 9.5 | **7** |
| 0.05 | FH no-imp | 0.6701 | +0.0067 | 0.428 | 9.6 | **7** |

The `elite_frac=0.2` control reproduces the paper run exactly (AUC 0.6687,
k=9.4, adaptation at fold 7), so the harness is valid.

**`adaptation_fold = 7` at every setting.** Cutting old-regime elites from 4
particles to 1 does not pull adaptation earlier, and AUC is unchanged (all
p > 0.10). **The elite fraction is not the cause of the three-fold lag between
correct detection (fold 4) and behavioural change (fold 7).**

### RESOLVED: it is the warm start on NON-trigger folds

`experiments/lag_factorial.py`, 2×2 factorial, 120/120 units, 30 seeds
(`results/lag_factorial.json`):

| cell | AUC | Δ vs ctrl | p | k | **adapt fold** |
|---|---|---|---|---|---|
| warm=on burst=scaled **[CONTROL]** | 0.6687 | (control) | — | 9.4 | 7 |
| warm=on burst=**FULL** | 0.6758 | +0.0071 | 0.7000 | 6.2 | 7 |
| **warm=off** burst=scaled | **0.6887** | **+0.0199** | **0.0004** | 10.9 | **6** |
| warm=off burst=FULL | 0.6772 | +0.0085 | 0.2534 | 6.6 | 6 |

The control reproduces fold 7 and AUC 0.6687 exactly, so the harness is valid.

- **The Phase 2 burst is cleared.** `+APSOLL` passes `p_trans=None` and gets a
  full burst; Full Hybrid passes the HMM value, which at fold 4 is **0.6902**, so
  it receives only 69% of the burst at the critical fold. Removing that throttle
  (`burst=FULL`) changes **nothing** — still fold 7, p=0.70.
- **The velocity update is cleared without a run.** `+APSOLL` and Full Hybrid
  both call `run_hybrid_orpsoc`, so they share the velocity update and phase
  schedule exactly, yet adapt at folds 6 and 7 respectively. A term common to
  both cannot explain a difference between them.
- **The warm start is the cause.** Removing it alone pulls adaptation to fold 6
  and is the only cell with a significant AUC gain (+0.0199, p=0.0004).

Per-fold recall of the post-switch signals:

```
                        f4     f5     f6     f7
CONTROL               0.17   0.07   0.37   1.70
warm=off burst=scaled 0.23   0.13   1.33   1.80   <- a full fold earlier
```

**Why the elite_frac sweep found nothing:** `elite_frac` governs only the
`hmm_trigger=True` branch, and the trigger fires once (fold 4). The damage is
done by the other branch —

```python
if not hmm_trigger:
    particles[0]["pos"] = ws     # stale gbest, EVERY non-trigger fold
```

— which re-seeds the pre-switch solution on folds 5-7, exactly where the sweep
was not looking.

Two secondary results:

- **The interventions interact negatively.** `warm=off burst=FULL` (0.6772) is
  *worse* than `warm=off burst=scaled` (0.6887): once the anchor is gone, the
  extra burst scatters a swarm that was already searching correctly.
- **Subset size confirms the compression mechanism.** Control picks k=9.4,
  `warm=off` picks **k=10.9** — larger, consistent with warm-start memory
  ratcheting subsets downward with no mechanism to re-add a dropped feature.

**Scale caveat:** 0.6887 remains below baseline's 0.7356. This is a real,
significant improvement *to the method*, not a reversal of the headline result.

`v2_drift` is the control level and returns `adaptation_fold=None` at every
setting, as expected: with no discrete break, post-switch signals never cross
the recall threshold.

> **Caveat on the 0.0 arm.** During this sweep `orpsoc_utils` floored the elite
> count at `max(1, ...)`, so `elite_frac=0.0` still produced ONE elite and was
> byte-identical to `0.05` (visible in the drift rows: 0.7894 / 0.7866 in both).
> The true "no population memory" endpoint was therefore never tested. Fixed
> afterwards via `_elite_count()`, which returns 0 for `elite_frac <= 0` while
> still rounding any positive fraction up to at least one particle. Given that
> 4 → 1 elites changed nothing, 1 → 0 is unlikely to, but it remains untested.

---

## Summary of manuscript impact

| Claim | Status |
|---|---|
| "Faster recovery than baseline" (§3.5) | **Not supported** — remove |
| "r2 recall rises for Full Hybrid but not OrPSOC" (§3.6) | **Refuted** — rises 3× more for OrPSOC |
| Jaccard dip-and-recovery (§3.6) | **Not significant** at the causally correct pair |
| Compactness trade-off (§3.7) | **Supported** — 90.9–97.3% of baseline AUC at 14–19% of features |
| Sector-ETF fold-2 observation (§3.3) | **Half-supported** — OrPSOC beats baseline (q=0.024), Full Hybrid does not |
