# Tier C Findings

Work-order §3 Tier C (C1–C8) — the real-data analyses that required the full
re-run. Source: `results/step9_real_data.json` plus the **40 step9 checkpoint
units** (2 datasets × 20 seeds), which retain the per-seed payloads the results
JSON was discarding. Reproduce with `python experiments/tier_c.py`;
machine-readable output in `results/tier_c.json`, figures in
`plots/tier_c_*.png`.

Companion documents: [`GENERAL_FINDINGS.md`](GENERAL_FINDINGS.md),
[`TIERA_FINDINGS.md`](TIERA_FINDINGS.md).

**No re-run was needed.** step9's JSON stored only aggregates, but the
checkpoints always held per-seed `fold_aucs`, `fold_selected`, `jaccard` and the
trigger fields. `step9_real_data.py` now persists `seed_aucs` / `seed_fold_aucs`
so future results files stand on their own.

| item | status |
|---|---|
| C1 five-condition table + figure from one code version | **done** (2026-08-07 pipeline run) |
| C2 five-condition table per real dataset | **done** below |
| C3 importance-reinit delta + paired Wilcoxon | **done** below |
| C4 detector trigger folds vs documented dates | **done** below |
| C5 per-seed variance (std *and* range) | **done** below |
| C6 20 vs 30 seeds | **decided**: 20 real / 30 synthetic, recorded in `step9_real_data.py` |
| C7 per-fold recovery figure, fifth condition | **done**, `plots/tier_c_*.png` |
| C8 Jaccard figure, fifth condition | **done**, `plots/tier_c_*.png` |

---

## C2 / C5 — Five-condition tables with per-seed spread

### sector_etf (20 seeds)

| condition | mean AUC | std | min | max | range | vs base | p |
|---|---|---|---|---|---|---|---|
| Baseline | **0.8083** | 0.0000 | 0.8083 | 0.8083 | 0.0000 | (base) | — |
| OrPSOC | 0.8052 | 0.0102 | 0.7836 | 0.8222 | 0.0386 | −0.0031 | **0.2455** |
| +APSOLL | 0.7831 | 0.0134 | 0.7535 | 0.8073 | 0.0538 | −0.0252 | 0.0000 |
| Full Hybrid | 0.7808 | 0.0212 | 0.7437 | 0.8146 | 0.0709 | −0.0275 | 0.0000 |
| FH no-imp | 0.7749 | 0.0162 | 0.7470 | 0.8106 | 0.0635 | −0.0334 | 0.0000 |

**OrPSOC is statistically indistinguishable from baseline on sector ETFs**
(p=0.2455) while using ~13 features against 58. Every adaptive variant is
significantly *worse*.

### fama_french (20 seeds)

| condition | mean AUC | std | min | max | range | vs base | p |
|---|---|---|---|---|---|---|---|
| Baseline | **0.8390** | 0.0000 | 0.8390 | 0.8390 | 0.0000 | (base) | — |
| OrPSOC | 0.7998 | 0.0211 | 0.7515 | 0.8310 | 0.0794 | −0.0392 | 0.0000 |
| +APSOLL | 0.7693 | 0.0240 | 0.7141 | 0.8068 | 0.0927 | −0.0696 | 0.0000 |
| Full Hybrid | 0.7802 | 0.0202 | 0.7256 | 0.8113 | 0.0857 | −0.0587 | 0.0000 |
| FH no-imp | 0.7776 | 0.0241 | 0.7195 | 0.8219 | 0.1023 | −0.0613 | 0.0000 |

### C5 — why paired per-seed tests are the correct lens

Baseline `std = 0.0000` on both datasets: with no PSO it is **deterministic**.
So the entire seed-to-seed spread is a property of the **selector**, and it is
large — up to **0.1023 AUC of range** on Fama-French, **0.0709** on sector ETFs.

The best seed of `FH no-imp` on Fama-French (0.8219) beats the worst seed of
`OrPSOC` (0.7515) by 0.070, despite `OrPSOC` having the better mean. Reporting
mean-vs-mean hides a spread wider than every effect in the study. This is the
concrete argument for paired per-seed Wilcoxon over mean comparisons.

---

## C3 — Importance-reinit on real data: **not significant**

Supervisor suggestion #2 (seed non-elite particles from the classifier's feature
importances on the most recent training window):

| dataset | full_hybrid | FH no-imp | delta | p (two-tailed) | p (greater) | seeds helped |
|---|---|---|---|---|---|---|
| sector_etf | 0.7808 | 0.7749 | +0.0059 | 0.3884 | 0.1942 | **11/20** |
| fama_french | 0.7802 | 0.7776 | +0.0026 | 0.3488 | 0.1744 | **11/20** |

**This is the important nuance.** On the *synthetic* benchmark importance-reinit
is strongly significant (+0.0717 on L1, +0.0434 on L2, both p<0.0001). On *real
markets* it is not: deltas of +0.0059 and +0.0026, both p>0.34, and it helps in
**11 of 20 seeds — a coin flip** — on both datasets independently.

> Any earlier statement that "the supervisor's suggestion #2 is validated by the
> data" was based on the synthetic levels alone. It does **not** carry to real
> markets. The mechanism is sound and measurable where regimes are engineered
> and clean; it does not survive contact with real market structure.

---

## C4 — Detector vs documented breaks

Documented break dates for both datasets: **2001-09-17, 2008-09-15, 2020-02-20**.

### sector_etf — detector broadly works

```
break_folds (test window contains a break) : [1, 5]
earliest causally possible response        : [2, 6]

fold            f0      f1      f2      f3      f4      f5      f6      f7
p_trans      0.995   0.026   1.000   0.001   0.002   0.001   1.000   0.003
raw fire         1       0       1       0       0       0       1       0
TRIGGERED     1.00    0.00    1.00    0.00    0.00    0.00    1.00    0.00
```

Fires at folds **0, 2, 6**. Break-adjacent set is `[1,2,5,6]`, so **2 of 3 fires
are break-adjacent** — folds 2 and 6 are exactly the earliest causally possible
responses to the two in-sample breaks. Fold 0 is a false positive (and is a
warm-up-adjacent region where the HMM has least data).

### fama_french — **the detector is essentially inert**

```
break_folds : [2, 5]
earliest causally possible response : [3, 6]

fold            f0      f1      f2      f3      f4      f5      f6      f7
p_trans      0.001   0.000   0.044   0.001   0.001   0.001   1.000   0.394
raw fire         0       0       0       0       0       0       1       1
TRIGGERED     0.00    0.00    0.00    0.00    0.00    0.00    1.00    0.00
```

**The detector fires exactly once, at fold 6.** `p_trans` sits at 0.0003–0.044
for folds 0–5 — nowhere near threshold — including fold 3, the earliest causally
possible response to the first documented break.

### Why this matters more than the AUC tables

The Full Hybrid's entire adaptive apparatus — Phase 2 burst, elite partial
restart, importance-guided reinit — is **gated on `hmm_trigger`**. On
Fama-French that gate opens once in eight folds. For seven of eight folds Full
Hybrid is therefore running as "+APSOLL with a warm start", with none of the
machinery that distinguishes it.

**So the Fama-French result is not evidence that adaptive selection fails
there — it is evidence that the detector never gave adaptation a chance to
run.** These are very different claims and the manuscript must not conflate
them. Diagnosing *why* the HMM cannot separate states on Fama-French is a
prerequisite for any claim about adaptive selection on that dataset.

A plausible mechanism worth testing: the detector observes the rolling standard
deviation of a *single* feature (`feat_names[0]`). Fama-French factors are
already-differenced, near-orthogonal return series, so a single factor's
volatility may simply not carry the regime signal that a sector ETF price series
does.

---

## C7 / C8 — Figures with the fifth condition

`plots/tier_c_sector_etf.png` and `plots/tier_c_fama_french.png`, each a pair:

- **C7 (left)** — per-fold test AUC, all **five** conditions, ±1 std band, with
  documented break folds (red dashed) and detector fires (orange dotted)
  overlaid, so cause and effect are visible together.
- **C8 (right)** — Jaccard between consecutive folds, all **five** conditions,
  with break folds marked.

`full_hybrid_noimp` was previously confined to step8's Figure 5; it now appears
in the main recovery and stability figures, which is what C7/C8 required.

---

## Summary of manuscript impact

| Item | Finding |
|---|---|
| Real-data headline | Baseline wins on both datasets; **OrPSOC ties on sector ETFs** (p=0.2455) at 13 features vs 58 |
| Importance-reinit (supervisor #2) | Significant on synthetic, **not significant on real** (11/20 seeds, p>0.34) |
| Detector on sector ETFs | Works — 2 of 3 fires break-adjacent, at the earliest causally possible folds |
| Detector on Fama-French | **Fires once in eight folds**; adaptive machinery mostly dormant |
| Per-seed spread | Up to 0.1023 AUC range vs a deterministic baseline — paired tests are mandatory |
| Seed count | 20 real / 30 synthetic; §2.4 must be amended |

### Recommended next step

Before any further claim about adaptive selection on Fama-French, diagnose the
detector: test `get_hmm_trigger` against a multivariate volatility proxy (e.g.
the first principal component of rolling volatility across all factors) instead
of `feat_names[0]`, and re-check fire folds against `[3, 6]`. Until then, the
Fama-French rows measure a dormant mechanism.
