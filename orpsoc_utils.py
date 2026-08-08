"""
orpsoc_utils.py — Shared utilities for the full hybrid pipeline
================================================================
Import this in every step file. Contains:
  - sigmoid / build_orthogonal_positions / partial_reinit
  - windowed_feature_importance / build_importance_guided_positions
  - crossover / hamming_diversity
  - evaluate()                  with APSOLL-normalized fitness
  - AdaptiveRegimeThreshold     (percentile + CUSUM auto-threshold)
  - feature_stability_ratio     (Jaccard across folds)
  - walk_forward_folds          (shared CV harness)
  - APSOLLAdaptiveC             (adaptive c tracker)
  - run_standard_orpsoc()       (condition 2 runner)
  - run_hybrid_orpsoc()         (conditions 3 & 4 runner)

WHY A SHARED MODULE?
────────────────────
Steps 3–8 all need the same PSO primitives. One copy prevents
the "I fixed a bug in step4 but forgot step7" problem.
Import with:   from orpsoc_utils import *

FIX LOG:
  - fillna(method="bfill") → .bfill()  (pandas FutureWarning fix)
  - run_standard_orpsoc() and run_hybrid_orpsoc() moved here so
    step7_ablation.py can import them (fixes NameError crash)
  - APSOLLAdaptiveC moved here so step7 doesn't need to inline it
  - if/elif fix in run_hybrid_orpsoc phase logic (double-decrement bug)
  - PHASE-1 INIT FIX: run_hybrid_orpsoc() now always starts in Phase 1.
    hmm_trigger_delay=7 delays external HMM signal by 7 iters so swarm
    can converge before leader-influence fires (fixes Full Hybrid < +APSOLL)
  - CUSUM calibration: AdaptiveRegimeThreshold.calibrate_from_baseline()
    computes slack = 0.5*std(baseline_obs) per Page (1954) instead of
    hardcoding 0.05
  - IMPORTANCE-GUIDED REINIT (professor suggestion #2): added
    windowed_feature_importance() + build_importance_guided_positions().
    run_hybrid_orpsoc() now seeds the NON-elite particles from LightGBM
    feature importances on the most recent training window when
    hmm_trigger=True (use_importance_reinit=True by default), replacing the
    blind orthogonal restart. Elites are still preserved (partial restart).
  - ALWAYS-ON TRIGGER BUGFIX (found on real-data run: fire-rate was 1.0 on
    7-8 of 8 folds on both sector_etf and fama_french):
      (a) AdaptiveRegimeThreshold.update() used to .pop() the triggering
          observation out of its own history, so a chronically-elevated
          P(Transition) signal could never inform (raise) the percentile
          threshold -- guaranteeing every future high reading re-triggers.
          Fixed: history is never discarded; a `cooldown` counter (default
          2) now suppresses re-triggering after a confirmed detection
          instead. Also added consecutive-raw-trigger tracking with a
          console WARNING if it hits 4, so this class of bug is caught in
          a 2-minute FAST_MODE run instead of after a 16-hour FULL run.
      (b) SimpleHMM (in step7_ablation.py / step9_real_data.py, not this
          file) floored sigma at a bare "+1e-4", negligible against real
          volatility scale, letting the low-vol state's sigma collapse and
          P(Transition) saturate to ~1.0 permanently. See the fix in those
          files: sigma floor is now 10% of the series' own std.

PERFORMANCE (no change to any computed number -- see test_equivalence.py):
  - n_jobs=1 restored on every LGBMClassifier. LightGBM defaults to n_jobs=-1
    and was spawning one OpenMP thread per core for fits on a few hundred
    rows; the sync overhead dominated. Measured 4.3x FASTER at n_jobs=1, with
    bit-identical AUCs. This matters more, not less, under process-level
    parallelism, where -1 would massively oversubscribe.
  - FoldEvalContext + evaluate_ctx() (work-order 2.4.b): the imputer and
    scaler are fit ONCE PER FOLD instead of once per particle evaluation
    (~1700x per PSO run). Exactly equivalent -- both compute per-column
    statistics, and the fit still uses X_p only. ~1.16x.
  - Position cache keyed on pos.astype(uint8).tobytes() rather than a tuple of
    n_features boxed Python ints.
  - Seed-level parallelism and checkpointing live in orpsoc_runner.py and are
    driven from step7_ablation.py / step9_real_data.py. Folds are NEVER
    parallelised; the walk-forward structure is untouched.
  See ARCHITECTURE.md for the full data-flow and leakage analysis.
"""

import numpy as np
import pandas as pd
import time
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore")

# PSO_FAST_EVAL=True  → LGBMClassifier (fast config) inside evaluate()
#                        Same model family as final AUC — no proxy mismatch.
# PSO_FAST_EVAL=False → LGBMClassifier (full config) — paper-quality, slower.
PSO_FAST_EVAL = True


# ══════════════════════════════════════════════════════════════════════════════
#  PSO PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════

def sigmoid(v: np.ndarray) -> np.ndarray:
    """Maps velocity → P(feature selected). Clip prevents overflow."""
    return 1.0 / (1.0 + np.exp(-np.clip(v, -500, 500)))


def build_orthogonal_positions(n_particles: int, n_features: int,
                               seed: int = 42) -> np.ndarray:
    """
    Orthogonal initialisation: each feature selected in exactly
    half the particles. Every pair of features covers all four
    combinations (0,0)(0,1)(1,0)(1,1) roughly equally.

    This gives maximum coverage of the 2^n_features search space
    from iteration zero — far better than random initialisation.
    """
    rng  = np.random.RandomState(seed)
    grid = np.zeros((n_particles, n_features), dtype=float)
    for j in range(n_features):
        n_ones = n_particles // 2
        shift  = (j * (n_particles // max(n_features, 1))) % n_particles
        for k in range(n_ones):
            grid[(shift + k) % n_particles, j] = 1.0
    return grid[rng.permutation(n_particles)]


def partial_reinit(particles: list, n_features: int,
                   elite_frac: float = 0.2, seed: int = None) -> list:
    """
    Regime-triggered partial re-initialisation (APSOLL extension).

    Keep top elite_frac particles (elite preservation).
    Re-initialise the rest orthogonally.

    Called when HMM or adaptive threshold detects a regime change.
    """
    rng = np.random.RandomState(seed or np.random.randint(1_000_000))
    n   = len(particles)
    k   = max(1, int(n * elite_frac))

    sorted_p = sorted(particles, key=lambda p: p["best_fit"], reverse=True)
    elites   = sorted_p[:k]

    new_pos = build_orthogonal_positions(n - k, n_features,
                                         seed=rng.randint(1_000_000))
    new_particles = list(elites)
    for ep in elites:
        ep["vel"] = rng.randn(n_features) * 0.1   # gentle velocity reset

    for i in range(n - k):
        pos = new_pos[i]
        new_particles.append({
            "pos":      pos.copy(),
            "vel":      rng.randn(n_features) * 0.1,
            "best_pos": pos.copy(),
            "best_fit": -1.0,
        })
    return new_particles


def crossover(pos_a: np.ndarray, pos_b: np.ndarray,
              rate: float, min_f: int, rng) -> tuple:
    """
    Two-point crossover. Children replace parents only if better (elitist).
    Enforces minimum feature count constraint after crossover.
    """
    if rng.random() > rate:
        return pos_a.copy(), pos_b.copy()
    n = len(pos_a)
    c1, c2 = sorted(rng.choice(range(1, n), size=2, replace=False))
    ca = np.concatenate([pos_a[:c1], pos_b[c1:c2], pos_a[c2:]])
    cb = np.concatenate([pos_b[:c1], pos_a[c1:c2], pos_b[c2:]])
    for child in [ca, cb]:
        n_sel = int(child.sum())
        if n_sel < min_f:
            zeros = np.where(child == 0)[0]
            need  = min_f - n_sel
            if len(zeros) >= need:
                child[rng.choice(zeros, size=need, replace=False)] = 1
    return ca, cb


def hamming_diversity(particles: list) -> float:
    """
    Mean pairwise Hamming distance across all particles.
    0.0 = fully converged, 0.5 = maximally diverse.
    """
    pos   = np.array([p["pos"] for p in particles])
    n, d  = pos.shape
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += np.sum(pos[i] != pos[j]) / d
            count += 1
    return total / count if count > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  FITNESS FUNCTION  (APSOLL-normalised)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(pos: np.ndarray, feat_names: list,
             X_tr, y_tr, X_va, y_va,
             min_f: int,
             theta: float = 0.5) -> float:
    """
    APSOLL-normalised fitness:
        Fitness = θ × AUC  +  (1-θ) × (1 - #selected / N)

    θ=0.7 weights accuracy 70%, compactness 30%.
    Scale-invariant: always in [0, 1] regardless of n_features.
    """
    idx = np.where(pos == 1)[0]
    if len(idx) < min_f:
        return -1.0
    cols = [feat_names[i] for i in idx]
    try:
        if PSO_FAST_EVAL:
            # Fast LightGBM: same model family as final scorer, no proxy mismatch.
            # Fewer estimators/leaves keeps PSO iteration cost low.
            model = LGBMClassifier(n_estimators=40, num_leaves=15,
                                   learning_rate=0.1, verbosity=-1,
                                   random_state=42, n_jobs=1)
        else:
            model = LGBMClassifier(n_estimators=100, num_leaves=31,
                                   learning_rate=0.1, verbosity=-1,
                                   random_state=42, n_jobs=1)
        pipe = Pipeline([
            ("imp",    SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model",  model),
        ])
        pipe.fit(X_tr[cols], y_tr)
        auc = roc_auc_score(y_va, pipe.predict_proba(X_va[cols])[:, 1])
    except Exception:
        return -1.0

    compactness = 1.0 - len(idx) / len(feat_names)
    return float(theta * auc + (1.0 - theta) * compactness)


# ══════════════════════════════════════════════════════════════════════════════
#  HOISTED PER-FOLD EVALUATION CONTEXT  (work-order 2.4.b — PERFORMANCE ONLY)
# ══════════════════════════════════════════════════════════════════════════════

class FoldEvalContext:
    """
    Pre-computed design matrices for one walk-forward fold's PSO fitness split.

    WHY
    ───
    evaluate() rebuilt SimpleImputer + StandardScaler inside a fresh sklearn
    Pipeline on EVERY particle evaluation -- ~1700 times per PSO run, ~6800
    times per fold. The fitted statistics are identical every single time,
    because they only ever depend on X_p, which is fixed for the fold. This
    class computes them once per fold; evaluate_ctx() then slices columns out
    of the already-transformed matrices.

    WHY THIS IS EXACTLY EQUIVALENT (not an approximation)
    ─────────────────────────────────────────────────────
    1. SimpleImputer(strategy="mean") and StandardScaler both compute
       PER-COLUMN statistics. Column j's mean/scale does not depend on which
       other columns are present. Therefore fitting on the full column set and
       slicing afterwards == fitting on the sliced subset. Verified directly:
       sc_all.mean_[cols] == sc_sub.mean_ and sc_all.scale_[cols] ==
       sc_sub.scale_, exactly.
    2. The statistics are still fit on X_p ONLY (the internal PSO training
       split), and X_v is only ever TRANSFORMED with them -- identical to the
       Pipeline this replaces. Work-order 2.4.b requires exactly this: "fit the
       imputer and scaler on the internal training split only (X_tr.iloc[:cut]),
       then transform both halves."
    3. X_te NEVER enters this object. The held-out test fold is scored only
       after PSO finishes, by the unchanged final-model block in each runner.

    Empirically confirmed bit-identical: 9 PSO runs across 3 folds x 3
    conditions produced identical AUCs to all 10 decimals and identical
    selected feature subsets, before vs after this change.

    LEAKAGE GUARD: SimpleImputer silently DROPS all-NaN columns, which would
    change the column count between the all-columns fit and a subset fit. All
    six project datasets have zero NaNs, but we assert rather than assume.
    """

    __slots__ = ("Ap", "Av", "yp", "yv", "n_feat", "model_factory")

    def __init__(self, X_p, y_p, X_v, y_v, feat_names, model_factory=None):
        # model_factory: zero-arg callable returning an unfitted sklearn-style
        # classifier used for PSO FITNESS. None -> the project default
        # LightGBM. Swapping it is the cleanest way to ask "how much is feature
        # selection worth to a classifier that cannot select for itself?" --
        # LightGBM is an embedded selector, so an external wrapper is competing
        # with machinery the model already has.
        self.model_factory = model_factory
        cols = list(feat_names)
        imp = SimpleImputer(strategy="mean").fit(X_p[cols])
        Xp_i = imp.transform(X_p[cols])
        if Xp_i.shape[1] != len(cols):
            raise ValueError(
                f"SimpleImputer dropped {len(cols) - Xp_i.shape[1]} all-NaN "
                f"column(s) from the internal training split. Column-slicing "
                f"equivalence no longer holds -- fix the data, do not proceed.")
        sc = StandardScaler().fit(Xp_i)
        self.Ap = np.ascontiguousarray(sc.transform(Xp_i))
        self.Av = np.ascontiguousarray(sc.transform(imp.transform(X_v[cols])))
        self.yp = np.asarray(y_p)
        self.yv = np.asarray(y_v)
        self.n_feat = len(cols)


def evaluate_ctx(pos: np.ndarray, ctx: "FoldEvalContext",
                 min_f: int, theta: float = 0.5) -> float:
    """
    APSOLL-normalised fitness against a pre-transformed FoldEvalContext.

    Numerically identical to evaluate(); see FoldEvalContext for the proof.
    Kept as a separate function so evaluate() remains available unchanged for
    the legacy step files and for the equivalence harness.
    """
    idx = np.where(pos == 1)[0]
    if len(idx) < min_f:
        return -1.0
    try:
        mf = getattr(ctx, "model_factory", None)
        if mf is not None:
            model = mf()
        elif PSO_FAST_EVAL:
            model = LGBMClassifier(n_estimators=40, num_leaves=15,
                                   learning_rate=0.1, verbosity=-1,
                                   random_state=42, n_jobs=1)
        else:
            model = LGBMClassifier(n_estimators=100, num_leaves=31,
                                   learning_rate=0.1, verbosity=-1,
                                   random_state=42, n_jobs=1)
        model.fit(ctx.Ap[:, idx], ctx.yp)
        auc = roc_auc_score(ctx.yv, model.predict_proba(ctx.Av[:, idx])[:, 1])
    except Exception:
        return -1.0

    compactness = 1.0 - len(idx) / ctx.n_feat
    return float(theta * auc + (1.0 - theta) * compactness)


# ══════════════════════════════════════════════════════════════════════════════
#  IMPORTANCE-GUIDED RE-INITIALISATION  (professor suggestion #2)
# ══════════════════════════════════════════════════════════════════════════════

def windowed_feature_importance(X_tr, y_tr, feat_names,
                                window_frac: float = 0.4,
                                seed: int = 42) -> np.ndarray:
    """
    Fit a fast LightGBM on the MOST RECENT window of the training data and
    return a normalised importance vector over features (sums to 1.0).

    WHY (professor suggestion #2)
    ──────────────────────────────
    After a regime change, reinitialising the swarm orthogonally / randomly
    means the fresh particles explore *blindly* in the new regime. Instead,
    fit the classifier on the most recent training window only, read its
    feature importances, and use them to bias which features the fresh
    particles prioritise. This gives the swarm a DATA-DRIVEN starting point
    for the new regime rather than a blind restart.

    "Most recent window only" = the trailing `window_frac` of X_tr, so the
    importances reflect the *current* regime, not the averaged history.

    Robustness: falls back to a uniform vector if the model cannot be fit,
    the window has a single class, or all importances are zero. The uniform
    fallback reproduces the old orthogonal-style blind behaviour, so callers
    never crash — they just lose the guidance.
    """
    n = len(feat_names)
    uniform = np.full(n, 1.0 / n)
    try:
        cut = max(30, int(len(X_tr) * window_frac))
        Xw  = X_tr[feat_names].iloc[-cut:]
        yw  = y_tr.iloc[-cut:]
        if yw.nunique() < 2:
            return uniform
        pipe = Pipeline([
            ("imp",    SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model",  LGBMClassifier(n_estimators=60, num_leaves=15,
                                      learning_rate=0.1, verbosity=-1,
                                      random_state=seed, n_jobs=1)),
        ])
        pipe.fit(Xw, yw)
        imp = np.asarray(pipe.named_steps["model"].feature_importances_,
                         dtype=float)
        if imp.sum() <= 0 or not np.isfinite(imp).all():
            return uniform
        # Laplace-style smoothing: every feature keeps a small floor
        # probability so no feature is permanently excluded from exploration.
        imp = imp + imp.sum() * 0.05 / n
        return imp / imp.sum()
    except Exception:
        return uniform


def build_importance_guided_positions(n_particles: int, n_features: int,
                                      prob: np.ndarray,
                                      min_f: int = 3,
                                      seed: int = 42) -> np.ndarray:
    """
    Build fresh particle positions biased by an importance probability vector.

    Feature j is switched on in each particle with a probability proportional
    to prob[j] (rescaled so the expected subset size is compact but still
    exploratory), and every particle is guaranteed at least min_f features.

    This is the importance-guided replacement for build_orthogonal_positions()
    used for the NON-elite particles during a regime-triggered restart.
    """
    rng = np.random.RandomState(seed)
    prob = np.asarray(prob, dtype=float)
    if prob.sum() <= 0 or not np.isfinite(prob).all():
        prob = np.full(n_features, 1.0 / n_features)
    else:
        prob = prob / prob.sum()

    # Target expected selection size ≈ one third of the features (compact,
    # matching the compactness pressure in evaluate()), distributed by prob.
    target = max(min_f + 1, n_features // 3)
    p_sel  = np.clip(prob * target, 0.02, 0.98)

    grid = (rng.rand(n_particles, n_features) < p_sel).astype(float)
    order = np.argsort(prob)[::-1]   # highest-importance features first
    for i in range(n_particles):
        if grid[i].sum() < min_f:
            for j in order:
                if grid[i].sum() >= min_f:
                    break
                grid[i, j] = 1.0
    return grid


# ══════════════════════════════════════════════════════════════════════════════
#  APSOLL ADAPTIVE-c  (moved here from step6 so step7 can import it)
# ══════════════════════════════════════════════════════════════════════════════

class APSOLLAdaptiveC:
    """
    Tracks consecutive fitness improvements and computes adaptive c.

    APSOLL equations 5 and 6:
      m = m+1  if fitness(t) > fitness(t-1),  else m=0
      c = (m/T)^(2/3) + 1

    c is always in [1.0, 2.0]:
    - c=1.0 when m=0: no recent improvement → explore widely
    - c=2.0 when m=T: steady improvement → exploit current direction

    MEASURED DEGENERACY (work-order 2.1.b — confirmed empirically, not predicted)
    ────────────────────────────────────────────────────────────────────────────
    The consumer's trigger is `c_t < 1.05`, and
        c < 1.05  ⟺  (m/T)^(2/3) < 0.05  ⟺  m < 0.05^1.5 · T
    At T=60 that is m < 0.671, so ONLY m=0 can ever fire; you would need T ≥ 90
    for m=1 to become capable of firing. The whole (m/T)^(2/3) curve and its
    [1,2] range are therefore inert, and the trigger collapses to the boolean
    "did gbest fail to improve on this single iteration?".

    Instrumented over 12 runs at the paper config (T=60, 20 particles) on
    regime_switch:
        first-fire iteration : [6,6,6,6,6,6,7,7,7,8,8,8]   never fired: 0/12
        max c_t observed     : 1.1908   (theoretical bound 2.0 never approached)
        max m observed       : 5        (out of T=60)
        trigger TRUE on      : 45-50 of 60 iterations
    A single flat gbest step is the NORMAL state of a converging swarm, so the
    condition is true ~80% of the time; it only matters once, because the phase
    machine reads it solely inside `if phase == 1` and phases are one-way.

    `stagnant_streak` supports the patience-based replacement — see
    run_hybrid_orpsoc(apsoll_patience=..., apsoll_rearm_after=...).
    """
    def __init__(self, max_iter: int):
        self.max_iter        = max_iter
        self.m               = 0
        self.c_hist          = []
        self.prev_fit        = None
        # Consecutive NON-improving iterations. Early-stopping "patience"
        # semantics: a genuine stagnation is a run of flat steps, not one.
        self.stagnant_streak = 0

    def update(self, current_fit: float) -> float:
        if self.prev_fit is None:
            # First observation: improvement is undefined. Counting it as
            # stagnation would hand the patience counter a free head start.
            pass
        elif current_fit > self.prev_fit:
            self.m += 1
            self.stagnant_streak = 0
        else:
            self.m = 0
            self.stagnant_streak += 1
        self.prev_fit = current_fit
        c = (self.m / max(self.max_iter, 1)) ** (2.0 / 3.0) + 1.0
        c = float(np.clip(c, 1.0, 2.0))
        self.c_hist.append(c)
        return c

    def reset_stagnation(self) -> None:
        """Clear the patience counter so it must re-accumulate after a re-arm."""
        self.stagnant_streak = 0


# ══════════════════════════════════════════════════════════════════════════════
#  ADAPTIVE REGIME THRESHOLD  (percentile + CUSUM)
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveRegimeThreshold:
    """
    Automatically calibrates the regime-change detection threshold
    to the dataset's own statistical behaviour.

    METHOD A — PERCENTILE
    ─────────────────────
    threshold = percentile(recent P(Trans), k)
    Fire when P(Trans) is unusually high for this dataset.

    METHOD B — CUSUM (Cumulative Sum Control Chart)
    ───────────────────────────────────────────────
    S(t) = max(0,  S(t-1)  +  P(Trans|t)  -  μ_ref  -  slack)
    Fire when S(t) > h.
    Accumulates evidence of a sustained shift; robust to single spikes.

    Parameters
    ──────────
    method       : "percentile" (recommended) or "cusum"
    lookback     : window of past observations for threshold computation
    percentile_k : trigger level (default 75 = top quartile)
    cusum_slack  : CUSUM k parameter (allowance, default 0.1)
    cusum_h      : CUSUM control limit (default 5.0)
    """

    def __init__(self, method: str = "percentile",
                 lookback: int = 50,
                 percentile_k: float = 75.0,
                 cusum_slack: float = 0.1,
                 cusum_h: float = 5.0,
                 cooldown: int = 1):
        self.method       = method
        self.lookback     = lookback
        self.k            = percentile_k
        self.slack        = cusum_slack
        self.h            = cusum_h
        self.cooldown     = cooldown   # BUGFIX: see update() below
        self.history      = []
        self.cusum_S      = 0.0
        self.triggered    = False
        self.threshold    = 0.5
        self.trigger_log  = []   # (iteration, p_trans, threshold) tuples
        self._cooldown_remaining        = 0
        self.consecutive_raw_triggers   = 0
        self.max_consecutive_raw_triggers = 0
        self.last_raw_fire              = False

    def calibrate_from_baseline(self, baseline_observations: list) -> float:
        """
        Calibrate CUSUM slack from pre-switch (stable regime) observations.

        Page (1954) recommends slack = 0.5 × std(baseline) so the CUSUM
        only accumulates when P(Trans) consistently exceeds the baseline
        by more than half a standard deviation — ignoring isolated spikes.

        Call this BEFORE the main fold loop, passing P(Trans) values from
        the pre-switch folds (e.g. folds 1–3 on Level 4).

        Returns the calibrated slack value.
        """
        if len(baseline_observations) < 2:
            return self.slack
        self.slack = 0.5 * float(np.std(baseline_observations))
        self.slack = max(self.slack, 1e-4)   # numerical floor
        return self.slack

    def update(self, p_trans: float, iteration: int = -1) -> bool:
        """
        Feed one observation. Returns True if regime change detected.
        Call once per walk-forward fold with P(Trans|t) from the HMM.

        BUGFIX (this method used to pop() the triggering observation out of
        history right after it fired, on the theory that a confirmed spike
        would otherwise raise the percentile bar and suppress detection on
        the very folds that matter. In practice, if P(Trans) is CHRONICALLY
        elevated -- which real financial rolling-volatility data produces --
        every single fold triggers, every single trigger gets deleted before
        it can inform the threshold, and the threshold never rises to match
        reality. This is a self-reinforcing "always on" loop, confirmed on
        both real datasets (fire-rate = 1.0 on 7-8 of 8 folds). The fix:
        NEVER discard real observations. Use a cooldown counter instead --
        after a confirmed trigger, suppress RE-triggering for `cooldown`
        subsequent calls so the swarm gets a chance to actually run in the
        new regime, while the threshold keeps calibrating against the true,
        BUGFIX 2 (self-referential threshold): the percentile branch used to
        compute the threshold from a window that INCLUDED the observation
        being judged against it -- i.e. p_trans was partly compared to
        itself. Confirmed directly on a real fold: p_trans=0.9999764 failed
        to fire because the self-inclusive 85th-percentile threshold worked
        out to 0.9999823 -- a margin of 0.0000059, smaller than floating-
        point noise, deciding whether a real detection counted. Excluding
        the current point from its own comparison window (judge each new
        reading against PRIOR history only, which is what the CUSUM branch
        below already did correctly) fixed it: same data, threshold drops
        to 0.9999538, and the same reading now correctly fires.
        """
        prior_window = self.history[-self.lookback:]   # BEFORE appending
        self.history.append(p_trans)
        window = self.history[-self.lookback:]

        if self.method == "percentile":
            if len(prior_window) >= 5:
                self.threshold = float(np.percentile(prior_window, self.k))
            raw_fire = bool(p_trans > self.threshold)

        elif self.method == "cusum":
            mu_ref = float(np.mean(window[:-1])) if len(window) > 1 else 0.3
            self.cusum_S = max(0.0,
                self.cusum_S + (p_trans - mu_ref - self.slack))
            raw_fire = bool(self.cusum_S > self.h)
            if raw_fire:
                self.cusum_S = 0.0   # reset after detection

        else:
            raise ValueError(
                f"Unknown method: {self.method}. Use 'percentile' or 'cusum'.")

        # Pathology tracking: this is the diagnostic that would have caught
        # the always-on bug BEFORE a 16-hour run instead of after it.
        if raw_fire:
            self.consecutive_raw_triggers += 1
        else:
            self.consecutive_raw_triggers = 0
        self.max_consecutive_raw_triggers = max(
            self.max_consecutive_raw_triggers, self.consecutive_raw_triggers)
        if self.consecutive_raw_triggers == 4:
            print(f"  [AdaptiveRegimeThreshold] WARNING: raw detector has "
                  f"fired 4 folds in a row (p_trans={p_trans:.3f}, "
                  f"threshold={self.threshold:.3f}). A real regime rarely "
                  f"stays 'transitioning' this long -- this usually means "
                  f"the underlying P(Transition) signal is saturated "
                  f"(near 0/1 with no in-between), not that the regime is "
                  f"genuinely persisting. Check HMM state separation.",
                  flush=True)

        # BUGFIX: expose the pre-cooldown decision. Without this, "triggered"
        # alone conflates two very different things -- "the signal says no"
        # and "the signal says yes but cooldown is suppressing it" -- and you
        # cannot tell which one you're looking at from the outside. Confirmed
        # this ambiguity in practice: fold 1 of a real run fired (cold-start
        # noise), consumed the cooldown, and fold 2 -- a DOCUMENTED real
        # break -- showed "no trigger" with no way to tell if that was a
        # genuine miss or a masked one. Read `threshold_obj.last_raw_fire`
        # right after calling update() to see the true underlying signal.
        self.last_raw_fire = raw_fire

        # Cooldown: a CONFIRMED trigger suppresses re-triggering for
        # `cooldown` further calls, without touching history/threshold.
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            self.triggered = False
        else:
            self.triggered = raw_fire
            if self.triggered:
                self._cooldown_remaining = self.cooldown

        if self.triggered:
            self.trigger_log.append((iteration, p_trans, self.threshold))

        return self.triggered

    def summary(self) -> dict:
        return {
            "method":          self.method,
            "n_triggers":      len(self.trigger_log),
            "triggers":        self.trigger_log,
            "final_threshold": self.threshold,
            "max_consecutive_raw_triggers": self.max_consecutive_raw_triggers,
            "cusum_slack":     self.slack,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE STABILITY METRIC  (Jaccard across folds)
# ══════════════════════════════════════════════════════════════════════════════

def _elite_count(elite_frac: float, n_particles: int) -> int:
    """
    Number of particles seeded from the carried gbest on a regime trigger.

    The floor is deliberately conditional. This used to be a flat
    `max(1, round(elite_frac * n_particles))`, which meant elite_frac=0 still
    produced ONE elite -- so "keep no population memory" was untestable, and an
    elite_frac sweep silently returned identical results for 0.0 and 0.05
    (observed: v2_drift AUC 0.7894/0.7866 byte-identical across both arms).

    elite_frac == 0 now means exactly zero elites, which is the endpoint the
    sweep needs. Any elite_frac > 0 still rounds UP to at least one particle, so
    a small-but-nonzero fraction on a small swarm cannot silently become "no
    memory at all" -- that would be a different and equally confusing bug.
    """
    if elite_frac <= 0:
        return 0
    return max(1, int(round(elite_frac * n_particles)))


def feature_stability_ratio(selected_sets: list, switch_pair: int = None) -> dict:
    """
    Jaccard similarity between consecutive walk-forward folds.

    1.0 = identical feature sets (perfectly stable)
    0.0 = completely different feature sets (maximally unstable)

    On Level 4 the expected pattern is:
      pre-switch : high Jaccard (same signal features each fold)
      around switch: sharp drop (swarm re-exploring)
      post-switch : high Jaccard again (new signal features stabilised)

    Parameters
    ──────────
    selected_sets : list of lists, one per fold e.g.
                    [["signal_0","signal_1",...], ["signal_0",...], ...]
    """
    n = len(selected_sets)
    if n < 2:
        return {"per_fold_jaccard": [], "pre_regime_stability": 1.0,
                "post_regime_stability": 1.0, "regime_adaptation_drop": 0.0}

    jaccard = []
    for i in range(n - 1):
        A = set(selected_sets[i])
        B = set(selected_sets[i + 1])
        union = A | B
        j = len(A & B) / len(union) if union else 1.0
        jaccard.append(j)

    # WHERE TO SPLIT pre / post.
    #
    # This used to be `mid = len(jaccard) // 2` -- the same "assume the break is
    # at the midpoint" defect that classify_folds() was written to fix for
    # FOLDS, left in place for the JACCARD split. It was correct only by
    # coincidence: with n_splits=8 and a break at row 500 of 1000, the midpoint
    # of the 7 fold-pairs lands on the true switch pair. Change n_splits, or use
    # a benchmark whose break is not at 50%, and every pre/post/drop number
    # silently describes the wrong folds.
    #
    # switch_pair is the index of the fold-pair spanning the break. Pair i
    # compares fold i with fold i+1, so if the break first enters the TEST
    # window of fold f (the straddle fold), the earliest pair across which the
    # subset can causally change is pair f -- a walk-forward learner cannot
    # react until the break is in its TRAINING window, one fold later.
    #
    # Passing None keeps the legacy midpoint AND flags it in the output, so a
    # stationary level (no break) does not silently report a meaningless
    # "regime adaptation drop" as if it were real.
    if switch_pair is None:
        mid = len(jaccard) // 2
        split_basis = "midpoint (NO break supplied -- not regime-aware)"
    else:
        mid = int(np.clip(switch_pair, 0, len(jaccard)))
        split_basis = f"switch_pair={switch_pair}"

    pre  = float(np.mean(jaccard[:mid]))  if mid > 0            else 1.0
    post = float(np.mean(jaccard[mid:]))  if mid < len(jaccard) else 1.0

    return {
        "per_fold_jaccard":       jaccard,
        "pre_regime_stability":   pre,
        "post_regime_stability":  post,
        "regime_adaptation_drop": pre - post,
        "split_index":            mid,
        "split_basis":            split_basis,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  WALK-FORWARD HARNESS  (shared, used by Steps 2–8)
# ══════════════════════════════════════════════════════════════════════════════

def walk_forward_folds(X: pd.DataFrame, y: pd.Series,
                       n_splits: int = 8, gap: int = 5,
                       min_train: int = 150) -> list:
    """
    Returns a list of (X_train, y_train, X_test, y_test, train_end) tuples.
    The gap between train_end and test_start removes rolling-window leakage.
    """
    n         = len(X)
    fold_size = (n - min_train - gap) // n_splits
    folds     = []

    for fold in range(n_splits):
        train_end  = min_train + fold * fold_size
        test_start = train_end + gap
        test_end   = test_start + fold_size
        if test_end > n:
            break
        folds.append((
            X.iloc[:train_end],
            y.iloc[:train_end],
            X.iloc[test_start:test_end],
            y.iloc[test_start:test_end],
            train_end,
        ))
    return folds


# ══════════════════════════════════════════════════════════════════════════════
#  FOLD PARTITIONING RELATIVE TO A KNOWN STRUCTURAL BREAK
# ══════════════════════════════════════════════════════════════════════════════

def classify_folds(folds, switch_index):
    """
    Label each walk-forward fold PRE / STRADDLE / POST relative to a known
    structural break at row `switch_index`, and report whether the fold's
    TRAINING window has seen any post-switch data yet.

    WHY THIS REPLACES `fold_idx < N_SPLITS // 2`
    ─────────────────────────────────────────────
    The old rule assumed the switch lands exactly at the midpoint fold. It does
    not. With switch_index=500, min_train=150, gap=5:

        n_splits=8 -> fold 4 test window [470, 574] CONTAINS t=500
        n_splits=6 -> fold 3 test window [435, 574] CONTAINS t=500

    In both cases the straddling fold was labelled PRE and pooled into the
    pre-switch aggregate, even though a third of its test rows come from the new
    regime. A straddle fold is neither pre nor post and must be reported
    separately or excluded -- pooling it contaminates BOTH group means.

    `train_sees_post` is the second, independent point. A selector cannot
    possibly adapt to a regime it has never been trained on. With n_splits=8,
    fold 4's training window is [0, 464] -- entirely pre-switch -- so the
    earliest fold at which ANY method could adapt is fold 5. Any "recovery
    speed" analysis that counts fold 4 as a failure to adapt is measuring an
    impossibility. This is the same walk-forward causality argument already made
    for the real data (manuscript 2.5); it was never applied to the synthetic side.

    Parameters
    ──────────
    folds        : output of walk_forward_folds()
    switch_index : integer row index of the structural break, or None for
                   stationary levels (everything is then labelled "pre")

    Returns a list of dicts, one per fold:
        {"phase": "pre"|"straddle"|"post", "train_sees_post": bool,
         "test_start": int, "test_end": int, "train_end": int,
         "frac_post_in_test": float}
    """
    out = []
    for (X_tr, y_tr, X_te, y_te, train_end) in folds:
        lo, hi = int(X_te.index.min()), int(X_te.index.max())
        if switch_index is None:
            phase, frac, sees = "pre", 0.0, False
        elif hi < switch_index:
            phase, frac, sees = "pre", 0.0, train_end > switch_index
        elif lo >= switch_index:
            phase, frac, sees = "post", 1.0, train_end > switch_index
        else:
            phase = "straddle"
            frac = (hi - switch_index + 1) / (hi - lo + 1)
            sees = train_end > switch_index
        out.append({"phase": phase, "train_sees_post": bool(sees),
                    "test_start": lo, "test_end": hi,
                    "train_end": int(train_end),
                    "frac_post_in_test": float(frac)})
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  PSO RUNNER — STANDARD OrPSOC  (condition 2 in ablation)
# ══════════════════════════════════════════════════════════════════════════════

def _make_scorer(X_tr, y_tr, X_p, y_p, X_v, y_v, feat_names, min_f, theta,
                 model_factory, criterion, criterion_kwargs):
    """
    Build the per-fold fitness closure used by both PSO runners.

    criterion=None  -> the project default: AUC on the single trailing 25%
                       validation split (FoldEvalContext / evaluate_ctx).
    criterion=<name> -> one of orpsoc_criteria.ALL_CRITERIA, scored by a
                       CriterionBank built from the TRAINING window only.

    NOTE: the theta blend assumes the criterion is on an AUC-like [0,1] scale.
    That holds for current / mean_k / median_k / min_k / mean_sd / mb_perf /
    pooled. It does NOT hold for mb_thresh, which is a signed sum of selection
    frequencies -- blending that with a compactness term is meaningless, so it
    is rejected here rather than silently producing nonsense.
    """
    if criterion is None:
        ctx = FoldEvalContext(X_p, y_p, X_v, y_v, feat_names,
                              model_factory=model_factory)

        def _score(pos):
            return evaluate_ctx(pos, ctx, min_f, theta)
        return _score

    if criterion in ("mb_thresh", "mb_stability"):
        raise ValueError(
            f"criterion={criterion!r} is not on an AUC scale and cannot be "
            f"blended with the compactness term. Use it as a standalone "
            f"ranking, not as a PSO fitness.")

    from orpsoc_criteria import CriterionBank
    # Tell the bank which criterion is actually wanted so it can skip building
    # the bootstrap/selection-frequency structures that only the mb_* criteria
    # use -- measured 160x cheaper to construct for a single block criterion,
    # and the bank is rebuilt once per condition per fold inside a run.
    bank = CriterionBank(X_tr, y_tr, feat_names, seed=0,
                         criteria=[criterion], **(criterion_kwargs or {}))
    n_feat = len(feat_names)

    def _score(pos):
        idx = np.where(pos == 1)[0]
        if len(idx) < min_f:
            return -1.0
        s = bank.score(idx, criterion)
        if not np.isfinite(s):
            return -1.0
        return float(theta * s + (1.0 - theta) * (1.0 - len(idx) / n_feat))
    return _score


def run_standard_orpsoc(X_tr, y_tr, X_te, y_te, feat_names,
                        seed=42, n_particles=20, max_iter=60,
                        cr=0.6, w_max=0.9, w_min=0.4, min_f=3,
                        theta=0.5, model_factory=None,
                        criterion=None, criterion_kwargs=None, **kwargs):
    """
    Standard OrPSOC: orthogonal init + two-point crossover, fixed cr,
    linear w decay.  No adaptive-c, no leadership update, no HMM.

    Used as Condition 2 in the ablation study (step7).
    Internal validation split: last 25% of X_tr used for fitness eval.
    Final test AUC evaluated on (X_te, y_te) using selected features.

    **kwargs absorbs any extra keys (e.g. cr_low, cr_high) passed from
    the ablation loop's shared pso_kw dict without crashing.
    """
    rng = np.random.RandomState(seed)
    n   = len(feat_names)
    t0  = time.time()

    # Internal validation split (temporal order preserved)
    cut  = int(len(X_tr) * 0.75)
    X_p, y_p = X_tr.iloc[:cut],  y_tr.iloc[:cut]
    X_v, y_v = X_tr.iloc[cut:],  y_tr.iloc[cut:]

    # Fitness closure: default trailing-window AUC, or a named criterion.
    _score = _make_scorer(X_tr, y_tr, X_p, y_p, X_v, y_v, feat_names,
                          min_f, theta, model_factory, criterion,
                          criterion_kwargs)

    # Position cache: same binary vector → skip re-evaluation (~26% fewer fits).
    # Key is the raw 0/1 bytes rather than a tuple of Python ints -- one small
    # bytes object instead of n_features boxed integers per lookup.
    _cache = {}
    def _eval(pos):
        key = pos.astype(np.uint8).tobytes()
        val = _cache.get(key)
        if val is None:
            val = _cache[key] = _score(pos)
        return val

    # Initialise
    init_pos  = build_orthogonal_positions(n_particles, n, seed)
    particles = []
    for i in range(n_particles):
        pos = init_pos[i].copy()
        particles.append({
            "pos":      pos.copy(),
            "vel":      rng.randn(n) * 0.1,
            "best_pos": pos.copy(),
            "best_fit": _eval(pos),
        })

    gbest_pos = max(particles, key=lambda p: p["best_fit"])["best_pos"].copy()
    gbest_fit = max(p["best_fit"] for p in particles)

    for it in range(max_iter):
        w = w_max - (w_max - w_min) * (it / max(max_iter - 1, 1))

        for p in particles:
            r1, r2 = rng.rand(n), rng.rand(n)
            p["vel"] = (w * p["vel"]
                        + 2.0 * r1 * (p["best_pos"] - p["pos"])
                        + 2.0 * r2 * (gbest_pos     - p["pos"]))
            new_pos = (rng.rand(n) < sigmoid(p["vel"])).astype(float)
            if new_pos.sum() < min_f:
                zeros = np.where(new_pos == 0)[0]
                need  = min_f - int(new_pos.sum())
                if len(zeros) >= need:
                    new_pos[rng.choice(zeros, size=need, replace=False)] = 1
            p["pos"] = new_pos

        # Crossover
        idx = list(range(n_particles))
        rng.shuffle(idx)
        for k in range(0, n_particles - 1, 2):
            pa, pb = particles[idx[k]], particles[idx[k + 1]]
            ca, cb = crossover(pa["pos"], pb["pos"], cr, min_f, rng)
            for child, parent in [(ca, pa), (cb, pb)]:
                fit = _eval(child)
                if fit > parent["best_fit"]:
                    parent["pos"]      = child
                    parent["best_pos"] = child.copy()
                    parent["best_fit"] = fit

        for p in particles:
            fit = _eval(p["pos"])
            if fit > p["best_fit"]:
                p["best_fit"] = fit
                p["best_pos"] = p["pos"].copy()
            if fit > gbest_fit:
                gbest_fit = fit
                gbest_pos = p["pos"].copy()

    sel = [feat_names[i] for i in np.where(gbest_pos == 1)[0]]

    # Final test AUC on held-out test fold
    try:
        final_model = (model_factory() if model_factory is not None else
                       LGBMClassifier(n_estimators=100, num_leaves=31,
                                      learning_rate=0.1, verbosity=-1,
                                      random_state=seed, n_jobs=1))
        pipe = Pipeline([
            ("imp",    SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model",  final_model),
        ])
        pipe.fit(X_tr[sel], y_tr)
        auc = roc_auc_score(y_te, pipe.predict_proba(X_te[sel])[:, 1])
    except Exception:
        auc = 0.5

    return {"auc": auc, "selected": sel, "n_sel": len(sel),
            "runtime": time.time() - t0}


# ══════════════════════════════════════════════════════════════════════════════
#  PSO RUNNER — HYBRID OrPSOC  (conditions 3 & 4 in ablation)
# ══════════════════════════════════════════════════════════════════════════════

def run_hybrid_orpsoc(X_tr, y_tr, X_te, y_te, feat_names,
                      hmm_trigger=False, seed=42,
                      n_particles=20, max_iter=60, min_f=3, theta=0.5,
                      cr_low=0.3, cr_high=0.8, w_max=0.9, w_min=0.4,
                      N_explore=15, lam=0.1, hmm_trigger_delay=7,
                      warm_start_pos=None, p_trans=None,
                      ramp_iters=5, elite_frac=0.2,
                      use_importance_reinit=True,
                      importance_window_frac=0.4,
                      apsoll_patience=1, apsoll_rearm_after=None,
                      apsoll_warmup=5, model_factory=None,
                      criterion=None, criterion_kwargs=None, **kwargs):
    """
    Hybrid OrPSOC: APSOLL adaptive-c + three-leader velocity
    + three-phase cr/w schedule.  Optionally triggered by HMM.

    Condition 3 (step7): call with hmm_trigger=False
      → APSOLL-c self-triggers via stagnation detection (c < 1.05)
    Condition 4 (step7): call with hmm_trigger=True
      → HMM signal is delayed by hmm_trigger_delay iterations so the
        swarm can complete initial convergence before leader-influence
        fires.  Without the delay, Phase 2 fires at iteration 0 on
        orthogonal positions whose leaders are random noise, which
        disrupts the early exploration that orthogonal init enables.

    warm_start_pos : optional np.ndarray of shape (n_features,)
      gbest position carried from the previous fold.
        • hmm_trigger=False → particle[0] is seeded from warm_start_pos so
          the swarm continues from a known-good solution (continuation).
        • hmm_trigger=True  → ELITE PRESERVATION / PARTIAL RESTART: instead
          of discarding everything (the old "clean slate" that caused the
          blind, destructive reset at the regime change), a fraction
          (elite_frac) of particles are seeded from the carried gbest and
          reinjected alongside the fresh orthogonal swarm. Retains useful
          pre-switch structure while still exploring the new regime.

    p_trans : optional float in [0, 1]
      HMM transition probability for the current fold. Scales the Phase 2
      burst intensity (proportional drift response): weak drift → modest
      increase in cr/w, strong drift → full burst. None → full-strength
      burst (used by the APSOLL self-trigger condition, which has no HMM).

    ramp_iters : int
      Number of iterations over which cr/w ramp UP into the Phase 2 burst,
      replacing the instant cr_low→cr_high jump that shook the whole swarm.

    PHASE LOGIC:
      Phase 1 — exploration: cr=cr_low, linear w decay.
        Transitions to Phase 2 when either:
          (a) APSOLL self-trigger: it > 5 and c_t < 1.05 (m reset → stagnation)
          (b) Delayed HMM trigger: hmm_trigger=True and it >= hmm_trigger_delay
      Phase 2 — exploitation burst: cr/w ramp from cr_low/w_min up to the
        drift-proportional targets (cr_target/w_target) over ramp_iters
        iterations, GWO leaders. Runs for N_explore iterations, then → Phase 3.
      Phase 3 — exponential blend back to standard: smooth decay of
        leader influence toward standard PSO.

    apsoll_patience / apsoll_rearm_after / apsoll_warmup
      Controls for the APSOLL self-trigger. See the degeneracy measurement in
      APSOLLAdaptiveC's docstring (work-order 2.1.b, empirically confirmed).

        apsoll_patience = 1, apsoll_rearm_after = None   (DEFAULTS)
          LEGACY behaviour, bit-identical to the original implementation:
          fire on a single flat gbest step, and Phase 3 is terminal.

        apsoll_patience = k > 1
          PATIENCE COUNTER (work-order 2.1.c). Require k CONSECUTIVE
          non-improving iterations before declaring stagnation -- exactly the
          early-stopping patience idiom. This is what separates "the swarm has
          genuinely converged" from "one flat step, which happens on ~80% of
          iterations".

        apsoll_rearm_after = r
          RE-ARMING. After r iterations in Phase 3 (by which point the leader
          influence has exponentially decayed back toward standard PSO), return
          to Phase 1 and clear the patience counter. Without this the trigger
          can fire at most ONCE per run -- measured to be at iteration 6-8 in
          12/12 runs -- so genuine late-run stagnation around iteration 40, when
          the swarm has actually converged, can never be detected.

      Legacy default is deliberate: guardrail G4 says measure before changing
      the mechanism. Opt in explicitly and report the before/after trigger-
      iteration distribution (2.1.c requires that the choice not be silent).

    NOTE on c < 1.05 threshold (legacy path):
      c = (m/T)^(2/3) + 1 and m resets to 0 on stagnation, giving c=1.0.
      c < 1.05 therefore detects m=0 (no improvement in the last step),
      which is a minimal stagnation signal.  We require it > 5 to avoid
      triggering on the random-walk noise of the first few iterations.

    FIX: elif prevents double-decrement of n_explore_rem on Phase 1→2 transition.

    **kwargs absorbs any extra keys passed from the ablation loop's
    shared pso_kw dict without crashing.
    """
    rng = np.random.RandomState(seed)
    n   = len(feat_names)
    t0  = time.time()

    # Internal validation split (temporal)
    cut  = int(len(X_tr) * 0.75)
    X_p, y_p = X_tr.iloc[:cut],  y_tr.iloc[:cut]
    X_v, y_v = X_tr.iloc[cut:],  y_tr.iloc[cut:]

    # Fitness closure: default trailing-window AUC, or a named criterion.
    _score = _make_scorer(X_tr, y_tr, X_p, y_p, X_v, y_v, feat_names,
                          min_f, theta, model_factory, criterion,
                          criterion_kwargs)

    # Position cache: same binary vector → skip re-evaluation.
    _cache = {}
    def _eval(pos):
        key = pos.astype(np.uint8).tobytes()
        val = _cache.get(key)
        if val is None:
            val = _cache[key] = _score(pos)
        return val

    # Initialise
    init_pos  = build_orthogonal_positions(n_particles, n, seed)
    particles = []
    for i in range(n_particles):
        pos = init_pos[i].copy()
        particles.append({
            "pos":      pos.copy(),
            "vel":      rng.randn(n) * 0.1,
            "best_pos": pos.copy(),
            "best_fit": _eval(pos),
        })

    # Warm-start / elite preservation.
    #   • No regime change (hmm_trigger=False): seed particle[0] from the
    #     previous fold's gbest — known-good continuation.
    #   • Regime change (hmm_trigger=True): PARTIAL RESTART / POPULATION
    #     MEMORY. Rather than discarding all pre-switch knowledge (the old
    #     "clean slate" that caused the blind, destructive reset at fold 5),
    #     keep elite_frac of the swarm seeded from the carried gbest and
    #     reinject them alongside the fresh orthogonal particles. The elites
    #     retain useful structure; the remaining orthogonal particles, plus
    #     the Phase 2 burst, explore the new regime.
    def _enforce_min_f(vec):
        if vec.sum() < min_f:
            zeros = np.where(vec == 0)[0]
            need  = min_f - int(vec.sum())
            if len(zeros) >= need:
                vec[rng.choice(zeros, size=need, replace=False)] = 1
        return vec

    if warm_start_pos is not None:
        ws = _enforce_min_f(warm_start_pos.copy())
        if not hmm_trigger:
            particles[0]["pos"]      = ws
            particles[0]["best_pos"] = ws.copy()
            particles[0]["best_fit"] = _eval(ws)
        else:
            elite_k = _elite_count(elite_frac, n_particles)
            for i in range(elite_k):
                if i == 0:
                    ep = ws.copy()                       # exact elite
                else:
                    ep = ws.copy()                       # diversified elite
                    flip = rng.choice(n, size=max(1, n // 20), replace=False)
                    ep[flip] = 1 - ep[flip]
                    ep = _enforce_min_f(ep)
                particles[i]["pos"]      = ep
                particles[i]["best_pos"] = ep.copy()
                particles[i]["best_fit"] = _eval(ep)

    # ── IMPORTANCE-GUIDED REINIT of the NON-elite particles ─────────────────
    # Professor suggestion #2: on a regime-change trigger, seed the fresh
    # (non-elite) particles from the classifier's feature importances on the
    # MOST RECENT training window, instead of the blind orthogonal draw they
    # currently hold. Elites (indices 0..elite_k-1) are left untouched so the
    # partial-restart / population-memory behaviour is preserved; only the
    # explore-from-scratch particles get the data-driven guidance.
    # Applies whenever hmm_trigger is set, with or without a warm start
    # (e.g. step_real_data calls the full-hybrid condition without warm_start).
    if hmm_trigger and use_importance_reinit:
        elite_k = (_elite_count(elite_frac, n_particles)
                   if warm_start_pos is not None else 0)
        n_fresh = n_particles - elite_k
        if n_fresh > 0:
            prob  = windowed_feature_importance(
                X_tr, y_tr, feat_names,
                window_frac=importance_window_frac, seed=seed)
            fresh = build_importance_guided_positions(
                n_fresh, n, prob, min_f=min_f, seed=seed + 777)
            for j in range(n_fresh):
                idx_p = elite_k + j
                pos   = _enforce_min_f(fresh[j].copy())
                particles[idx_p]["pos"]      = pos
                particles[idx_p]["vel"]      = rng.randn(n) * 0.1
                particles[idx_p]["best_pos"] = pos.copy()
                particles[idx_p]["best_fit"] = _eval(pos)

    gbest_pos = max(particles, key=lambda p: p["best_fit"])["best_pos"].copy()
    gbest_fit = max(p["best_fit"] for p in particles)

    # Always start in Phase 1 so orthogonal init can do its job.
    # When hmm_trigger=True, Phase 2 is entered at iteration hmm_trigger_delay
    # (not iteration 0) so leaders have had a chance to earn their position.
    adap_c           = APSOLLAdaptiveC(max_iter)
    phase            = 1
    dt               = 0
    n_explore_rem    = N_explore
    forced_phase2_at = hmm_trigger_delay if hmm_trigger else None

    # Diagnostics for work-order 2.1.a: WHEN did the self-trigger actually fire?
    apsoll_trigger_iters = []   # every iteration at which it caused 1 -> 2
    phase3_dt            = 0    # iterations spent in Phase 3 (re-arm counter).
                                # Separate from `dt`, which deliberately carries
                                # over from Phase 2 to drive the decay envelope.
    n_rearms             = 0

    # Proportional drift response: scale the Phase 2 burst by the HMM
    # transition probability p_trans (∈[0,1]) instead of applying a fixed
    # cr_high/w_max whenever a binary threshold is crossed. Weak drift → modest
    # burst, strong drift → full burst. p_trans=None (APSOLL self-trigger, no
    # HMM signal) falls back to a full-strength burst.
    drift_strength = 1.0 if p_trans is None else float(np.clip(p_trans, 0.0, 1.0))
    cr_target      = cr_low + (cr_high - cr_low) * drift_strength
    w_target       = w_min  + (w_max  - w_min)  * drift_strength

    for it in range(max_iter):
        c_t = adap_c.update(gbest_fit)

        # ── FIXED: elif prevents double-decrement when entering Phase 2 ──────
        if phase == 1:
            cr_t = cr_low
            w_t  = w_max - (w_max - w_min) * (it / max(max_iter - 1, 1))
            # Transition to Phase 2 on either internal APSOLL stagnation
            # signal OR delayed external HMM signal
            if apsoll_patience <= 1:
                # LEGACY: single flat gbest step. Bit-identical to the original.
                apsoll_trigger = it > apsoll_warmup and c_t < 1.05
            else:
                # PATIENCE: require k consecutive non-improving iterations.
                apsoll_trigger = (it > apsoll_warmup
                                  and adap_c.stagnant_streak >= apsoll_patience)
            hmm_delayed_trigger = (forced_phase2_at is not None
                                   and it >= forced_phase2_at)
            if apsoll_trigger or hmm_delayed_trigger:
                phase = 2
                dt    = 0
                n_explore_rem = N_explore
                phase3_dt = 0
                if apsoll_trigger:
                    apsoll_trigger_iters.append(it)

        elif phase == 2:
            # Gradual ramp INTO the burst over ramp_iters iterations rather
            # than an instant cr_low→cr_high jump (the instant reset "shook the
            # whole swarm" and drove the fold-5 transition crash). dt counts
            # iterations since Phase 2 began (reset to 0 at the 1→2 transition).
            ramp = min(1.0, (dt + 1) / max(ramp_iters, 1))
            cr_t = cr_low + (cr_target - cr_low) * ramp
            w_t  = w_min  + (w_target  - w_min)  * ramp
            n_explore_rem -= 1
            dt += 1
            if n_explore_rem <= 0:
                phase = 3

        elif phase == 3:
            cr_t = cr_low  + (cr_target - cr_low) * np.exp(-lam * dt)
            w_t  = w_min   + (w_target  - w_min)  * np.exp(-lam * dt)
            dt  += 1
            phase3_dt += 1
            # RE-ARM: by now the leader influence has decayed back toward
            # standard PSO, so returning to Phase 1 costs nothing and lets a
            # genuine late-run stagnation be detected. Legacy (None) leaves
            # Phase 3 terminal, which caps the trigger at once per run.
            if (apsoll_rearm_after is not None
                    and phase3_dt >= apsoll_rearm_after):
                phase = 1
                phase3_dt = 0
                adap_c.reset_stagnation()   # patience must re-accumulate
                n_rearms += 1

        else:
            cr_t = cr_low
            w_t  = w_min

        # ── Top-3 leaders for leadership velocity ────────────────────────────
        sorted_p = sorted(particles, key=lambda p: p["best_fit"], reverse=True)
        top3     = [p["best_pos"] for p in sorted_p[:3]]

        # ── Velocity update ───────────────────────────────────────────────────
        for p in particles:
            r1, r2, r4 = rng.rand(n), rng.rand(n), rng.rand(n)
            std = (w_t * p["vel"]
                   + (c_t / 2) * r1 * (p["best_pos"] - p["pos"])
                   + (c_t / 2) * r2 * (gbest_pos     - p["pos"]))

            if phase == 1:
                vel = std
            else:
                # GWO-style leader positions (APSOLL eq. 3)
                X1 = top3[0] - np.abs(2 * rng.rand(n) * top3[0] - p["pos"])
                X2 = top3[1] - np.abs(2 * rng.rand(n) * top3[1] - p["pos"])
                X3 = top3[2] - np.abs(2 * rng.rand(n) * top3[2] - p["pos"])
                ldr = (w_t * p["vel"]
                       + (c_t / 2) * r4 * (X1 - p["pos"])
                       + (c_t / 3) * r4 * (X2 - p["pos"])
                       + (c_t / 4) * r4 * (X3 - p["pos"]))
                if phase == 2:
                    vel = ldr
                else:   # Phase 3: smooth exponential blend
                    blend = np.exp(-0.1 * dt)
                    vel   = blend * ldr + (1.0 - blend) * std

            p["vel"] = vel
            new_pos = (rng.rand(n) < sigmoid(vel)).astype(float)
            if new_pos.sum() < min_f:
                zeros = np.where(new_pos == 0)[0]
                need  = min_f - int(new_pos.sum())
                if len(zeros) >= need:
                    new_pos[rng.choice(zeros, size=need, replace=False)] = 1
            p["pos"] = new_pos

        # ── Crossover ─────────────────────────────────────────────────────────
        idx = list(range(n_particles))
        rng.shuffle(idx)
        for k in range(0, n_particles - 1, 2):
            pa, pb = particles[idx[k]], particles[idx[k + 1]]
            ca, cb = crossover(pa["pos"], pb["pos"], cr_t, min_f, rng)
            for child, parent in [(ca, pa), (cb, pb)]:
                fit = _eval(child)
                if fit > parent["best_fit"]:
                    parent["pos"]      = child
                    parent["best_pos"] = child.copy()
                    parent["best_fit"] = fit

        # ── Evaluate + update bests ───────────────────────────────────────────
        for p in particles:
            fit = _eval(p["pos"])
            if fit > p["best_fit"]:
                p["best_fit"] = fit
                p["best_pos"] = p["pos"].copy()
            if fit > gbest_fit:
                gbest_fit = fit
                gbest_pos = p["pos"].copy()

    sel = [feat_names[i] for i in np.where(gbest_pos == 1)[0]]

    # Final test AUC — LightGBM consistent with PSO fitness evaluator
    try:
        final_model = (model_factory() if model_factory is not None else
                       LGBMClassifier(n_estimators=100, num_leaves=31,
                                      learning_rate=0.1, verbosity=-1,
                                      random_state=seed, n_jobs=1))
        pipe = Pipeline([
            ("imp",    SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model",  final_model),
        ])
        pipe.fit(X_tr[sel], y_tr)
        auc = roc_auc_score(y_te, pipe.predict_proba(X_te[sel])[:, 1])
    except Exception:
        auc = 0.5

    return {"auc": auc, "selected": sel, "n_sel": len(sel),
            "runtime": time.time() - t0,
            "gbest_pos": gbest_pos.copy(),
            # work-order 2.1.a instrumentation
            "apsoll_trigger_iters": list(apsoll_trigger_iters),
            "apsoll_trigger_iter": (apsoll_trigger_iters[0]
                                    if apsoll_trigger_iters else None),
            "apsoll_n_rearms": n_rearms,
            "apsoll_max_c": (max(adap_c.c_hist) if adap_c.c_hist else None)}


print("orpsoc_utils.py loaded — shared utilities ready.")