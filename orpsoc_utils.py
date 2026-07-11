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
             theta: float = 0.7) -> float:
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
                                   learning_rate=0.1, verbosity=-1, random_state=42)
        else:
            model = LGBMClassifier(n_estimators=100, num_leaves=31,
                                   learning_rate=0.1, verbosity=-1, random_state=42)
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
                                      random_state=seed)),
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
    """
    def __init__(self, max_iter: int):
        self.max_iter = max_iter
        self.m        = 0
        self.c_hist   = []
        self.prev_fit = None

    def update(self, current_fit: float) -> float:
        if self.prev_fit is not None and current_fit > self.prev_fit:
            self.m += 1
        else:
            self.m  = 0
        self.prev_fit = current_fit
        c = (self.m / max(self.max_iter, 1)) ** (2.0 / 3.0) + 1.0
        c = float(np.clip(c, 1.0, 2.0))
        self.c_hist.append(c)
        return c


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
                 cooldown: int = 2):
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

def feature_stability_ratio(selected_sets: list) -> dict:
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

    mid  = len(jaccard) // 2
    pre  = float(np.mean(jaccard[:mid]))  if mid > 0           else 1.0
    post = float(np.mean(jaccard[mid:]))  if mid < len(jaccard) else 1.0

    return {
        "per_fold_jaccard":       jaccard,
        "pre_regime_stability":   pre,
        "post_regime_stability":  post,
        "regime_adaptation_drop": pre - post,
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
#  PSO RUNNER — STANDARD OrPSOC  (condition 2 in ablation)
# ══════════════════════════════════════════════════════════════════════════════

def run_standard_orpsoc(X_tr, y_tr, X_te, y_te, feat_names,
                        seed=42, n_particles=20, max_iter=60,
                        cr=0.6, w_max=0.9, w_min=0.4, min_f=3,
                        theta=0.7, **kwargs):
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

    # Position cache: same binary vector → skip re-evaluation (~30% fewer fits)
    _cache = {}
    def _eval(pos):
        key = tuple(pos.astype(int))
        if key not in _cache:
            _cache[key] = evaluate(pos, feat_names, X_p, y_p, X_v, y_v, min_f, theta)
        return _cache[key]

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
        pipe = Pipeline([
            ("imp",    SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model",  LGBMClassifier(n_estimators=100, num_leaves=31,
                                      learning_rate=0.1, verbosity=-1,
                                      random_state=seed))
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
                      n_particles=20, max_iter=60, min_f=3, theta=0.7,
                      cr_low=0.3, cr_high=0.8, w_max=0.9, w_min=0.4,
                      N_explore=15, lam=0.1, hmm_trigger_delay=7,
                      warm_start_pos=None, p_trans=None,
                      ramp_iters=5, elite_frac=0.2,
                      use_importance_reinit=True,
                      importance_window_frac=0.4, **kwargs):
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

    NOTE on c < 1.05 threshold:
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

    # Position cache: same binary vector → skip re-evaluation
    _cache = {}
    def _eval(pos):
        key = tuple(pos.astype(int))
        if key not in _cache:
            _cache[key] = evaluate(pos, feat_names, X_p, y_p, X_v, y_v, min_f, theta)
        return _cache[key]

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
            elite_k = max(1, int(round(elite_frac * n_particles)))
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
        elite_k = (max(1, int(round(elite_frac * n_particles)))
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
            apsoll_trigger      = it > 5 and c_t < 1.05
            hmm_delayed_trigger = (forced_phase2_at is not None
                                   and it >= forced_phase2_at)
            if apsoll_trigger or hmm_delayed_trigger:
                phase = 2
                dt    = 0
                n_explore_rem = N_explore

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
        pipe = Pipeline([
            ("imp",    SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model",  LGBMClassifier(n_estimators=100, num_leaves=31,
                                      learning_rate=0.1, verbosity=-1,
                                      random_state=seed))
        ])
        pipe.fit(X_tr[sel], y_tr)
        auc = roc_auc_score(y_te, pipe.predict_proba(X_te[sel])[:, 1])
    except Exception:
        auc = 0.5

    return {"auc": auc, "selected": sel, "n_sel": len(sel),
            "runtime": time.time() - t0,
            "gbest_pos": gbest_pos.copy()}


print("orpsoc_utils.py loaded — shared utilities ready.")