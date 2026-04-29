"""
orpsoc_utils.py — Shared utilities for the full hybrid pipeline
================================================================
Import this in every step file. Contains:
  - sigmoid / build_orthogonal_positions / partial_reinit
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
"""

import numpy as np
import pandas as pd
import time
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")


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
        pipe = Pipeline([
            ("imp",    SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model",  XGBClassifier(
                n_estimators=80, max_depth=4,
                learning_rate=0.1, verbosity=0, random_state=42
            ))
        ])
        pipe.fit(X_tr[cols], y_tr)
        auc = roc_auc_score(y_va, pipe.predict_proba(X_va[cols])[:, 1])
    except Exception:
        return -1.0

    compactness = 1.0 - len(idx) / len(feat_names)
    return float(theta * auc + (1.0 - theta) * compactness)


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
                 cusum_h: float = 5.0):
        self.method       = method
        self.lookback     = lookback
        self.k            = percentile_k
        self.slack        = cusum_slack
        self.h            = cusum_h
        self.history      = []
        self.cusum_S      = 0.0
        self.triggered    = False
        self.threshold    = 0.5
        self.trigger_log  = []   # (iteration, p_trans, threshold) tuples

    def update(self, p_trans: float, iteration: int = -1) -> bool:
        """
        Feed one observation. Returns True if regime change detected.
        Call once per walk-forward fold with P(Trans|t) from the HMM.
        """
        self.history.append(p_trans)
        window = self.history[-self.lookback:]

        if self.method == "percentile":
            if len(window) >= 5:
                self.threshold = float(np.percentile(window, self.k))
            self.triggered = bool(p_trans > self.threshold)

        elif self.method == "cusum":
            mu_ref = float(np.mean(window[:-1])) if len(window) > 1 else 0.3
            self.cusum_S = max(0.0,
                self.cusum_S + (p_trans - mu_ref - self.slack))
            self.triggered = bool(self.cusum_S > self.h)
            if self.triggered:
                self.cusum_S = 0.0   # reset after detection

        else:
            raise ValueError(
                f"Unknown method: {self.method}. Use 'percentile' or 'cusum'.")

        if self.triggered:
            self.trigger_log.append((iteration, p_trans, self.threshold))

        return self.triggered

    def summary(self) -> dict:
        return {
            "method":          self.method,
            "n_triggers":      len(self.trigger_log),
            "triggers":        self.trigger_log,
            "final_threshold": self.threshold,
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

    # Initialise
    init_pos  = build_orthogonal_positions(n_particles, n, seed)
    particles = []
    for i in range(n_particles):
        pos = init_pos[i].copy()
        particles.append({
            "pos":      pos.copy(),
            "vel":      rng.randn(n) * 0.1,
            "best_pos": pos.copy(),
            "best_fit": evaluate(pos, feat_names, X_p, y_p, X_v, y_v, min_f, theta),
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
                fit = evaluate(child, feat_names, X_p, y_p, X_v, y_v, min_f, theta)
                if fit > parent["best_fit"]:
                    parent["pos"]      = child
                    parent["best_pos"] = child.copy()
                    parent["best_fit"] = fit

        for p in particles:
            fit = evaluate(p["pos"], feat_names, X_p, y_p, X_v, y_v, min_f, theta)
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
            ("model",  XGBClassifier(n_estimators=80, max_depth=4,
                                     learning_rate=0.1, verbosity=0,
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
                      N_explore=15, lam=0.1, **kwargs):
    """
    Hybrid OrPSOC: APSOLL adaptive-c + three-leader velocity
    + three-phase cr/w schedule.  Optionally triggered by HMM.

    Condition 3 (step7): call with hmm_trigger=False
      → APSOLL-c self-triggers via stagnation detection
    Condition 4 (step7): call with hmm_trigger=True
      → HMM detection passed in externally, swarm enters Phase 2 at start

    FIX: Phase logic uses elif (not if) to prevent double-decrement
    of n_explore_rem when transitioning from Phase 1 → Phase 2.

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

    # Initialise
    init_pos  = build_orthogonal_positions(n_particles, n, seed)
    particles = []
    for i in range(n_particles):
        pos = init_pos[i].copy()
        particles.append({
            "pos":      pos.copy(),
            "vel":      rng.randn(n) * 0.1,
            "best_pos": pos.copy(),
            "best_fit": evaluate(pos, feat_names, X_p, y_p, X_v, y_v, min_f, theta),
        })

    gbest_pos = max(particles, key=lambda p: p["best_fit"])["best_pos"].copy()
    gbest_fit = max(p["best_fit"] for p in particles)

    # Phase state — if HMM fired externally, start in Phase 2 immediately
    adap_c        = APSOLLAdaptiveC(max_iter)
    phase         = 2 if hmm_trigger else 1
    dt            = 0
    n_explore_rem = N_explore

    for it in range(max_iter):
        c_t = adap_c.update(gbest_fit)

        # ── FIXED: elif prevents double-decrement when entering Phase 2 ──────
        if phase == 1:
            cr_t = cr_low
            w_t  = w_max - (w_max - w_min) * (it / max(max_iter - 1, 1))
            # Soft self-trigger: c collapses → stagnation detected
            if it > 5 and c_t < 1.05:
                phase = 2
                dt    = 0
                n_explore_rem = N_explore

        elif phase == 2:
            cr_t = cr_high
            w_t  = w_max
            n_explore_rem -= 1
            dt += 1
            if n_explore_rem <= 0:
                phase = 3

        elif phase == 3:
            cr_t = cr_low  + (cr_high - cr_low) * np.exp(-lam * dt)
            w_t  = w_min   + (w_max   - w_min)  * np.exp(-lam * dt)
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
                fit = evaluate(child, feat_names, X_p, y_p, X_v, y_v, min_f, theta)
                if fit > parent["best_fit"]:
                    parent["pos"]      = child
                    parent["best_pos"] = child.copy()
                    parent["best_fit"] = fit

        # ── Evaluate + update bests ───────────────────────────────────────────
        for p in particles:
            fit = evaluate(p["pos"], feat_names, X_p, y_p, X_v, y_v, min_f, theta)
            if fit > p["best_fit"]:
                p["best_fit"] = fit
                p["best_pos"] = p["pos"].copy()
            if fit > gbest_fit:
                gbest_fit = fit
                gbest_pos = p["pos"].copy()

    sel = [feat_names[i] for i in np.where(gbest_pos == 1)[0]]

    # Final test AUC
    try:
        pipe = Pipeline([
            ("imp",    SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model",  XGBClassifier(n_estimators=80, max_depth=4,
                                     learning_rate=0.1, verbosity=0,
                                     random_state=seed))
        ])
        pipe.fit(X_tr[sel], y_tr)
        auc = roc_auc_score(y_te, pipe.predict_proba(X_te[sel])[:, 1])
    except Exception:
        auc = 0.5

    return {"auc": auc, "selected": sel, "n_sel": len(sel),
            "runtime": time.time() - t0}


print("orpsoc_utils.py loaded — shared utilities ready.")