"""
STEP 7 — Ablation Study & Comparison Matrix
=============================================
Layer 5 in the full 7-layer architecture.

Runs 4 conditions × 4 levels × 30 seeds × 8 folds.
This is the experiment that generates your paper's main results table.

FIX LOG:
  - run_standard_orpsoc and run_hybrid_orpsoc now IMPORTED from
    orpsoc_utils (fixes NameError that crashed the ablation loop)
  - APSOLLAdaptiveC also imported (no longer needs inlining here)
  - fillna(method="bfill") → .bfill()  (pandas FutureWarning fix)
  - Phase logic in local run_hybrid_orpsoc uses elif (double-decrement fix)
  - N_SPLITS saved into config dict so step8 can read it without NameError
  - Duplicate "+APSOLL" key removed from CONDITIONS dict

Run with:
    python step7_ablation.py
"""

# ── import guard ─────────────────────────────────────────────────────────────
# This file is a SCRIPT, not a module. It executes its whole pipeline at module
# level, so `import step7_ablation` runs the entire thing as a side effect --
# for step7_ablation that is a 4-hour ablation triggered by an innocent-looking
# import. Fail loudly instead.
#
# `globals().get("__name__", ...)` rather than a bare `__name__`: helpers are
# reused by exec'ing the section above a marker into a fresh namespace (see
# experiments/apsoll_sweep.py), and that namespace has no __name__ at all.
if globals().get("__name__", "__main__") != "__main__":
    raise ImportError(
        "step7_ablation.py is a script, not an importable module -- importing it would "
        "execute the full pipeline. To reuse a helper, exec the section above "
        "the main loop into a fresh namespace (see experiments/apsoll_sweep.py)."
    )
# ─────────────────────────────────────────────────────────────────────────────


import os
# Pin BLAS/OpenMP BEFORE numpy and lightgbm are imported -- they read these at
# load time. Without this, each of the N worker processes would additionally
# spawn one OpenMP thread per core. See orpsoc_runner.pin_threads().
from orpsoc_runner import (pin_threads, default_workers, provenance,
                           CheckpointStore)
pin_threads(1)

import numpy as np
import pandas as pd
import pickle
import json
import time
import warnings
warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from functools import partial
os.makedirs("results", exist_ok=True)
os.makedirs("plots",   exist_ok=True)

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier
from scipy.stats import wilcoxon   # for the importance-reinit ablation test

from orpsoc_utils import (
    sigmoid, build_orthogonal_positions, partial_reinit,
    crossover, hamming_diversity, evaluate,
    walk_forward_folds, feature_stability_ratio,
    AdaptiveRegimeThreshold, classify_folds,
    run_standard_orpsoc,   # FIX: imported so it exists in scope
    run_hybrid_orpsoc,     # FIX: imported so it exists in scope
)

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

FAST_MODE   = False    # True = reduced iters for debugging; False = paper runs
N_SEEDS     = 10 if FAST_MODE else 30   # 10 seeds gives enough variance signal
MAX_ITER    = 20 if FAST_MODE else 60
N_PARTICLES = 10 if FAST_MODE else 20
N_SPLITS    = 6  if FAST_MODE else 8

# ── Execution controls (do NOT affect any computed number) ───────────────────
# N_JOBS parallelises over SEEDS ONLY. Folds always run sequentially, in
# chronological order, inside a single worker -- the walk-forward structure is
# untouched. Override with ORPSOC_N_JOBS=<n> in the environment.
N_JOBS      = default_workers()
# Per-(level, seed) checkpointing. Resuming is safe because a seed is a
# complete, self-contained walk-forward sequence with order-independent RNG.
# Checkpoints are scoped by a hash of CONFIG + the source of orpsoc_utils.py
# and this file, so a config or code change can never silently reload stale
# numbers (guardrail G3). Set to False to always recompute from scratch.
USE_CHECKPOINTS = True

# ── Benchmark version ────────────────────────────────────────────────────────
# "v2" is the corrected benchmark (make_benchmark_v2.py): five INDEPENDENT
# signal latents instead of five noisy copies of one, noise features drawn from
# the same process as the signal so the two are separable only through y, and
# L3 carrying genuine CONCEPT drift rather than a covariate shift that leaves
# the decision boundary stationary.
#
# "v1" reproduces the original step1 datasets. v1 and v2 numbers are NOT
# comparable -- the signal structure differs entirely -- so the output filename
# carries the version and step8 is pointed at whichever you want to analyse.
BENCHMARK_VERSION = "v2"        # "v1" | "v2"
_P = "" if BENCHMARK_VERSION == "v1" else "v2_"

LEVELS = {
    f"{_P}null":          f"Level 0 — True Null ({BENCHMARK_VERSION})",
    f"{_P}white_noise":   f"Level 1 — White Noise ({BENCHMARK_VERSION})",
    f"{_P}ar1":           f"Level 2 — AR(1) Stationary ({BENCHMARK_VERSION})",
    f"{_P}drift":         f"Level 3 — Drift ({BENCHMARK_VERSION})",
    f"{_P}regime_switch": f"Level 4 — Regime Switch ({BENCHMARK_VERSION})",
}

# Row index of the known structural break per level, or None if stationary.
# Consumed by orpsoc_utils.classify_folds() to label folds pre/straddle/post
# from the ACTUAL break location instead of assuming it sits at N_SPLITS // 2.
SWITCH_INDEX = {
    f"{_P}null": None, f"{_P}white_noise": None, f"{_P}ar1": None,
    f"{_P}drift": None, f"{_P}regime_switch": 500,
}

# ── Secondary pass: repeat the ablation with a LOW-CAPACITY classifier ───────
# LightGBM is an embedded feature selector, so an external wrapper competes with
# machinery the model already has -- measured on real data, the same 5-feature
# subset is worth -0.020 to LightGBM and +0.055 to logistic regression. Running
# the ablation a second time under LogReg turns "selection does not help" into a
# quantified statement about classifier capacity.
#
# Restricted to the two non-stationary levels, which is where the question
# bites; the stationary levels add cost without adding evidence. Set to [] to
# skip the pass entirely. Results are stored under "<level>@logreg" so they sit
# beside the primary numbers without colliding with them.
LOGREG_PASS_LEVELS = [f"{_P}drift", f"{_P}regime_switch"]

# Fitness criterion for the LogReg pass. The primary LightGBM pass stays on the
# project default (`current`, i.e. AUC on the single trailing validation split);
# the LogReg pass uses median_k, which is where the evidence supports it:
#
#   median_k vs current, actual PSO, 5 seeds (experiments/phase3_mediank.py)
#     sector_etf . LogReg    0.8157 -> 0.8312   +0.0155   and seed sd 0.0185 -> 0.0062
#     sector_etf . LightGBM  0.7997 -> 0.7843   -0.0154
#
# median_k is the replicated winner of the criterion bake-off on both datasets
# (33% +-10 real, 43% +-12 synthetic, vs current's 23% and 28%), but it helps
# only the low-capacity classifier in an actual search. Applying it per-pass
# reports it where it works instead of averaging the two effects away.
LOGREG_PASS_CRITERION = "median_k"      # None = use the default criterion

# Output is versioned so a v2 run cannot silently overwrite v1 numbers.
RESULTS_PATH = ("results/step7_ablation.json" if BENCHMARK_VERSION == "v1"
                else f"results/step7_ablation_{BENCHMARK_VERSION}.json")

# ── Fitness compactness weight (work-order: expose theta) ────────────────────
# Fitness = THETA * AUC + (1 - THETA) * (1 - k/N).
#   THETA = 1.0 -> pure AUC, no compactness pressure
#   THETA = 0.5 -> project default (see below)
# The marginal cost of one extra feature is (1-THETA)/N, so a feature must buy
# more than (1-THETA)/(N*THETA) AUC to be worth keeping. At THETA=0.5, N=50 that
# is 0.0200; at the former default of 0.7 it was 0.0086.
#
# THETA IS A PARETO DIAL, NOT A NUISANCE PARAMETER. Measured frontier on v1
# regime_switch (Standard OrPSOC, 4 seeds, 60 iters, 20 particles):
#     theta=0.5  AUC 0.8500 (95.8% of baseline)   9.50 features (19% of 50)
#     theta=0.7  AUC 0.8482 (95.6%)              11.03 features (22%)  DOMINATED
#     theta=0.9  AUC 0.8505 (95.8%)              14.88 features (30%)
#     theta=1.0  AUC 0.8880 (100.1%)             25.62 features (51%)
# theta=0.7 is dominated by theta=0.5: same AUC, fewer features. Hence the
# default moves to 0.5, the compact end of the frontier. theta=1.0 removes the
# compactness term entirely and reaches AUC parity with the all-features
# baseline, but at 51% of the features it no longer supports the compactness
# claim. Set THETA_SWEEP to a list to report the whole frontier.
THETA        = 0.5
THETA_SWEEP  = None      # e.g. [0.5, 0.7, 0.9, 1.0]

# ── APSOLL stagnation trigger (work-order 2.1.c) ─────────────────────────────
# (1, None) = LEGACY: fire on a single flat gbest step, Phase 3 terminal.
# Measured degeneracy at this setting (40 runs, 60 iters, 20 particles):
#   first-fire histogram {6: 26, 7: 10, 8: 4}, never-fired 0/40 -- i.e. EVERY
#   trigger in every run fired at iteration 6-8 and could never fire again.
# Patience + re-arm spreads firing across the run; see APSOLLAdaptiveC.
# Kept at legacy until a seeded sweep justifies a value (G4: measure first,
# 2.1.c: do not pick one silently).
APSOLL_PATIENCE    = 1
APSOLL_REARM_AFTER = None

# THETA and the APSOLL settings all change the numbers, so they belong in the
# provenance hash -- otherwise a theta sweep would silently reuse checkpoints
# written at a different theta (guardrail G3).
CONFIG = {"fast_mode": FAST_MODE, "n_seeds": N_SEEDS, "max_iter": MAX_ITER,
          "n_particles": N_PARTICLES, "n_splits": N_SPLITS,
          "theta": THETA, "theta_sweep": THETA_SWEEP,
          "apsoll_patience": APSOLL_PATIENCE,
          "apsoll_rearm_after": APSOLL_REARM_AFTER,
          "levels": list(LEVELS),
          "benchmark_version": BENCHMARK_VERSION,
          "logreg_pass_levels": LOGREG_PASS_LEVELS,
          "logreg_pass_criterion": LOGREG_PASS_CRITERION}
PROV   = provenance(CONFIG, ["orpsoc_utils.py", "step7_ablation.py"])

print("=" * 65)
print("  STEP 7: Ablation Study")
print(f"  Mode: {'FAST (debug)' if FAST_MODE else 'FULL (paper)'}")
print(f"  Seeds={N_SEEDS}  MaxIter={MAX_ITER}  Particles={N_PARTICLES}"
      f"  Splits={N_SPLITS}")
print(f"  Workers={N_JOBS} (seeds in parallel; folds always sequential)")
print(f"  Provenance={PROV['hash']}  checkpoints={'on' if USE_CHECKPOINTS else 'off'}")
print("=" * 65)
print()


# ══════════════════════════════════════════════════════════════════════════════
#  HMM — inlined from step5 so step7 is self-contained
# ══════════════════════════════════════════════════════════════════════════════

class SimpleHMM:
    """2-state Gaussian HMM fitted with Baum-Welch EM. (Copied from step5.)"""

    def __init__(self, n_iter=50, tol=1e-4):
        self.n_iter = n_iter
        self.tol    = tol
        self.fitted = False
        self.pi = self.A = self.mu = self.sigma = None

    def _emission_prob(self, x):
        eps   = 1e-8
        probs = np.zeros((len(x), 2))
        for k in range(2):
            diff = x - self.mu[k]
            sig  = self.sigma[k] + eps
            # BUGFIX: identical fix + full rationale as step9_real_data.py's
            # SimpleHMM._emission_prob(). Was "/(sig + sqrt(2*pi))" (additive,
            # wrong); correct Gaussian normalizer is "sig * sqrt(2*pi)". This
            # was the dominant cause of P(Transition) sticking near 1.0
            # permanently once any crisis entered the training window.
            probs[:, k] = np.exp(-0.5 * (diff / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
        return np.clip(probs, 1e-300, None)

    def fit(self, x):
        T  = len(x)
        sx = np.sort(x)
        mid = len(sx) // 2
        # BUGFIX: see identical fix + full rationale in step9_real_data.py's
        # SimpleHMM.fit(). Short version: a bare "+1e-4" floor is negligible
        # against real data scale and lets the low-vol state's sigma collapse,
        # saturating P(Transition) near 1.0 permanently. Floor at 10% of the
        # series' own std instead.
        sigma_floor = max(1e-4, 0.1 * float(np.std(x)))
        self.mu    = np.array([np.mean(sx[:mid]), np.mean(sx[mid:])])
        self.sigma = np.maximum(
            np.array([np.std(sx[:mid]), np.std(sx[mid:])]), sigma_floor)
        self.pi    = np.array([0.7, 0.3])
        self.A     = np.array([[0.95, 0.05], [0.10, 0.90]])
        log_lik_prev = -np.inf
        for _ in range(self.n_iter):
            B = self._emission_prob(x)
            alpha = np.zeros((T, 2)); alpha[0] = self.pi * B[0]
            scale = np.zeros(T);     scale[0]  = alpha[0].sum()
            alpha[0] /= scale[0] + 1e-300
            for t in range(1, T):
                alpha[t] = (alpha[t-1] @ self.A) * B[t]
                scale[t] = alpha[t].sum(); alpha[t] /= scale[t] + 1e-300
            beta = np.ones((T, 2)); beta[-1] = 1.0
            for t in range(T-2, -1, -1):
                beta[t] = self.A @ (B[t+1] * beta[t+1])
                beta[t] /= beta[t].sum() + 1e-300
            gamma = alpha * beta; gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300
            xi = np.zeros((T-1, 2, 2))
            for t in range(T-1):
                xi[t] = np.outer(alpha[t], beta[t+1] * B[t+1]) * self.A
                xi[t] /= xi[t].sum() + 1e-300
            self.pi    = gamma[0]
            self.A     = xi.sum(0) / (xi.sum(0).sum(1, keepdims=True) + 1e-300)
            self.mu    = (gamma * x[:,None]).sum(0) / (gamma.sum(0) + 1e-300)
            self.sigma = np.maximum(
                np.sqrt((gamma * (x[:,None] - self.mu)**2).sum(0)
                       / (gamma.sum(0) + 1e-300)), sigma_floor)
            log_lik = np.sum(np.log(scale + 1e-300))
            if abs(log_lik - log_lik_prev) < self.tol: break
            log_lik_prev = log_lik
        self.fitted = True
        return self

    def predict_proba(self, x):
        T = len(x); B = self._emission_prob(x)
        alpha = np.zeros((T, 2)); alpha[0] = self.pi * B[0]; alpha[0] /= alpha[0].sum() + 1e-300
        for t in range(1, T):
            alpha[t] = (alpha[t-1] @ self.A) * B[t]; alpha[t] /= alpha[t].sum() + 1e-300
        beta = np.ones((T, 2)); beta[-1] = 1.0
        for t in range(T-2, -1, -1):
            beta[t] = self.A @ (B[t+1] * beta[t+1]); beta[t] /= beta[t].sum() + 1e-300
        g = alpha * beta; g /= g.sum(axis=1, keepdims=True) + 1e-300
        return g


# ══════════════════════════════════════════════════════════════════════════════
#  CONDITION 1 — BASELINE (all features, LightGBM)
# ══════════════════════════════════════════════════════════════════════════════

def make_model(kind, seed=42):
    """
    Classifier factory for the ablation.

    A module-level function rather than a lambda because joblib has to pickle it
    to reach the worker processes; functools.partial(make_model, "logreg") is
    picklable, a closure is not.

    "lgbm"   LightGBM — the project baseline. An EMBEDDED feature selector: it
             evaluates every feature at every split and simply never splits on
             useless ones, which is why an external wrapper has so little to add.
    "logreg" Logistic regression — no built-in selection, and sensitive to the
             collinearity that external selection removes. Measured on real
             data, the same 5-feature subset is worth -0.020 to LightGBM and
             +0.055 to LogReg. Running the ablation under both turns "selection
             does not help" into a statement about classifier capacity.
    """
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=200)
    return LGBMClassifier(n_estimators=100, num_leaves=31, learning_rate=0.1,
                          verbosity=-1, random_state=seed, n_jobs=1)


def run_baseline(X_tr, y_tr, X_te, y_te, feat_names, seed, model="lgbm"):
    """No feature selection — use all features, with the pass's classifier."""
    try:
        pipe = Pipeline([
            ("imp",    SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model",  make_model(model, seed))
        ])
        pipe.fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:,1])
    except Exception:
        auc = 0.5
    return {"auc": auc, "selected": list(feat_names), "n_sel": len(feat_names)}


# ══════════════════════════════════════════════════════════════════════════════
#  HMM TRIGGER HELPER — checks whether HMM fires for a given fold
# ══════════════════════════════════════════════════════════════════════════════

def get_hmm_trigger(X_train, feat_name="signal_0",
                    rolling_window=20,
                    percentile_k=85.0,
                    threshold_obj=None,
                    warmup_min_obs=150):
    """
    Fit HMM on training window, get P(Trans), check threshold.
    Returns (triggered: bool, p_trans: float, is_warmup: bool).

    FIX: .bfill() replaces deprecated fillna(method="bfill")

    BUGFIX (warm-start problem): see identical fix + full rationale in
    step9_real_data.py's get_hmm_trigger(). Short version: too little
    training data makes the HMM's state separation unreliable and can
    produce a falsely-confident p_trans, which then fires, consumes the
    cooldown, and can mask detection at the actual first real break. Below
    `warmup_min_obs` valid observations, p_trans is computed for diagnostics
    but the detector is not allowed to act on it.
    """
    obs = pd.Series(X_train[feat_name].values).rolling(rolling_window).std()
    obs = obs.bfill().values   # FIX: was fillna(method="bfill")
    if len(obs) < 30:
        return False, 0.0, True
    is_warmup = len(obs) < warmup_min_obs
    hmm = SimpleHMM(n_iter=30, tol=1e-3)
    try:
        hmm.fit(obs)
        gamma = hmm.predict_proba(obs)
        p_trans = float(gamma[-1, 1])
    except Exception:
        return False, 0.0, is_warmup
    if is_warmup:
        return False, p_trans, True   # diagnostic only -- not acted on
    triggered = threshold_obj.update(p_trans) if threshold_obj else (p_trans > 0.5)
    return triggered, p_trans, False


# ══════════════════════════════════════════════════════════════════════════════
#  MASTER ABLATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

# FIX: Removed duplicate "+APSOLL" key — "apsoll" is the key that matches
#      step7's JSON output.  The "+APSOLL" entry would be silently overwritten
#      in Python dicts anyway, but removing it prevents confusion.
CONDITIONS = {
    "baseline":         "Baseline (all features)",
    "standard_orpsoc":  "Standard OrPSOC",
    "apsoll":           "+APSOLL (no HMM)",
    "full_hybrid":      "Full Hybrid",
    "full_hybrid_noimp":"Full Hybrid (no imp-reinit)",
}

ALL_RESULTS = {}

t_global   = time.time()


def run_one_seed(level_key, seed, signal_r1, signal_r2, theta=None,
                 model="lgbm", criterion=None):
    """
    Run ONE complete walk-forward sequence for one seed of one level.

    This is the unit of parallelism and the unit of checkpointing. It is the
    body of the former `for seed in range(N_SEEDS)` loop, moved verbatim into a
    function — the fold loop inside it is unchanged and still runs fold
    1 -> 2 -> ... -> N in chronological order.

    Self-contained by design: the dataset is loaded and the folds are rebuilt
    here rather than passed in, so a worker process shares no mutable state
    with the parent or with any sibling worker. `walk_forward_folds` is
    deterministic, so the folds rebuilt here are identical to the parent's.

    Everything stateful lives INSIDE this function — `hmm_threshold`,
    `warm_start_fh`, `warm_start_fh_noimp` — which is exactly where it lived
    before (it was constructed at the top of the seed loop). Nothing crosses a
    seed boundary, which is why seeds may run in any order or concurrently.
    """
    with open(f"data/{level_key}.pkl", "rb") as f:
        data = pickle.load(f)
    X, y = data["X"], data["y"]
    feat_names = list(X.columns)
    folds = walk_forward_folds(X, y, n_splits=N_SPLITS, gap=5, min_train=150)
    # Label folds pre / straddle / post from the ACTUAL break index rather than
    # assuming it lands at N_SPLITS // 2 (which mislabels the straddling fold).
    fold_phase = classify_folds(folds, SWITCH_INDEX.get(level_key))
    # Does the ground-truth signal set ROTATE at the switch? Only then is the
    # union-of-all-signals denominator wrong (see _recall_active).
    _rotates = SWITCH_INDEX.get(level_key) is not None
    theta = THETA if theta is None else theta

    t_seed = time.time()
    seed_results = {cond: {"fold_aucs": [], "fold_selected": [],
                            "runtimes": []}
                    for cond in CONDITIONS}

    # Shared HMM threshold object per seed (stateful across folds)
    # k=85 reduces false alarms vs k=75 (see step5 sensitivity table)
    hmm_threshold = AdaptiveRegimeThreshold(method="percentile",
                                            lookback=50,
                                            percentile_k=85.0)

    # Warm-start position carried from fold k → fold k+1 (Full Hybrid only).
    # Reset to None at the start of each seed so seeds are independent.
    warm_start_fh = None
    # Separate warm-start chain for the no-importance-reinit ablation so the
    # two Full-Hybrid variants never contaminate each other's memory.
    warm_start_fh_noimp = None

    for fold_idx, (X_tr, y_tr, X_te, y_te, train_end) in enumerate(folds):
        if len(y_te.unique()) < 2:
            continue

        pso_kw = dict(
            feat_names=feat_names, seed=seed + fold_idx * 1000,
            n_particles=N_PARTICLES, max_iter=MAX_ITER,
            min_f=3, theta=theta,
            cr_low=0.3, cr_high=0.8, w_max=0.9, w_min=0.4,
            N_explore=max(5, MAX_ITER // 4), lam=0.1,
            apsoll_patience=APSOLL_PATIENCE,
            apsoll_rearm_after=APSOLL_REARM_AFTER,
            model_factory=(None if model == "lgbm"
                           else partial(make_model, model)),
            criterion=criterion,
        )

        # Ground-truth signal features for recall scoring.
        # signal_r1 (regime 1 signals) + signal_r2 (regime 2 signals)
        # are known because we generated the data; recall = fraction found.
        true_signals = set(signal_r1 + signal_r2)
        ph = fold_phase[fold_idx]
        # RETAINED so step8 and existing analysis keep working, but the
        # semantics CHANGED and downstream code must know it:
        #   old: fold_idx < N_SPLITS//2  -> the straddling fold counted as PRE
        #   new: phase == "pre"          -> the straddling fold counts as POST
        # Neither is correct; a straddle fold is neither. Pooling it into POST
        # is the less wrong of the two (its test window is 71% post-switch at
        # n_splits=8), but any honest analysis should use `fold_phase` and
        # report the straddle fold separately or drop it.
        is_pre_switch = (ph["phase"] == "pre")

        def _recall(selected):
            """
            UNION recall: fraction of ALL ground-truth signal features found.

            ⚠ ON L4 THIS METRIC REWARDS THE WRONG BEHAVIOUR. The signal set
            rotates at the switch: s0-s2 are predictive only before it and
            s3-s4 only after. Measured on the v2 generator, pre-switch window:
            AUC(s0-s2)=0.938 vs AUC(s3-s4)=0.514; post-switch: 0.545 vs 0.943.

            Union recall CAN still reach 1.0 -- a selector need only keep all
            five features, including the three or two that are currently dead
            weight (confirmed empirically: folds 5-8 score 1.00). But a selector
            behaving CORRECTLY, i.e. dropping the regime that has gone inert,
            scores at most 3/5 = 0.60 pre-switch and 2/5 = 0.40 post-switch.
            So the union denominator pays for carrying dead features and
            penalises correct adaptation -- the exact behaviour the study is
            trying to measure.

            Rotation is what a regime switch IS, so this is not a defect to fix
            in the generator (v1 and v2 both rotate, by design). It means the
            UNION denominator is the wrong denominator. Use
            `fold_recall_active` below.

            Retained unchanged so step8 and the existing tables keep working.
            """
            if not true_signals:
                return float("nan")
            return len(true_signals & set(selected)) / len(true_signals)

        def _recall_active(selected, phase):
            """
            Recall against the features that are ACTUALLY predictive in this
            fold's regime, so the attainable maximum is 1.0 everywhere.

            Levels with no switch (SWITCH_INDEX is None): every signal feature
            is live throughout, so this is identical to the union recall.
            Levels with a switch: score against signal_r1 before it and
            signal_r2 after it. A straddling fold has no single well-defined
            target set -- its test window spans both regimes -- so it returns
            NaN rather than a number that silently means two different things.
            """
            if not true_signals:
                return float("nan")
            if not _rotates:
                return len(true_signals & set(selected)) / len(true_signals)
            if phase == "pre" and signal_r1:
                return len(set(signal_r1) & set(selected)) / len(signal_r1)
            if phase == "post" and signal_r2:
                return len(set(signal_r2) & set(selected)) / len(signal_r2)
            return float("nan")

        def _r1_hits(selected):
            return len(set(signal_r1) & set(selected))

        def _r2_hits(selected):
            return len(set(signal_r2) & set(selected))

        # ── Condition 1: Baseline ──────────────────────────────────────
        t0  = time.time()
        r1  = run_baseline(X_tr, y_tr, X_te, y_te, feat_names,
                           seed + fold_idx * 1000, model=model)
        seed_results["baseline"]["fold_aucs"].append(r1["auc"])
        seed_results["baseline"]["fold_selected"].append(r1["selected"])
        seed_results["baseline"]["runtimes"].append(time.time() - t0)
        seed_results["baseline"].setdefault("fold_recall", []).append(
            _recall(r1["selected"]))
        seed_results["baseline"].setdefault("fold_r1_hits", []).append(
            _r1_hits(r1["selected"]))
        seed_results["baseline"].setdefault("fold_r2_hits", []).append(
            _r2_hits(r1["selected"]))
        seed_results["baseline"].setdefault("fold_is_pre", []).append(
            is_pre_switch)

        # ── Condition 2: Standard OrPSOC (from orpsoc_utils) ───────────
        r2 = run_standard_orpsoc(X_tr, y_tr, X_te, y_te, **pso_kw)
        seed_results["standard_orpsoc"]["fold_aucs"].append(r2["auc"])
        seed_results["standard_orpsoc"]["fold_selected"].append(r2["selected"])
        seed_results["standard_orpsoc"]["runtimes"].append(r2["runtime"])
        seed_results["standard_orpsoc"].setdefault("fold_recall", []).append(
            _recall(r2["selected"]))
        seed_results["standard_orpsoc"].setdefault("fold_r1_hits", []).append(
            _r1_hits(r2["selected"]))
        seed_results["standard_orpsoc"].setdefault("fold_r2_hits", []).append(
            _r2_hits(r2["selected"]))
        seed_results["standard_orpsoc"].setdefault("fold_is_pre", []).append(
            is_pre_switch)

        # ── Condition 3: +APSOLL, no HMM (from orpsoc_utils) ──────────
        r3 = run_hybrid_orpsoc(X_tr, y_tr, X_te, y_te,
                               hmm_trigger=False, **pso_kw)
        seed_results["apsoll"]["fold_aucs"].append(r3["auc"])
        seed_results["apsoll"]["fold_selected"].append(r3["selected"])
        seed_results["apsoll"]["runtimes"].append(r3["runtime"])
        seed_results["apsoll"].setdefault("fold_recall", []).append(
            _recall(r3["selected"]))
        seed_results["apsoll"].setdefault("fold_r1_hits", []).append(
            _r1_hits(r3["selected"]))
        seed_results["apsoll"].setdefault("fold_r2_hits", []).append(
            _r2_hits(r3["selected"]))
        seed_results["apsoll"].setdefault("fold_is_pre", []).append(
            is_pre_switch)

        # ── Condition 4: Full Hybrid (HMM trigger + warm-start) ────────
        triggered, p_trans, is_warmup = get_hmm_trigger(
            X_tr, feat_name=feat_names[0],
            threshold_obj=hmm_threshold
        )
        # Carry gbest from fold k forward on EVERY fold.
        #   • No regime change → particle[0] warm-started (continuation).
        #   • Regime change    → the runner uses the carried gbest only to
        #     seed a small set of ELITE particles (partial restart /
        #     population memory), retaining pre-switch knowledge without
        #     anchoring the whole swarm to the old regime.
        # p_trans scales the Phase 2 burst intensity (proportional drift
        # response) instead of a fixed cr_high/w_max on a binary trigger.
        r4 = run_hybrid_orpsoc(X_tr, y_tr, X_te, y_te,
                               hmm_trigger=triggered,
                               warm_start_pos=warm_start_fh,
                               p_trans=p_trans,
                               **pso_kw)
        warm_start_fh = r4["gbest_pos"]

        seed_results["full_hybrid"]["fold_aucs"].append(r4["auc"])
        seed_results["full_hybrid"]["fold_selected"].append(r4["selected"])
        seed_results["full_hybrid"]["runtimes"].append(r4["runtime"])
        seed_results["full_hybrid"].setdefault("fold_recall", []).append(
            _recall(r4["selected"]))
        seed_results["full_hybrid"].setdefault("fold_r1_hits", []).append(
            _r1_hits(r4["selected"]))
        seed_results["full_hybrid"].setdefault("fold_r2_hits", []).append(
            _r2_hits(r4["selected"]))
        seed_results["full_hybrid"].setdefault("fold_is_pre", []).append(
            is_pre_switch)
        # Log the FULL diagnostic picture, not just the gated boolean:
        # raw_fire = did the underlying signal cross the bar BEFORE
        # cooldown/warmup gating (None during warmup); is_warmup = was
        # this fold too early in the walk-forward sequence to trust the
        # detector at all. Without these, "triggered=False" is ambiguous
        # between "no signal" and "signal present but suppressed".
        raw_fire = (None if is_warmup else
                   bool(getattr(hmm_threshold, "last_raw_fire", False)))
        seed_results["full_hybrid"].setdefault("fold_triggered", []).append(
            bool(triggered))
        seed_results["full_hybrid"].setdefault("fold_p_trans", []).append(
            float(p_trans))
        seed_results["full_hybrid"].setdefault("fold_raw_fire", []).append(
            raw_fire)
        seed_results["full_hybrid"].setdefault("fold_is_warmup", []).append(
            bool(is_warmup))

        # ── Condition 5: Full Hybrid WITHOUT importance-guided reinit ──
        # Identical to Full Hybrid in every respect (same trigger, same
        # p_trans, same elite partial-restart) EXCEPT the fresh particles
        # are reinitialised blindly (orthogonal) rather than from recent-
        # window feature importances. The Full-Hybrid−vs−this delta is the
        # ISOLATED marginal contribution of importance-guided reinit
        # (professor suggestion #2). Own warm-start chain (see above).
        r5 = run_hybrid_orpsoc(X_tr, y_tr, X_te, y_te,
                               hmm_trigger=triggered,
                               warm_start_pos=warm_start_fh_noimp,
                               p_trans=p_trans,
                               use_importance_reinit=False,
                               **pso_kw)
        warm_start_fh_noimp = r5["gbest_pos"]

        seed_results["full_hybrid_noimp"]["fold_aucs"].append(r5["auc"])
        seed_results["full_hybrid_noimp"]["fold_selected"].append(r5["selected"])
        seed_results["full_hybrid_noimp"]["runtimes"].append(r5["runtime"])
        seed_results["full_hybrid_noimp"].setdefault("fold_recall", []).append(
            _recall(r5["selected"]))
        seed_results["full_hybrid_noimp"].setdefault("fold_r1_hits", []).append(
            _r1_hits(r5["selected"]))
        seed_results["full_hybrid_noimp"].setdefault("fold_r2_hits", []).append(
            _r2_hits(r5["selected"]))
        seed_results["full_hybrid_noimp"].setdefault("fold_is_pre", []).append(
            is_pre_switch)

        # ── Fold phase (pre / straddle / post) — condition-independent ─────
        # Recorded for EVERY condition so any downstream analysis can do the
        # 3-way split without re-deriving the geometry. `train_sees_post` says
        # whether adaptation was even possible on this fold.
        _sel_by_cond = {"baseline": r1, "standard_orpsoc": r2, "apsoll": r3,
                        "full_hybrid": r4, "full_hybrid_noimp": r5}
        for cond in CONDITIONS:
            seed_results[cond].setdefault("fold_recall_active", []).append(
                _recall_active(_sel_by_cond[cond]["selected"], ph["phase"]))
            seed_results[cond].setdefault("fold_phase", []).append(ph["phase"])
            seed_results[cond].setdefault("fold_train_sees_post", []).append(
                ph["train_sees_post"])
            seed_results[cond].setdefault("fold_frac_post_in_test", []).append(
                ph["frac_post_in_test"])

        # ── APSOLL trigger diagnostics (work-order 2.1.a) ──────────────────
        # Previously ONLY full_hybrid logged trigger diagnostics and the
        # `apsoll` condition logged none -- which is the entire reason the
        # stagnation trigger went unmeasured. Both are logged now.
        for cond, res in (("apsoll", r3), ("full_hybrid", r4),
                          ("full_hybrid_noimp", r5)):
            seed_results[cond].setdefault("fold_apsoll_trigger_iter", []).append(
                res.get("apsoll_trigger_iter"))
            seed_results[cond].setdefault("fold_apsoll_trigger_iters", []).append(
                res.get("apsoll_trigger_iters", []))
            seed_results[cond].setdefault("fold_apsoll_n_rearms", []).append(
                res.get("apsoll_n_rearms", 0))
            seed_results[cond].setdefault("fold_apsoll_max_c", []).append(
                res.get("apsoll_max_c"))

    # Jaccard stability for this seed.
    #
    # The pre/post split is now driven by the ACTUAL break, not the midpoint of
    # the fold-pair list. `fold_phase` already locates the straddle fold from
    # SWITCH_INDEX; pair i compares folds i and i+1, so the first pair across
    # which the subset can causally change is the straddle fold's own index.
    # Stationary levels have no straddle fold and pass None, which makes
    # feature_stability_ratio label its output as not regime-aware instead of
    # reporting a meaningless "adaptation drop" on a level with no regime.
    _straddle = [i for i, p in enumerate(fold_phase) if p["phase"] == "straddle"]
    _switch_pair = _straddle[0] if _straddle else None
    for cond in CONDITIONS:
        seed_results[cond]["jaccard"] = feature_stability_ratio(
            seed_results[cond]["fold_selected"], switch_pair=_switch_pair)

    auc_str = "  ".join(
        f"{cond[:3]}={np.mean(seed_results[cond]['fold_aucs']):.3f}"
        for cond in CONDITIONS)
    print(f"  [{level_key}] seed={seed + 1:2d}/{N_SEEDS}  {auc_str}  "
          f"({time.time() - t_seed:.0f}s)", flush=True)
    return seed_results


# ══════════════════════════════════════════════════════════════════════════════
#  DRIVER — levels sequentially, seeds in parallel, one checkpoint per seed
# ══════════════════════════════════════════════════════════════════════════════
#  Levels stay sequential: they are already a hard reset (each rebinds X, y,
#  folds and level_results from its own .pkl), so there is nothing to gain and
#  it keeps the console log readable. Seeds are dispatched to N_JOBS worker
#  processes; each worker runs its whole fold sequence in order.
# ══════════════════════════════════════════════════════════════════════════════

STORE = CheckpointStore("results/checkpoints", "step7", PROV,
                        enabled=USE_CHECKPOINTS)


def _seed_or_checkpoint(level_key, seed, signal_r1, signal_r2, theta,
                        model="lgbm", criterion=None):
    """Return a cached seed result if one exists, else compute and cache it."""
    unit = (f"{level_key}_seed{seed:03d}_th{theta:.2f}_{model}"
            f"_{criterion or 'default'}")
    cached = STORE.load(unit)
    if cached is not None:
        return cached
    result = run_one_seed(level_key, seed, signal_r1, signal_r2, theta=theta,
                          model=model, criterion=criterion)
    STORE.save(unit, result)
    return result


# THETA_SWEEP=None -> a single run at THETA (the normal case). Otherwise the
# whole ablation runs once per theta and every result is keyed by it, so a sweep
# never overwrites the primary run.
THETAS = [THETA] if not THETA_SWEEP else list(THETA_SWEEP)

for theta in THETAS:
  for level_key, level_name in LEVELS.items():
    tag = "" if not THETA_SWEEP else f"  [theta={theta}]"
    print(f"\n{'─' * 65}")
    print(f"  {level_name}{tag}")
    print(f"{'─' * 65}", flush=True)

    # Signal-feature names for recall scoring. Read from the level's own
    # dataset; the workers reload the data themselves.
    with open(f"data/{level_key}.pkl", "rb") as f:
        _cols = list(pickle.load(f)["X"].columns)
    signal_all = [c for c in _cols if c.startswith("signal_")]
    signal_r1  = [c for c in signal_all if c in ["signal_0", "signal_1", "signal_2"]]
    signal_r2  = [c for c in signal_all if c in ["signal_3", "signal_4"]]

    units = [f"{level_key}_seed{s:03d}_th{theta:.2f}" for s in range(N_SEEDS)]
    if USE_CHECKPOINTS:
        print(f"  checkpoints: {STORE.summary(units)}", flush=True)

    # joblib returns results in SUBMISSION order, so level_results stays in
    # seed order exactly as the old sequential append did.
    seed_results_list = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_seed_or_checkpoint)(level_key, seed, signal_r1, signal_r2, theta)
        for seed in range(N_SEEDS)
    )

    level_results = {cond: [sr[cond] for sr in seed_results_list]
                     for cond in CONDITIONS}

    elapsed = time.time() - t_global
    print(f"  {level_name}{tag} done — elapsed {elapsed / 60:.1f} min", flush=True)

    if THETA_SWEEP:
        ALL_RESULTS[f"{level_key}@theta{theta:.2f}"] = {
            "level_name": f"{level_name} (theta={theta})",
            "signal_r1": signal_r1, "signal_r2": signal_r2,
            "theta": theta, "conditions": level_results,
        }
        continue

    ALL_RESULTS[level_key] = {
        "level_name": level_name,
        "signal_r1":  signal_r1,
        "signal_r2":  signal_r2,
        "conditions": level_results,
    }

# ══════════════════════════════════════════════════════════════════════════════
#  SECONDARY PASS — the same ablation under a low-capacity classifier
# ══════════════════════════════════════════════════════════════════════════════
#  Runs only AFTER the primary LightGBM pass has completed, on the levels named
#  in LOGREG_PASS_LEVELS. Identical folds, identical seeds, identical
#  conditions -- the ONLY thing that changes is the classifier used for both the
#  PSO fitness and the final scoring, so any difference is attributable to
#  classifier capacity rather than to the protocol.
# ══════════════════════════════════════════════════════════════════════════════

for level_key in LOGREG_PASS_LEVELS:
    if level_key not in LEVELS:
        print(f"\n  [logreg pass] skipping {level_key}: not in LEVELS", flush=True)
        continue
    label = f"{LEVELS[level_key]} · LogReg/{LOGREG_PASS_CRITERION or 'default'}"
    print(f"\n{'─' * 65}")
    print(f"  {label}")
    print(f"{'─' * 65}", flush=True)

    with open(f"data/{level_key}.pkl", "rb") as f:
        _cols = list(pickle.load(f)["X"].columns)
    signal_all = [c for c in _cols if c.startswith("signal_")]
    signal_r1  = [c for c in signal_all if c in ["signal_0", "signal_1", "signal_2"]]
    signal_r2  = [c for c in signal_all if c in ["signal_3", "signal_4"]]

    units = [f"{level_key}_seed{s:03d}_th{THETA:.2f}_logreg"
             f"_{LOGREG_PASS_CRITERION or 'default'}" for s in range(N_SEEDS)]
    if USE_CHECKPOINTS:
        print(f"  checkpoints: {STORE.summary(units)}", flush=True)

    seed_results_list = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_seed_or_checkpoint)(level_key, seed, signal_r1, signal_r2,
                                     THETA, "logreg", LOGREG_PASS_CRITERION)
        for seed in range(N_SEEDS)
    )
    ALL_RESULTS[f"{level_key}@logreg"] = {
        "level_name": label,
        "signal_r1":  signal_r1,
        "signal_r2":  signal_r2,
        "classifier": "logreg",
        "criterion":  LOGREG_PASS_CRITERION or "default",
        "conditions": {cond: [sr[cond] for sr in seed_results_list]
                       for cond in CONDITIONS},
    }
    print(f"  {label} done — elapsed {(time.time() - t_global) / 60:.1f} min",
          flush=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARISE AND SAVE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  SUMMARY TABLE  (mean AUC ± std across seeds)")
print("=" * 65)
header = f"{'Level':<28}" + "".join(f"{'Cond'+str(i+1):>14}" for i in range(len(CONDITIONS)))
print(header)

summary = {}
for level_key, ldata in ALL_RESULTS.items():
    row = f"{ldata['level_name'][:27]:<28}"
    summary[level_key] = {}
    for cond in CONDITIONS:
        seed_aucs = [np.mean(sr["fold_aucs"]) for sr in ldata["conditions"][cond]]
        m, s = np.mean(seed_aucs), np.std(seed_aucs)
        row += f"{m:.3f}±{s:.3f}  "

        # Overall recall (nan-safe)
        seed_recalls = []
        for sr in ldata["conditions"][cond]:
            recalls = [v for v in sr.get("fold_recall", [])
                       if not (isinstance(v, float) and np.isnan(v))]
            if recalls:
                seed_recalls.append(float(np.mean(recalls)))
        mean_recall = float(np.mean(seed_recalls)) if seed_recalls else float("nan")

        # Pre/post switch recall split (Level 4 diagnostic)
        pre_recalls, post_recalls = [], []
        for sr in ldata["conditions"][cond]:
            is_pre = sr.get("fold_is_pre", [])
            recalls = sr.get("fold_recall", [])
            pre  = [r for r, p in zip(recalls, is_pre)
                    if p and not (isinstance(r, float) and np.isnan(r))]
            post = [r for r, p in zip(recalls, is_pre)
                    if not p and not (isinstance(r, float) and np.isnan(r))]
            if pre:  pre_recalls.append(float(np.mean(pre)))
            if post: post_recalls.append(float(np.mean(post)))
        mean_recall_pre  = float(np.mean(pre_recalls))  if pre_recalls  else float("nan")
        mean_recall_post = float(np.mean(post_recalls)) if post_recalls else float("nan")

        # Per-fold r1/r2 hit counts averaged across seeds
        fold_r1 = np.array([sr.get("fold_r1_hits", []) for sr in ldata["conditions"][cond]])
        fold_r2 = np.array([sr.get("fold_r2_hits", []) for sr in ldata["conditions"][cond]])
        mean_r1_per_fold = fold_r1.mean(axis=0).tolist() if fold_r1.size > 0 else []
        mean_r2_per_fold = fold_r2.mean(axis=0).tolist() if fold_r2.size > 0 else []

        summary[level_key][cond] = {
            "mean": float(m), "std": float(s),
            "seed_aucs":        [float(v) for v in seed_aucs],
            "mean_recall":      mean_recall,
            "mean_recall_pre":  mean_recall_pre,
            "mean_recall_post": mean_recall_post,
            "mean_r1_hits_per_fold": mean_r1_per_fold,
            "mean_r2_hits_per_fold": mean_r2_per_fold,
        }
    print(row)

print()
print("  FEATURE RECALL  (fraction of true signal features recovered)")
print(f"{'Level':<28}" + "".join(f"{'Cond'+str(i+1):>14}" for i in range(len(CONDITIONS))))
for level_key, ldata in ALL_RESULTS.items():
    row = f"{ldata['level_name'][:27]:<28}"
    for cond in CONDITIONS:
        mr = summary[level_key][cond].get("mean_recall", float("nan"))
        row += f"{'nan' if np.isnan(mr) else f'{mr:.3f}':>14}"
    print(row)

print()
print("  PRE/POST SWITCH RECALL  (Level 4 — Regime Switch only)")
print(f"{'Condition':<22}  {'Pre-switch':>12}  {'Post-switch':>12}")
if "regime_switch" in summary:
    for cond, label in CONDITIONS.items():
        pre  = summary["regime_switch"][cond].get("mean_recall_pre",  float("nan"))
        post = summary["regime_switch"][cond].get("mean_recall_post", float("nan"))
        def _fmt(v): return "nan" if (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"
        print(f"  {label[:20]:<20}  {_fmt(pre):>12}  {_fmt(post):>12}")

# ══════════════════════════════════════════════════════════════════════════════
#  IMPORTANCE-REINIT ABLATION  (Full Hybrid  vs  Full Hybrid without imp-reinit)
# ══════════════════════════════════════════════════════════════════════════════
# This is the ISOLATED contribution of professor suggestion #2. Positive delta
# on regime_switch (especially post-switch) = importance-guided reinit helps.

print()
print("=" * 65)
print("  IMPORTANCE-REINIT ABLATION  (Full Hybrid − no-imp variant)")
print("=" * 65)
print(f"  {'Level':<20}  {'FullHybrid':>11}  {'NoImpReinit':>11}  "
      f"{'Δ AUC':>8}  {'Wilcoxon p':>11}")
importance_ablation = {}
for level_key in ALL_RESULTS:
    imp   = np.array(summary[level_key]["full_hybrid"]["seed_aucs"])
    noimp = np.array(summary[level_key]["full_hybrid_noimp"]["seed_aucs"])
    delta = float(imp.mean() - noimp.mean())
    # Paired one-sided Wilcoxon: does imp-reinit BEAT no-imp across seeds?
    p_val = float("nan")
    try:
        if len(imp) == len(noimp) and np.any(imp - noimp != 0):
            _, p_val = wilcoxon(imp, noimp, alternative="greater")
    except Exception:
        pass
    importance_ablation[level_key] = {
        "full_hybrid_mean":       float(imp.mean()),
        "full_hybrid_noimp_mean": float(noimp.mean()),
        "delta_auc":              delta,
        "wilcoxon_p_greater":     p_val,
    }
    p_str = "nan" if np.isnan(p_val) else f"{p_val:.4f}"
    _nm = ALL_RESULTS[level_key]["level_name"]
    print(f"  {_nm[:19]:<20}  {imp.mean():>11.4f}  "
          f"{noimp.mean():>11.4f}  {delta:>+8.4f}  {p_str:>11}")

# Post-switch-only delta on regime_switch (where imp-reinit is designed to act)
if "regime_switch" in ALL_RESULTS:
    def _post_switch_mean(cond):
        vals = []
        for sr in ALL_RESULTS["regime_switch"]["conditions"][cond]:
            aucs   = sr["fold_aucs"]
            is_pre = sr.get("fold_is_pre", [True] * len(aucs))
            post   = [a for a, p in zip(aucs, is_pre) if not p]
            if post:
                vals.append(np.mean(post))
        return float(np.mean(vals)) if vals else float("nan")
    ps_imp   = _post_switch_mean("full_hybrid")
    ps_noimp = _post_switch_mean("full_hybrid_noimp")
    print()
    print(f"  Post-switch folds only (regime_switch):")
    print(f"    Full Hybrid           : {ps_imp:.4f}")
    print(f"    Full Hybrid (no imp)  : {ps_noimp:.4f}")
    print(f"    Δ (imp-reinit effect) : {ps_imp - ps_noimp:+.4f}")
    importance_ablation["regime_switch_postswitch"] = {
        "full_hybrid": ps_imp, "full_hybrid_noimp": ps_noimp,
        "delta": float(ps_imp - ps_noimp),
    }

# ── HMM trigger readout: did the detector fire, and where? ────────────────────
print()
print("=" * 65)
print("  HMM TRIGGER READOUT  (fold-by-fold; p_trans/raw_fire/warmup are")
print("  seed-invariant -- pure function of the data -- so seed 0 is shown)")
print("=" * 65)
print("  A trigger at/just-after the switch fold = detector working.")
trigger_log = {}
for level_key in ALL_RESULTS:
    fh_seeds = ALL_RESULTS[level_key]["conditions"]["full_hybrid"]
    trig = np.array([sr.get("fold_triggered", []) for sr in fh_seeds
                     if sr.get("fold_triggered")], dtype=float)
    if trig.size == 0:
        continue
    fire_rate = trig.mean(axis=0).tolist()   # per-fold fraction of seeds firing
    s0        = fh_seeds[0]
    p_arr     = s0.get("fold_p_trans", [])
    raw_arr   = s0.get("fold_raw_fire", [None] * len(p_arr))
    warm_arr  = s0.get("fold_is_warmup", [False] * len(p_arr))
    trigger_log[level_key] = {"fire_rate_per_fold": fire_rate,
                              "p_trans_per_fold": p_arr,
                              "raw_fire_per_fold": raw_arr,
                              "is_warmup_per_fold": warm_arr}
    sw = N_SPLITS // 2
    marks = "".join("^" if abs(i - sw) <= 1 else " " for i in range(len(fire_rate)))
    def _fmt_raw(v, w):
        if w: return "  W"
        return "  1" if v else "  0"
    print(f"  {ALL_RESULTS[level_key]['level_name']}")
    print(f"    fold        : " + " ".join(f"{i+1:>4}" for i in range(len(fire_rate))))
    print(f"    p_trans     : " + " ".join(f"{v:>4.2f}" for v in p_arr))
    print(f"    raw_fire    : " + " ".join(f"{_fmt_raw(v,w):>4}" for v, w in zip(raw_arr, warm_arr)))
    print(f"    final trig  : " + " ".join(f"{v:>4.2f}" for v in fire_rate))
    print(f"    near-switch : {'    '.join('')}{marks}   (^ = fold within 1 of switch)")
    print(f"    (W = warm-up, not enough training data yet -- not evaluated, "
          f"not the same as 'no signal')")


# FIX: N_SPLITS saved into config so step8 can read it as cfg["n_splits"]
# `provenance` records the config+source hash this run was produced under, so
# two result files can be compared for comparability without guesswork (G3).
save = {"config": dict(CONFIG),
        "provenance": PROV,
        "summary": summary,
        "importance_ablation": importance_ablation,
        "trigger_log": trigger_log}


# Deep-serialise ALL_RESULTS (convert numpy floats)
def to_json(obj):
    if isinstance(obj, (np.floating, np.integer)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict):  return {k: to_json(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [to_json(v) for v in obj]
    return obj

save["full_results"] = to_json(ALL_RESULTS)

with open(RESULTS_PATH, "w") as f:
    json.dump(save, f, indent=2)
print(f"\nSaved: {RESULTS_PATH}")
print(f"Total time: {(time.time()-t_global)/60:.1f} min")
print("\nNext: run step8_results.py")