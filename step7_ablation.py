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

import numpy as np
import pandas as pd
import pickle
import json
import time
import warnings
warnings.filterwarnings("ignore")
import os
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
    AdaptiveRegimeThreshold,
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

LEVELS = {
    "white_noise":   "Level 1 — White Noise",
    "ar1":           "Level 2 — AR(1) Stationary",
    "drift":         "Level 3 — Drift",
    "regime_switch": "Level 4 — Regime Switch",
}

print("=" * 65)
print("  STEP 7: Ablation Study")
print(f"  Mode: {'FAST (debug)' if FAST_MODE else 'FULL (paper)'}")
print(f"  Seeds={N_SEEDS}  MaxIter={MAX_ITER}  Particles={N_PARTICLES}")
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
            probs[:, k] = (np.exp(-0.5 * (diff / (self.sigma[k] + eps)) ** 2)
                           / (self.sigma[k] + eps + np.sqrt(2 * np.pi)))
        return np.clip(probs, 1e-300, None)

    def fit(self, x):
        T  = len(x)
        sx = np.sort(x)
        mid = len(sx) // 2
        self.mu    = np.array([np.mean(sx[:mid]), np.mean(sx[mid:])])
        self.sigma = np.array([np.std(sx[:mid]) + 1e-4, np.std(sx[mid:]) + 1e-4])
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
            self.sigma = np.sqrt((gamma * (x[:,None] - self.mu)**2).sum(0)
                                 / (gamma.sum(0) + 1e-300)) + 1e-4
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

def run_baseline(X_tr, y_tr, X_te, y_te, feat_names, seed):
    """No feature selection — use all features. LightGBM consistent with PSO conditions."""
    try:
        pipe = Pipeline([
            ("imp",    SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model",  LGBMClassifier(n_estimators=100, num_leaves=31,
                                      learning_rate=0.1, verbosity=-1,
                                      random_state=seed))
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
                    threshold_obj=None):
    """
    Fit HMM on training window, get P(Trans), check threshold.
    Returns (triggered: bool, p_trans: float).

    FIX: .bfill() replaces deprecated fillna(method="bfill")
    """
    obs = pd.Series(X_train[feat_name].values).rolling(rolling_window).std()
    obs = obs.bfill().values   # FIX: was fillna(method="bfill")
    if len(obs) < 30:
        return False, 0.0
    hmm = SimpleHMM(n_iter=30, tol=1e-3)
    try:
        hmm.fit(obs)
        gamma = hmm.predict_proba(obs)
        p_trans = float(gamma[-1, 1])
    except Exception:
        return False, 0.0
    triggered = threshold_obj.update(p_trans) if threshold_obj else (p_trans > 0.5)
    return triggered, p_trans


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

total_runs = len(LEVELS) * len(CONDITIONS) * N_SEEDS
run_count  = 0
t_global   = time.time()

for level_key, level_name in LEVELS.items():
    print(f"\n{'─'*65}")
    print(f"  {level_name}")
    print(f"{'─'*65}")

    # Load dataset
    with open(f"data/{level_key}.pkl", "rb") as f:
        data = pickle.load(f)
    X, y   = data["X"], data["y"]
    feat_names = list(X.columns)

    # Check if signal cols exist for recall scoring
    signal_all = [c for c in feat_names if c.startswith("signal_")]
    signal_r1  = [c for c in signal_all if c in ["signal_0","signal_1","signal_2"]]
    signal_r2  = [c for c in signal_all if c in ["signal_3","signal_4"]]

    folds = walk_forward_folds(X, y, n_splits=N_SPLITS, gap=5, min_train=150)

    level_results = {cond: [] for cond in CONDITIONS}

    switch_fold = N_SPLITS // 2   # folds < switch_fold are pre-switch

    for seed in range(N_SEEDS):
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
                min_f=3, theta=0.7,
                cr_low=0.3, cr_high=0.8, w_max=0.9, w_min=0.4,
                N_explore=max(5, MAX_ITER // 4), lam=0.1,
            )

            # Ground-truth signal features for recall scoring.
            # signal_r1 (regime 1 signals) + signal_r2 (regime 2 signals)
            # are known because we generated the data; recall = fraction found.
            true_signals = set(signal_r1 + signal_r2)
            is_pre_switch = (fold_idx < switch_fold)

            def _recall(selected):
                if not true_signals:
                    return float("nan")
                return len(true_signals & set(selected)) / len(true_signals)

            def _r1_hits(selected):
                return len(set(signal_r1) & set(selected))

            def _r2_hits(selected):
                return len(set(signal_r2) & set(selected))

            # ── Condition 1: Baseline ──────────────────────────────────────
            t0  = time.time()
            r1  = run_baseline(X_tr, y_tr, X_te, y_te, feat_names,
                               seed + fold_idx * 1000)
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
            triggered, p_trans = get_hmm_trigger(
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
            # Log the HMM trigger + P(Transition) per fold so we can verify the
            # detector actually fires at the regime boundary (fold ≈ switch_fold).
            seed_results["full_hybrid"].setdefault("fold_triggered", []).append(
                bool(triggered))
            seed_results["full_hybrid"].setdefault("fold_p_trans", []).append(
                float(p_trans))

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

        # Compute Jaccard stability for this seed
        for cond in CONDITIONS:
            jac = feature_stability_ratio(seed_results[cond]["fold_selected"])
            seed_results[cond]["jaccard"] = jac
            level_results[cond].append(seed_results[cond])

        run_count += len(CONDITIONS)
        elapsed = time.time() - t_global
        rate    = run_count / max(elapsed, 1)
        eta     = (total_runs - run_count) / max(rate, 1e-6)

        # Mean AUC for last seed across conditions
        auc_str = "  ".join(
            f"{cond[:3]}={np.mean(seed_results[cond]['fold_aucs']):.3f}"
            for cond in CONDITIONS
        )
        print(f"  seed={seed+1:2d}/{N_SEEDS}  {auc_str}  "
              f"ETA={eta/60:.1f}min")

    ALL_RESULTS[level_key] = {
        "level_name": level_name,
        "signal_r1":  signal_r1,
        "signal_r2":  signal_r2,
        "conditions": level_results,
    }

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
    print(f"  {LEVELS[level_key][:19]:<20}  {imp.mean():>11.4f}  "
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
print("  HMM TRIGGER READOUT  (fold-by-fold, averaged across seeds)")
print("=" * 65)
print("  A trigger at/just-after the switch fold = detector working.")
trigger_log = {}
for level_key in ALL_RESULTS:
    fh_seeds = ALL_RESULTS[level_key]["conditions"]["full_hybrid"]
    trig = np.array([sr.get("fold_triggered", []) for sr in fh_seeds
                     if sr.get("fold_triggered")], dtype=float)
    ptr  = np.array([sr.get("fold_p_trans", []) for sr in fh_seeds
                     if sr.get("fold_p_trans")], dtype=float)
    if trig.size == 0:
        continue
    fire_rate = trig.mean(axis=0).tolist()   # per-fold fraction of seeds firing
    p_mean    = ptr.mean(axis=0).tolist()
    trigger_log[level_key] = {"fire_rate_per_fold": fire_rate,
                              "p_trans_per_fold": p_mean}
    sw = N_SPLITS // 2
    marks = "".join("^" if abs(i - sw) <= 1 else " " for i in range(len(fire_rate)))
    print(f"  {LEVELS[level_key]}")
    print(f"    fold        : " + " ".join(f"{i+1:>4}" for i in range(len(fire_rate))))
    print(f"    fire-rate   : " + " ".join(f"{v:>4.2f}" for v in fire_rate))
    print(f"    P(trans)    : " + " ".join(f"{v:>4.2f}" for v in p_mean))
    print(f"    near-switch : {'    '.join('')}{marks}   (^ = fold within 1 of switch)")


# FIX: N_SPLITS saved into config so step8 can read it as cfg["n_splits"]
save = {"config": {"fast_mode": FAST_MODE, "n_seeds": N_SEEDS,
                    "max_iter": MAX_ITER, "n_particles": N_PARTICLES,
                    "n_splits": N_SPLITS},
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

with open("results/step7_ablation.json", "w") as f:
    json.dump(save, f, indent=2)
print("\nSaved: results/step7_ablation.json")
print(f"Total time: {(time.time()-t_global)/60:.1f} min")
print("\nNext: run step8_results.py")