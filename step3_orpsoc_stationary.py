"""
STEP 3 — Standard OrPSOC on Stationary Data
=============================================
First time you run OrPSOC. Level 2 (AR1) only.

WHY STATIONARY FIRST?
──────────────────────
If OrPSOC cannot find the 5 known signal features on SIMPLE,
STABLE data — there is no point trying it on non-stationary data.
Validate the foundation before building on it.

WHAT SUCCESS LOOKS LIKE HERE
──────────────────────────────
  - Recall >= 0.6  (finds at least 3 of 5 signal features)
  - Fitness increases monotonically over iterations
  - Diversity starts HIGH (orthogonal init), decreases (convergence)
  - Test AUC of selected subset >= test AUC of all 50 features

FIX LOG (vs original):
  - Removed local duplicate definitions of sigmoid, crossover etc.
    → now imported from orpsoc_utils (single source of truth)
  - evaluate() now uses APSOLL-normalised fitness (theta=0.7)
    consistent with steps 6/7 (was flat-penalty before)

Run with:
    python step3_orpsoc_stationary.py
"""

import numpy as np
import pandas as pd
import pickle
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")
import os
import json
os.makedirs("plots",   exist_ok=True)
os.makedirs("results", exist_ok=True)

# ── All PSO primitives come from the shared utility module ────────────────────
from orpsoc_utils import (
    sigmoid, build_orthogonal_positions, crossover,
    hamming_diversity, evaluate,
)


# ══════════════════════════════════════════════════════════════════════════════
#  ORPSOC — Standard implementation (local runner for step3 standalone use)
# ══════════════════════════════════════════════════════════════════════════════

def run_orpsoc(
    X_train, y_train, X_val, y_val,
    feat_names,
    n_particles = 20,
    max_iter    = 40,
    min_f       = 5,
    w_start     = 0.9,
    w_end       = 0.4,
    c1          = 2.0,
    c2          = 2.0,
    cr          = 0.6,
    theta       = 0.7,
    seed        = 42,
    verbose     = True,
):
    """
    Standard OrPSOC with FIXED crossover rate and APSOLL-normalised fitness.

    PARAMETER GUIDE:
    ─────────────────
    w_start=0.9: start with high inertia (explore widely)
    w_end=0.4:   end with low inertia (fine-tune around good solutions)
    c1=c2=2.0:   balanced pull toward personal best and global best
    cr=0.6:      60% of particle pairs undergo crossover each iteration
    theta=0.7:   APSOLL fitness weight (0.7 AUC + 0.3 compactness)
    """
    rng = np.random.RandomState(seed)
    n   = len(feat_names)
    t0  = time.time()

    # ── Initialize with orthogonal positions ──────────────────────────────────
    init_pos  = build_orthogonal_positions(n_particles, n, seed)
    particles = []
    for i in range(n_particles):
        pos   = init_pos[i].copy()
        n_sel = int(pos.sum())
        if n_sel < min_f:
            zeros = np.where(pos == 0)[0]
            pos[rng.choice(zeros, size=min_f - n_sel, replace=False)] = 1
        particles.append({
            "pos":      pos,
            "vel":      rng.uniform(-3, 3, size=n),
            "best_pos": pos.copy(),
            "best_fit": -np.inf,
        })

    # ── Evaluate initial positions ────────────────────────────────────────────
    gbest_pos = None
    gbest_fit = -np.inf

    for p in particles:
        fit = evaluate(p["pos"], feat_names, X_train, y_train,
                       X_val, y_val, min_f, theta)
        p["best_fit"] = fit
        if fit > gbest_fit:
            gbest_fit = fit
            gbest_pos = p["pos"].copy()

    fit_hist = [gbest_fit]
    div_hist = [hamming_diversity(particles)]

    if verbose:
        print(f"  {'Iter':>5}  {'Best Fitness':>13}  {'Diversity':>10}  {'N_selected':>11}")
        print(f"  {'─' * 46}")

    # ── Main loop ─────────────────────────────────────────────────────────────
    for it in range(max_iter):

        w = w_start - (w_start - w_end) * (it / max(max_iter - 1, 1))

        # ── Velocity + position update ────────────────────────────────────────
        for p in particles:
            r1 = rng.rand(n)
            r2 = rng.rand(n)
            vel = (w  * p["vel"]
                   + c1 * r1 * (p["best_pos"] - p["pos"])
                   + c2 * r2 * (gbest_pos     - p["pos"]))
            vel = np.clip(vel, -6.0, 6.0)
            p["vel"] = vel
            new_pos = (rng.rand(n) < sigmoid(vel)).astype(float)
            n_sel = int(new_pos.sum())
            if n_sel < min_f:
                zeros = np.where(new_pos == 0)[0]
                need  = min_f - n_sel
                if len(zeros) >= need:
                    new_pos[rng.choice(zeros, size=need, replace=False)] = 1
            p["pos"] = new_pos

        # ── Crossover phase ───────────────────────────────────────────────────
        idx_list = list(range(n_particles))
        rng.shuffle(idx_list)
        for k in range(0, n_particles - 1, 2):
            pa = particles[idx_list[k]]
            pb = particles[idx_list[k + 1]]
            ca, cb = crossover(pa["pos"], pb["pos"], cr, min_f, rng)
            for child, parent in [(ca, pa), (cb, pb)]:
                fit = evaluate(child, feat_names, X_train, y_train,
                               X_val, y_val, min_f, theta)
                if fit > parent["best_fit"]:
                    parent["pos"]      = child
                    parent["best_pos"] = child.copy()
                    parent["best_fit"] = fit

        # ── Evaluate + update bests ───────────────────────────────────────────
        for p in particles:
            fit = evaluate(p["pos"], feat_names, X_train, y_train,
                           X_val, y_val, min_f, theta)
            if fit > p["best_fit"]:
                p["best_fit"] = fit
                p["best_pos"] = p["pos"].copy()
            if fit > gbest_fit:
                gbest_fit = fit
                gbest_pos = p["pos"].copy()

        fit_hist.append(gbest_fit)
        div_hist.append(hamming_diversity(particles))

        if verbose and (it % 5 == 0 or it == max_iter - 1):
            print(f"  {it+1:>5}  {gbest_fit:>13.4f}  "
                  f"{div_hist[-1]:>10.3f}  {int(gbest_pos.sum()):>11}")

    elapsed  = time.time() - t0
    sel_idx  = np.where(gbest_pos == 1)[0]
    sel_cols = [feat_names[i] for i in sel_idx]

    return {
        "selected":     sel_cols,
        "fit_hist":     fit_hist,
        "div_hist":     div_hist,
        "best_fitness": gbest_fit,
        "runtime":      elapsed,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — Run on Level 2 (stationary AR1) only
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 55)
print("  STEP 3: Standard OrPSOC on Level 2 (Stationary AR1)")
print("=" * 55)
print()
print("Ground truth: signal_0 ... signal_4 are predictive.")
print("Everything else (noise_*) is pure random.")
print()

with open("data/ar1.pkl", "rb") as f:
    data = pickle.load(f)
X, y = data["X"], data["y"]
feat_names = list(X.columns)

# Temporal split — NEVER shuffle
n           = len(X)
split_train = int(n * 0.60)
split_val   = int(n * 0.80)

X_train = X.iloc[:split_train]
y_train = y.iloc[:split_train]
X_val   = X.iloc[split_train:split_val]
y_val   = y.iloc[split_train:split_val]
X_test  = X.iloc[split_val:]
y_test  = y.iloc[split_val:]

print(f"Train: {len(X_train)} samples  Val: {len(X_val)}  Test: {len(X_test)}")
print()

print("Running OrPSOC...")
result = run_orpsoc(
    X_train, y_train, X_val, y_val,
    feat_names, n_particles=20, max_iter=40,
    min_f=5, theta=0.7, verbose=True,
)
print()


# ── Analyse what was selected ─────────────────────────────────────────────────
selected    = result["selected"]
signal_cols = [c for c in feat_names if c.startswith("signal_")]
noise_cols  = [c for c in feat_names if c.startswith("noise_")]

tp = [f for f in selected if f in signal_cols]
fp = [f for f in selected if f in noise_cols]
fn = [f for f in signal_cols if f not in selected]

precision = len(tp) / len(selected) if selected else 0
recall    = len(tp) / len(signal_cols)
f1        = (2 * precision * recall / (precision + recall)
             if (precision + recall) > 0 else 0)

print("=" * 55)
print("  RESULTS")
print("=" * 55)
print()
print(f"  Total selected    : {len(selected)}")
print(f"  True positives    : {len(tp)} / 5  {tp}")
print(f"  False positives   : {len(fp)}      {fp[:3]}{'...' if len(fp)>3 else ''}")
print(f"  Missed signals    : {len(fn)}      {fn}")
print()
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1        : {f1:.4f}")
print()


def quick_auc(cols, X_tr, y_tr, X_te, y_te):
    pipe = Pipeline([
        ("imp",    SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model",  XGBClassifier(
            n_estimators=80, max_depth=4,
            learning_rate=0.1, verbosity=0, random_state=42
        ))
    ])
    pipe.fit(X_tr[cols], y_tr)
    return float(roc_auc_score(y_te, pipe.predict_proba(X_te[cols])[:, 1]))


auc_all      = quick_auc(feat_names,  X_train, y_train, X_test, y_test)
auc_selected = quick_auc(selected,    X_train, y_train, X_test, y_test)
auc_oracle   = quick_auc(signal_cols, X_train, y_train, X_test, y_test)

print("  Test AUC Comparison:")
print(f"    All 50 features (baseline) : {auc_all:.4f}")
print(f"    OrPSOC selected            : {auc_selected:.4f}  "
      f"({'better' if auc_selected > auc_all else 'worse'} than all-features)")
print(f"    Oracle (signal only)       : {auc_oracle:.4f}")
print(f"    Runtime                    : {result['runtime']:.1f}s")
print()

checks = [
    ("Recall >= 0.6",       recall    >= 0.6),
    ("Fitness improved",    result["fit_hist"][-1] > result["fit_hist"][0]),
    ("Diversity decreased", result["div_hist"][-1] < result["div_hist"][0]),
    ("AUC >= all-features", auc_selected >= auc_all - 0.01),
]
print("  Self-checks:")
all_pass = True
for label, ok in checks:
    sym = "PASS" if ok else "FAIL"
    print(f"    {sym}  {label}")
    if not ok:
        all_pass = False
print()
if all_pass:
    print("  All checks passed. OrPSOC works on stationary data.")
else:
    print("  Some checks failed. See notes below.")
print()


# ── Convergence plot ──────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
iters = range(len(result["fit_hist"]))

ax1.plot(iters, result["fit_hist"], "b-o", markersize=4, linewidth=2,
         label="Global best fitness")
ax1.axhline(auc_oracle, color="green", linestyle="--", alpha=0.7,
            label=f"Oracle AUC = {auc_oracle:.4f}")
ax1.axhline(auc_all, color="orange", linestyle=":", alpha=0.7,
            label=f"All-features AUC = {auc_all:.4f}")
ax1.set_ylabel("Fitness (θ-normalised AUC)", fontsize=10)
ax1.set_title("OrPSOC Convergence on Level 2 — Stationary AR(1)\n"
              "Fitness should rise; diversity should fall", fontsize=10)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

ax2.plot(iters, result["div_hist"], "r-s", markersize=4, linewidth=2,
         label="Swarm diversity (Hamming)")
ax2.axhline(0, color="gray", linestyle="--", alpha=0.4, label="Fully converged")
ax2.set_ylabel("Swarm Diversity", fontsize=10)
ax2.set_xlabel("Iteration", fontsize=10)
ax2.set_title("Diversity: starts high (orthogonal init), decreases as swarm converges",
              fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/step3_orpsoc_convergence.png", dpi=150, bbox_inches="tight")
print("Saved: plots/step3_orpsoc_convergence.png")

with open("results/step3_orpsoc_stationary.json", "w") as f:
    json.dump({
        "selected":   selected,
        "precision":  precision,
        "recall":     recall,
        "f1":         f1,
        "auc_all":    auc_all,
        "auc_orpsoc": auc_selected,
        "auc_oracle": auc_oracle,
        "runtime":    result["runtime"],
    }, f, indent=2)
print("Saved: results/step3_orpsoc_stationary.json")
print()
print("=" * 55)
print("  STEP 3 COMPLETE")
print("=" * 55)
print()
print("RECORD THESE IN YOUR NOTEBOOK:")
print(f"  Recall   : {recall:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  AUC gain : {auc_selected - auc_all:+.4f} vs all features")
print()
print("If recall > 0.6 and AUC >= all-features:")
print("  OrPSOC foundation is valid. Proceed to Step 4.")
print()
print("Next: run step4_adaptive_crossover.py")