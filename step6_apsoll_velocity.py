"""
STEP 6 — APSOLL Adaptive Velocity + Leadership Update
======================================================
Integrates the two APSOLL mechanisms into your OrPSOC engine.

⚠ CANONICAL-ENGINE NOTE
────────────────────────
This file contains a STANDALONE, pedagogical implementation of the hybrid
engine, kept for the step-by-step narrative and the single-run Level-4 plot.
The PAPER-GRADE engine actually used by step7_ablation.py, step8_results.py
and step_real_data.py lives in orpsoc_utils.run_hybrid_orpsoc(). That version
additionally has: elite-preserving partial restart, importance-guided reinit,
proportional (P-transition-scaled) drift response, and the delayed-HMM trigger.
This local copy has been patched to share the gradual Phase-2 ramp so it no
longer contradicts the canonical engine, but for any reported result import
from orpsoc_utils rather than re-running this file.

WHAT IS NEW VERSUS STEP 3/4:
──────────────────────────────
Step 3/4 used:  fixed c1=2.0, c2=2.0  (same pull strength every iteration)
Step 6 adds:    adaptive c = (m/T)^0.67 + 1  (APSOLL equation 6)

Step 3/4 used:  two-term PSO velocity
Step 6 adds:    three-leader velocity in Phase 2 (APSOLL equation 7)

Step 3/4 used:  AUC - flat_penalty fitness
Step 6 adds:    θ*AUC + (1-θ)*(1 - #sel/N) fitness (APSOLL equation 10)

FIX LOG:
  - Phase logic changed from if/if/if → if/elif/elif to prevent
    double-decrement of n_explore_rem on Phase 1 → Phase 2 transition.
  - Phase variables initialised completely before the main loop.

Run with:
    python step6_apsoll_velocity.py
"""

import numpy as np
import pandas as pd
import pickle
import time
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")
import os
os.makedirs("plots",   exist_ok=True)
os.makedirs("results", exist_ok=True)

from orpsoc_utils import (
    sigmoid, build_orthogonal_positions, crossover,
    hamming_diversity, evaluate, walk_forward_folds,
    feature_stability_ratio, APSOLLAdaptiveC,
)


# ══════════════════════════════════════════════════════════════════════════════
#  HYBRID VELOCITY UPDATE
# ══════════════════════════════════════════════════════════════════════════════

def hybrid_velocity_update(p: dict, gbest_pos: np.ndarray,
                           top3_positions: list,
                           w: float, c_adaptive: float,
                           phase: int, dt: int,
                           n_features: int, rng) -> np.ndarray:
    """
    Chooses velocity update formula based on current phase.

    Phase 1 — Standard two-term PSO with adaptive c:
      v = w*v + (c/2)*r1*(pbest-x) + (c/2)*r2*(gbest-x)

    Phase 2 — Three-leader APSOLL update (maximum diversity):
      v = w*v + (c/2)*r4*(X1-x) + (c/3)*r4*(X2-x) + (c/4)*r4*(X3-x)
      where X1,X2,X3 are positions informed by top-3 leaders (GWO-style)

    Phase 3 — Exponential blend from leadership back to standard:
      v = blend*leadership + (1-blend)*standard
      blend decays from 1.0 to ~0 as dt grows (λ=0.1)
    """
    r1 = rng.rand(n_features)
    r2 = rng.rand(n_features)
    r4 = rng.rand(n_features)

    # Standard two-term
    standard = (w * p["vel"]
                + (c_adaptive / 2) * r1 * (p["best_pos"] - p["pos"])
                + (c_adaptive / 2) * r2 * (gbest_pos - p["pos"]))

    if phase == 1:
        return standard

    # Build leadership vectors using GWO-style distance (APSOLL eq. 3)
    if len(top3_positions) >= 3:
        X1 = top3_positions[0] - np.abs(2 * rng.rand(n_features) * top3_positions[0] - p["pos"])
        X2 = top3_positions[1] - np.abs(2 * rng.rand(n_features) * top3_positions[1] - p["pos"])
        X3 = top3_positions[2] - np.abs(2 * rng.rand(n_features) * top3_positions[2] - p["pos"])
    else:
        X1 = X2 = X3 = gbest_pos

    leadership = (w * p["vel"]
                  + (c_adaptive / 2) * r4 * (X1 - p["pos"])
                  + (c_adaptive / 3) * r4 * (X2 - p["pos"])
                  + (c_adaptive / 4) * r4 * (X3 - p["pos"]))

    if phase == 2:
        return leadership

    # Phase 3: smooth exponential blend
    blend = np.exp(-0.1 * dt)
    return blend * leadership + (1.0 - blend) * standard


# ══════════════════════════════════════════════════════════════════════════════
#  FULL HYBRID ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def run_hybrid_orpsoc(
    X_train, y_train, X_val, y_val,
    feat_names: list,
    hmm_trigger: bool = False,
    n_particles: int  = 20,
    max_iter:    int  = 60,
    min_f:       int  = 5,
    theta:       float = 0.7,
    cr_low:      float = 0.3,
    cr_high:     float = 0.8,
    w_min:       float = 0.4,
    w_max:       float = 0.9,
    N_explore:   int   = 20,
    lam:         float = 0.1,
    ramp_iters:  int   = 5,
    seed:        int   = 42,
    verbose:     bool  = True,
) -> dict:
    """
    Full hybrid engine: OrPSOC + APSOLL-c + three-leader + normalised fitness.

    hmm_trigger: set True when the HMM detector fired for this fold.
    When True, the engine starts in Phase 2 (transition) immediately.
    When False, starts in Phase 1 (stable) and monitors fitness internally.

    Returns: dict with selected, fit_hist, div_hist, c_hist, cr_hist,
             w_hist, phase_hist, runtime
    """
    rng      = np.random.RandomState(seed)
    n        = len(feat_names)
    t0       = time.time()

    # ── Initialise particles ──────────────────────────────────────────────────
    init_pos = build_orthogonal_positions(n_particles, n, seed)
    particles = []
    for i in range(n_particles):
        pos = init_pos[i].copy()
        particles.append({
            "pos":      pos.copy(),
            "vel":      rng.randn(n) * 0.1,
            "best_pos": pos.copy(),
            "best_fit": evaluate(pos, feat_names, X_train, y_train,
                                  X_val, y_val, min_f, theta),
        })

    gbest_pos = max(particles, key=lambda p: p["best_fit"])["best_pos"].copy()
    gbest_fit = max(p["best_fit"] for p in particles)

    # ── Adaptive c tracker ────────────────────────────────────────────────────
    adaptive_c = APSOLLAdaptiveC(max_iter)

    # ── Phase state (fully initialised) ──────────────────────────────────────
    # If HMM triggered, start in Phase 2 immediately
    phase         = 2 if hmm_trigger else 1
    t_change      = 0 if hmm_trigger else None
    dt            = 0   # iterations since phase 2 started
    n_explore_rem = N_explore

    fit_hist   = []
    div_hist   = []
    cr_hist    = []
    w_hist     = []
    c_hist     = []
    phase_hist = []

    if verbose:
        print(f"  {'Iter':>5}  {'Fitness':>10}  {'Diversity':>10}  "
              f"{'c':>6}  {'cr':>6}  {'w':>6}  {'Phase':>6}  {'#Sel':>5}")

    for it in range(max_iter):

        # ── Compute adaptive c ────────────────────────────────────────────────
        c_t = adaptive_c.update(gbest_fit)

        # ── Compute phase parameters (FIXED: elif prevents double-decrement) ──
        if phase == 1:
            cr_t = cr_low
            w_t  = w_max - (w_max - w_min) * (it / max(max_iter - 1, 1))
            # Check if APSOLL-c signals stagnation (c dropped back to ~1.0)
            if it > 5 and c_t < 1.05 and len(fit_hist) >= 3:
                # Soft self-trigger: c collapses → enter transition
                phase         = 2
                t_change      = it
                dt            = 0
                n_explore_rem = N_explore

        elif phase == 2:
            # Gradual ramp INTO the burst over ramp_iters iterations instead of
            # an instant cr_low->cr_high jump (the instant reset "shook the whole
            # swarm" and caused the fold-5 transition crash the professor flagged).
            ramp = min(1.0, (dt + 1) / max(ramp_iters, 1))
            cr_t = cr_low + (cr_high - cr_low) * ramp
            w_t  = w_min  + (w_max  - w_min)  * ramp
            n_explore_rem -= 1
            dt += 1
            if n_explore_rem <= 0:
                phase = 3

        elif phase == 3:
            cr_t = cr_low + (cr_high - cr_low) * np.exp(-lam * dt)
            w_t  = w_min  + (w_max  - w_min)   * np.exp(-lam * dt)
            dt  += 1

        else:
            cr_t = cr_low
            w_t  = w_min

        # ── Get top-3 leaders for leadership velocity ─────────────────────────
        sorted_p = sorted(particles, key=lambda p: p["best_fit"], reverse=True)
        top3     = [p["best_pos"] for p in sorted_p[:3]]

        # ── Update velocities and positions ───────────────────────────────────
        for p in particles:
            vel = hybrid_velocity_update(
                p, gbest_pos, top3, w_t, c_t, phase, dt, n, rng)
            p["vel"] = vel
            new_pos = (rng.rand(n) < sigmoid(vel)).astype(float)
            n_sel = int(new_pos.sum())
            if n_sel < min_f:
                zeros = np.where(new_pos == 0)[0]
                need  = min_f - n_sel
                if len(zeros) >= need:
                    new_pos[rng.choice(zeros, size=need, replace=False)] = 1
            p["pos"] = new_pos

        # ── Crossover ─────────────────────────────────────────────────────────
        idx_list = list(range(n_particles))
        rng.shuffle(idx_list)
        for k in range(0, n_particles - 1, 2):
            pa = particles[idx_list[k]]
            pb = particles[idx_list[k + 1]]
            ca, cb = crossover(pa["pos"], pb["pos"], cr_t, min_f, rng)
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
        cr_hist.append(cr_t)
        w_hist.append(w_t)
        c_hist.append(c_t)
        phase_hist.append(phase)

        if verbose and (it % 10 == 0 or it == max_iter - 1):
            print(f"  {it+1:>5}  {gbest_fit:>10.4f}  "
                  f"{div_hist[-1]:>10.3f}  {c_t:>6.3f}  "
                  f"{cr_t:>6.3f}  {w_t:>6.3f}  {phase:>6}  "
                  f"{int(gbest_pos.sum()):>5}")

    sel_idx  = np.where(gbest_pos == 1)[0]
    sel_cols = [feat_names[i] for i in sel_idx]

    return {
        "selected":    sel_cols,
        "fit_hist":    fit_hist,
        "div_hist":    div_hist,
        "cr_hist":     cr_hist,
        "w_hist":      w_hist,
        "c_hist":      c_hist,
        "phase_hist":  phase_hist,
        "best_fitness": gbest_fit,
        "runtime":     time.time() - t0,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — Compare standard vs APSOLL-hybrid on Level 4
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("  STEP 6: APSOLL Adaptive Velocity on Level 4")
print("=" * 60)
print()

with open("data/regime_switch.pkl", "rb") as f:
    data = pickle.load(f)
X, y = data["X"], data["y"]
feat_names = list(X.columns)

n           = len(X)
split_train = int(n * 0.55)
split_val   = int(n * 0.75)
X_train = X.iloc[:split_train]
y_train = y.iloc[:split_train]
X_val   = X.iloc[split_train:split_val]
y_val   = y.iloc[split_train:split_val]

signal_r1  = ["signal_0", "signal_1", "signal_2"]
signal_r2  = ["signal_3", "signal_4"]

print("Running Hybrid OrPSOC (APSOLL-c + leadership + normalised fitness)...")
print()
result = run_hybrid_orpsoc(
    X_train, y_train, X_val, y_val, feat_names,
    hmm_trigger=False,     # will self-trigger via APSOLL-c collapse
    n_particles=20, max_iter=60, min_f=5, theta=0.7,
    cr_low=0.3, cr_high=0.8,
    w_min=0.4, w_max=0.9,
    N_explore=20, lam=0.1,
    verbose=True,
)
print()

# Score selection
selected = result["selected"]
tp_r1    = [f for f in selected if f in signal_r1]
tp_r2    = [f for f in selected if f in signal_r2]
recall_r1 = len(tp_r1) / len(signal_r1)
recall_r2 = len(tp_r2) / len(signal_r2)

print("=" * 60)
print("  RESULTS")
print("=" * 60)
print(f"  Selected features: {selected}")
print(f"  Regime 1 recall (signal_0,1,2): {recall_r1:.4f}")
print(f"  Regime 2 recall (signal_3,4):   {recall_r2:.4f}")
print(f"  Combined recall:                {(recall_r1+recall_r2)/2:.4f}")
print(f"  Runtime: {result['runtime']:.1f}s")
print()

# Plot
fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
iters = range(len(result["fit_hist"]))

axes[0].plot(iters, result["fit_hist"], "b-", linewidth=2)
axes[0].set_title("Fitness — APSOLL Hybrid OrPSOC (Level 4 Regime Switch)", fontsize=10)
axes[0].set_ylabel("Fitness (θ-normalised)", fontsize=9)
axes[0].grid(True, alpha=0.3)

axes[1].plot(iters, result["c_hist"], "g-", linewidth=2)
axes[1].set_title("Adaptive c(t) — resets to 1.0 on stagnation (APSOLL eq.6)", fontsize=10)
axes[1].axhline(1.0, color="gray", linestyle=":", alpha=0.5)
axes[1].axhline(2.0, color="gray", linestyle=":", alpha=0.5)
axes[1].set_ylabel("c value", fontsize=9)
axes[1].set_ylim([0.9, 2.1])
axes[1].grid(True, alpha=0.3)

axes[2].plot(iters, result["cr_hist"], "r-", linewidth=2, label="cr(t)")
axes[2].plot(iters, result["w_hist"], "b--", linewidth=2, label="w(t)")
axes[2].set_title("Crossover Rate cr(t) and Inertia w(t)", fontsize=10)
axes[2].set_ylabel("Value", fontsize=9)
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

phase_arr = np.array(result["phase_hist"])
for ph, col, label in [(1, "green", "Stable"), (2, "red", "Transition"), (3, "orange", "Post-trans")]:
    mask = phase_arr == ph
    axes[3].fill_between(iters, 0, mask.astype(float), alpha=0.4,
                         color=col, label=label)
axes[3].set_title("Phase History (1=Stable, 2=Transition, 3=Post-Trans)", fontsize=10)
axes[3].set_xlabel("Iteration", fontsize=9)
axes[3].set_ylabel("Phase", fontsize=9)
axes[3].legend(fontsize=8, loc="upper right")
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/step6_apsoll_hybrid.png", dpi=150, bbox_inches="tight")
print("Saved: plots/step6_apsoll_hybrid.png")

with open("results/step6_apsoll_hybrid.json", "w") as f:
    json.dump({
        "selected": selected,
        "recall_r1": recall_r1, "recall_r2": recall_r2,
        "fit_hist": result["fit_hist"],
        "c_hist":   result["c_hist"],
        "phase_hist": result["phase_hist"],
        "runtime": result["runtime"],
    }, f, indent=2)
print("Saved: results/step6_apsoll_hybrid.json")
print()
print("=" * 60)
print("  STEP 6 COMPLETE")
print("=" * 60)
print()
print("Next: run step7_ablation.py")