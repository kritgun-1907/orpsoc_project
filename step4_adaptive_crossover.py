"""
STEP 4 — Adaptive Crossover OrPSOC on Regime-Switch Data
==========================================================
This is the core experiment. This is what you email Indu about.

⚠ CANONICAL-ENGINE NOTE
────────────────────────
This file is the STANDALONE, pedagogical origin of the three-phase adaptive
crossover idea, kept for the mechanism plots (cr(t)/w(t) trajectories).
The PAPER-GRADE engine used for reported results (step7/step8/step_real_data)
lives in orpsoc_utils.run_hybrid_orpsoc(), which additionally has elite-
preserving partial restart, importance-guided reinit, and P(Transition)-scaled
drift response. The AdaptiveCRW.step() logic below has been patched to ramp
INTO the Phase-2 burst (rather than jumping instantly), so it is consistent
with the canonical engine — but treat orpsoc_utils as the source of truth.

YOUR FORMULATION (from the screenshot):
─────────────────────────────────────────
Three phases relative to each detected regime change at t_change:

  PRE-CHANGE STABLE PHASE (t < t_change):
    cr(t) = cr_low = 0.3        low crossover, exploit stable optimum
    w(t)  = w_low + decay * t   gradual inertia decay within phase

  TRANSITION PHASE (t_change <= t < t_change + N_explore):
    cr(t) = cr_high = 0.9       maximum diversity injection
    w(t)  = w_max = 0.9         maximum exploration momentum

  POST-TRANSITION ADAPTATION PHASE (t >= t_change + N_explore):
    cr(t) = cr_low + (cr_high - cr_low) * exp(-lambda * (t - t_change - N_explore))
    w(t)  = w_min + (w_max - w_min)   * exp(-lambda * (t - t_change - N_explore))

WHERE lambda = 0.1: after 50 iterations, both cr and w
have returned to within 1/e of baseline.

WHAT THIS STEP DOES:
──────────────────────
1. Runs STANDARD OrPSOC on Level 4 (regime-switch) data
2. Runs ADAPTIVE OrPSOC on the same data
3. Compares feature recovery BEFORE and AFTER the switch
4. Plots the cr(t) and w(t) trajectories so you can see the mechanism

THE KEY METRIC:
────────────────
Recovery speed = how many iterations after the regime switch
until OrPSOC is selecting the correct features again.

Standard OrPSOC: slow or never recovers.
Adaptive OrPSOC: detects the switch and re-explores fast.

Run with:
    python step4_adaptive_crossover.py
"""

# ── import guard ─────────────────────────────────────────────────────────────
# This file is a SCRIPT, not a module. It executes its whole pipeline at module
# level, so `import step4_adaptive_crossover` runs the entire thing as a side effect --
# for step7_ablation that is a 4-hour ablation triggered by an innocent-looking
# import. Fail loudly instead.
#
# `globals().get("__name__", ...)` rather than a bare `__name__`: helpers are
# reused by exec'ing the section above a marker into a fresh namespace (see
# experiments/apsoll_sweep.py), and that namespace has no __name__ at all.
if globals().get("__name__", "__main__") != "__main__":
    raise ImportError(
        "step4_adaptive_crossover.py is a script, not an importable module -- importing it would "
        "execute the full pipeline. To reuse a helper, exec the section above "
        "the main loop into a fresh namespace (see experiments/apsoll_sweep.py)."
    )
# ─────────────────────────────────────────────────────────────────────────────


import numpy as np
import pandas as pd
import pickle
import time
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier   # CONSISTENCY FIX: was XGBClassifier.
# NOTE: step4 is the legacy/pedagogical engine (see banner above). Its local
# evaluate() still uses a FLAT-penalty fitness (auc - penalty*excess), unlike
# the canonical theta-normalised fitness in orpsoc_utils.evaluate(). Only the
# model is aligned to LightGBM here; for reported results use orpsoc_utils.
import warnings
warnings.filterwarnings("ignore")
import os
os.makedirs("plots",   exist_ok=True)
os.makedirs("results", exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED UTILITIES (same as Step 3)
# ══════════════════════════════════════════════════════════════════════════════

def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-np.clip(v, -500, 500)))

def build_orthogonal_positions(n_particles, n_features):
    rng  = np.random.RandomState(seed=42)
    grid = np.zeros((n_particles, n_features), dtype=float)
    for j in range(n_features):
        n_ones = n_particles // 2
        shift  = (j * (n_particles // max(n_features, 1))) % n_particles
        for k in range(n_ones):
            grid[(shift + k) % n_particles, j] = 1.0
    return grid[rng.permutation(n_particles)]

def crossover(pos_a, pos_b, rate, min_f, rng):
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

def hamming_diversity(particles):
    pos   = np.array([p["pos"] for p in particles])
    n, d  = pos.shape
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += np.sum(pos[i] != pos[j]) / d
            count += 1
    return total / count if count > 0 else 0.0

def evaluate(pos, feat_names, X_tr, y_tr, X_va, y_va, min_f, penalty=0.001):
    idx = np.where(pos == 1)[0]
    if len(idx) < min_f:
        return -1.0
    cols = [feat_names[i] for i in idx]
    try:
        pipe = Pipeline([
            ("imp",    SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model",  LGBMClassifier(
                n_estimators=80, num_leaves=15,
                learning_rate=0.1, verbosity=-1, random_state=42
            ))
        ])
        pipe.fit(X_tr[cols], y_tr)
        auc = roc_auc_score(y_va, pipe.predict_proba(X_va[cols])[:, 1])
    except Exception:
        return -1.0
    return float(auc - penalty * max(0, len(idx) - min_f))


# ══════════════════════════════════════════════════════════════════════════════
#  REGIME CHANGE DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

def detect_regime_change(
    fit_hist:   list,
    window:     int   = 5,
    threshold:  float = 0.05,
) -> bool:
    """
    Detect whether a regime change has occurred by monitoring
    fitness degradation.

    HOW IT WORKS:
    ─────────────
    Look at the last `window` fitness values.
    If the CURRENT fitness is more than `threshold` below the
    BEST fitness in that window — the landscape has shifted.
    The feature subset that was optimal is no longer optimal.
    This is the signal to trigger diversity injection.

    WHY FITNESS DEGRADATION?
    ──────────────────────────
    In a stable regime, OrPSOC converges → fitness plateau.
    When a regime change occurs, the new data distribution
    makes previously good features less predictive.
    Fitness DROPS. The detector catches this drop.

    LIMITATION (honest):
    ─────────────────────
    This is a simple heuristic. In real financial data, you
    would use a more robust detector (ADWIN, Page-Hinkley, GMM).
    But for the synthetic experiment, fitness degradation is
    a reliable signal because we control the ground truth.

    Parameters
    ──────────
    fit_hist  : list of best fitness values per iteration
    window    : how many recent iterations to look back
    threshold : how much drop triggers detection (0.05 = 5% AUC drop)

    Returns
    ───────
    True if regime change detected, False otherwise.
    """
    if len(fit_hist) < window + 1:
        return False
    recent_best = max(fit_hist[-window:])
    current     = fit_hist[-1]
    return (recent_best - current) > threshold


# ══════════════════════════════════════════════════════════════════════════════
#  THE ADAPTIVE CROSSOVER RATE MECHANISM
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveCRW:
    """
    Implements the three-phase adaptive crossover rate and
    inertia weight formulation from your screenshot.

    YOUR FORMULATION:
    ──────────────────
    Phase 1 — Stable (t < t_change):
      cr(t) = cr_low = 0.3
      w(t)  = w_high linearly decaying to w_low

    Phase 2 — Transition (t_change <= t < t_change + N_explore):
      cr(t) = cr_high = 0.9   (maximum diversity injection)
      w(t)  = w_max = 0.9     (maximum exploration)

    Phase 3 — Post-transition (t >= t_change + N_explore):
      cr(t) = cr_low + (cr_high - cr_low) * exp(-lambda * dt)
      w(t)  = w_min + (w_max - w_min)   * exp(-lambda * dt)
      where dt = t - t_change - N_explore

    INTUITION FOR EACH PHASE:
    ──────────────────────────
    Phase 1: The market is stable. We know which features work.
    Keep crossover low (0.3) so we don't disrupt a good solution.
    Let inertia slowly decay so swarm fine-tunes within current best.

    Phase 2: Regime change detected! Old feature subset is wrong.
    Max crossover (0.9) forces the swarm to explore radically new
    combinations. Max inertia (0.9) gives particles momentum to
    escape the converged basin and explore distant regions.
    N_explore = 20 iterations of this aggressive re-exploration.

    Phase 3: Swarm has explored. Now re-converge on the new optimum.
    Exponentially decay both cr and w back to stable-phase values.
    Lambda = 0.1 → takes ~50 iterations to fully return.
    This is smooth and does not disrupt the newly forming convergence.

    WHY EXPONENTIAL DECAY (not linear)?
    ─────────────────────────────────────
    Linear decay: parameter changes at constant rate throughout.
    Exponential decay: parameter changes FAST initially (urgency to
    exploit new information) then SLOWLY (fine-tuning).
    This matches how optimization should behave post-regime-change.
    """

    def __init__(
        self,
        cr_low    = 0.3,
        cr_high   = 0.9,
        w_min     = 0.4,
        w_max     = 0.9,
        N_explore = 20,
        lam       = 0.1,
        w_start   = 0.9,
        w_end     = 0.4,
        max_iter  = 60,
        ramp_iters = 5,
    ):
        self.cr_low    = cr_low
        self.cr_high   = cr_high
        self.w_min     = w_min
        self.w_max     = w_max
        self.N_explore = N_explore
        self.lam       = lam
        self.w_start   = w_start
        self.w_end     = w_end
        self.max_iter  = max_iter
        self.ramp_iters = ramp_iters

        # State
        self.t_change  = None   # iteration when regime change was detected
        self.phase     = "stable"

        # History for plotting
        self.cr_history = []
        self.w_history  = []
        self.phase_history = []

    def step(self, iteration: int, regime_changed: bool) -> tuple:
        """
        Given the current iteration and whether a regime change was
        just detected, return (cr, w) for this iteration.

        Parameters
        ──────────
        iteration      : current PSO iteration (0-indexed)
        regime_changed : True if detector fired this iteration

        Returns
        ───────
        (cr, w) : crossover rate and inertia weight for this iteration
        """

        # ── Regime change just detected → enter transition phase ──────────────
        if regime_changed and self.phase == "stable":
            self.t_change = iteration
            self.phase    = "transition"

        # ── Determine current phase ───────────────────────────────────────────
        if self.phase == "stable" or self.t_change is None:
            # Phase 1: normal linear decay, low crossover
            cr = self.cr_low
            w  = (self.w_start
                  - (self.w_start - self.w_end)
                  * (iteration / max(self.max_iter - 1, 1)))

        elif self.phase == "transition":
            dt = iteration - self.t_change
            if dt < self.N_explore:
                # Phase 2: ramp UP into the diversity burst over ramp_iters
                # iterations rather than an instant jump to cr_high/w_max.
                # The instant jump "shook the whole swarm" and drove the
                # fold-5 transition crash the professor flagged.
                ramp = min(1.0, (dt + 1) / max(self.ramp_iters, 1))
                cr = self.cr_low + (self.cr_high - self.cr_low) * ramp
                w  = self.w_min  + (self.w_max   - self.w_min)  * ramp
            else:
                # Phase 3: exponential decay back to baseline
                self.phase = "adapting"
                dt_adapt   = dt - self.N_explore
                cr = self.cr_low + (self.cr_high - self.cr_low) * np.exp(-self.lam * dt_adapt)
                w  = self.w_min  + (self.w_max   - self.w_min)  * np.exp(-self.lam * dt_adapt)

        elif self.phase == "adapting":
            dt_adapt = (iteration - self.t_change) - self.N_explore
            cr = self.cr_low + (self.cr_high - self.cr_low) * np.exp(-self.lam * dt_adapt)
            w  = self.w_min  + (self.w_max   - self.w_min)  * np.exp(-self.lam * dt_adapt)

        else:
            cr = self.cr_low
            w  = self.w_end

        self.cr_history.append(cr)
        self.w_history.append(w)
        self.phase_history.append(self.phase)

        return float(cr), float(w)


# ══════════════════════════════════════════════════════════════════════════════
#  ORPSOC — STANDARD (fixed cr and w)
# ══════════════════════════════════════════════════════════════════════════════

def run_standard_orpsoc(
    X_train, y_train, X_val, y_val, feat_names,
    n_particles=20, max_iter=60, min_f=5,
    w_start=0.9, w_end=0.4, c1=2.0, c2=2.0, cr=0.6,
    seed=42, verbose=True,
):
    """Standard OrPSOC with fixed crossover rate throughout."""
    rng = np.random.RandomState(seed)
    n   = len(feat_names)
    t0  = time.time()

    init_pos  = build_orthogonal_positions(n_particles, n)
    particles = []
    for i in range(n_particles):
        pos = init_pos[i].copy()
        n_sel = int(pos.sum())
        if n_sel < min_f:
            zeros = np.where(pos == 0)[0]
            pos[rng.choice(zeros, size=min_f - n_sel, replace=False)] = 1
        particles.append({
            "pos": pos, "vel": rng.uniform(-3, 3, size=n),
            "best_pos": pos.copy(), "best_fit": -np.inf,
        })

    gbest_pos, gbest_fit = None, -np.inf
    for p in particles:
        fit = evaluate(p["pos"], feat_names, X_train, y_train, X_val, y_val, min_f)
        p["best_fit"] = fit
        if fit > gbest_fit:
            gbest_fit = fit
            gbest_pos = p["pos"].copy()

    fit_hist = [gbest_fit]
    div_hist = [hamming_diversity(particles)]

    if verbose:
        print(f"  {'Iter':>5}  {'Fitness':>9}  {'Diversity':>10}  {'cr':>5}  {'w':>5}")

    for it in range(max_iter):
        w_cur = w_start - (w_start - w_end) * (it / max(max_iter - 1, 1))

        for p in particles:
            r1 = rng.rand(n)
            r2 = rng.rand(n)
            vel = (w_cur * p["vel"]
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

        idx_list = list(range(n_particles))
        rng.shuffle(idx_list)
        for k in range(0, n_particles - 1, 2):
            pa = particles[idx_list[k]]
            pb = particles[idx_list[k + 1]]
            ca, cb = crossover(pa["pos"], pb["pos"], cr, min_f, rng)
            for child, parent in [(ca, pa), (cb, pb)]:
                fit = evaluate(child, feat_names, X_train, y_train, X_val, y_val, min_f)
                if fit > parent["best_fit"]:
                    parent["pos"]      = child
                    parent["best_pos"] = child.copy()
                    parent["best_fit"] = fit

        for p in particles:
            fit = evaluate(p["pos"], feat_names, X_train, y_train, X_val, y_val, min_f)
            if fit > p["best_fit"]:
                p["best_fit"] = fit
                p["best_pos"] = p["pos"].copy()
            if fit > gbest_fit:
                gbest_fit = fit
                gbest_pos = p["pos"].copy()

        fit_hist.append(gbest_fit)
        div_hist.append(hamming_diversity(particles))

        if verbose and (it % 10 == 0 or it == max_iter - 1):
            print(f"  {it+1:>5}  {gbest_fit:>9.4f}  "
                  f"{div_hist[-1]:>10.3f}  {cr:>5.2f}  {w_cur:>5.2f}")

    sel_idx  = np.where(gbest_pos == 1)[0]
    return {
        "selected":  [feat_names[i] for i in sel_idx],
        "fit_hist":  fit_hist,
        "div_hist":  div_hist,
        "runtime":   time.time() - t0,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ORPSOC — ADAPTIVE (your formulation from the screenshot)
# ══════════════════════════════════════════════════════════════════════════════

def run_adaptive_orpsoc(
    X_train, y_train, X_val, y_val, feat_names,
    n_particles=20, max_iter=60, min_f=5,
    c1=2.0, c2=2.0,
    cr_low=0.3, cr_high=0.9,
    w_min=0.4, w_max=0.9,
    N_explore=20, lam=0.1,
    detect_window=5, detect_threshold=0.05,
    seed=42, verbose=True,
):
    """
    AdaptiveOrPSOC — your three-phase formulation implemented.

    Key differences from standard OrPSOC:
    1. Crossover rate is NOT fixed — it responds to regime changes
    2. Inertia weight follows the three-phase schedule, not linear decay
    3. A regime change detector monitors fitness degradation
    4. On detection: cr and w spike to maximum for N_explore iterations
    5. Then both decay exponentially back to stable-phase values
    """
    rng    = np.random.RandomState(seed)
    n      = len(feat_names)
    t0     = time.time()
    crw    = AdaptiveCRW(
        cr_low=cr_low, cr_high=cr_high,
        w_min=w_min, w_max=w_max,
        N_explore=N_explore, lam=lam,
        w_start=w_max, w_end=w_min,
        max_iter=max_iter,
    )

    init_pos  = build_orthogonal_positions(n_particles, n)
    particles = []
    for i in range(n_particles):
        pos = init_pos[i].copy()
        n_sel = int(pos.sum())
        if n_sel < min_f:
            zeros = np.where(pos == 0)[0]
            pos[rng.choice(zeros, size=min_f - n_sel, replace=False)] = 1
        particles.append({
            "pos": pos, "vel": rng.uniform(-3, 3, size=n),
            "best_pos": pos.copy(), "best_fit": -np.inf,
        })

    gbest_pos, gbest_fit = None, -np.inf
    for p in particles:
        fit = evaluate(p["pos"], feat_names, X_train, y_train, X_val, y_val, min_f)
        p["best_fit"] = fit
        if fit > gbest_fit:
            gbest_fit = fit
            gbest_pos = p["pos"].copy()

    fit_hist      = [gbest_fit]
    div_hist      = [hamming_diversity(particles)]
    change_iters  = []   # record when detections fired

    if verbose:
        print(f"  {'Iter':>5}  {'Fitness':>9}  {'Div':>6}  {'cr':>5}  {'w':>5}  Phase")

    for it in range(max_iter):
        # ── Detect regime change ──────────────────────────────────────────────
        regime_changed = detect_regime_change(
            fit_hist, window=detect_window, threshold=detect_threshold
        )
        if regime_changed:
            change_iters.append(it)

        # ── Get adaptive cr and w for this iteration ──────────────────────────
        cr_cur, w_cur = crw.step(it, regime_changed)

        # ── Velocity + position update ────────────────────────────────────────
        for p in particles:
            r1 = rng.rand(n)
            r2 = rng.rand(n)
            vel = (w_cur * p["vel"]
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
            ca, cb = crossover(pa["pos"], pb["pos"], cr_cur, min_f, rng)
            for child, parent in [(ca, pa), (cb, pb)]:
                fit = evaluate(child, feat_names, X_train, y_train, X_val, y_val, min_f)
                if fit > parent["best_fit"]:
                    parent["pos"]      = child
                    parent["best_pos"] = child.copy()
                    parent["best_fit"] = fit

        # ── Evaluate + update bests ───────────────────────────────────────────
        for p in particles:
            fit = evaluate(p["pos"], feat_names, X_train, y_train, X_val, y_val, min_f)
            if fit > p["best_fit"]:
                p["best_fit"] = fit
                p["best_pos"] = p["pos"].copy()
            if fit > gbest_fit:
                gbest_fit = fit
                gbest_pos = p["pos"].copy()

        fit_hist.append(gbest_fit)
        div_hist.append(hamming_diversity(particles))

        if verbose and (it % 10 == 0 or it == max_iter - 1):
            flag = " <<< DETECTED" if regime_changed else ""
            print(f"  {it+1:>5}  {gbest_fit:>9.4f}  "
                  f"{div_hist[-1]:>6.3f}  {cr_cur:>5.2f}  {w_cur:>5.2f}"
                  f"  {crw.phase:<12}{flag}")

    sel_idx = np.where(gbest_pos == 1)[0]
    return {
        "selected":     [feat_names[i] for i in sel_idx],
        "fit_hist":     fit_hist,
        "div_hist":     div_hist,
        "cr_hist":      crw.cr_history,
        "w_hist":       crw.w_history,
        "phase_hist":   crw.phase_history,
        "change_iters": change_iters,
        "runtime":      time.time() - t0,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — Run both on Level 4 (Regime Switch)
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("  STEP 4: Adaptive vs Standard OrPSOC on Level 4 (Regime Switch)")
print("=" * 60)
print()

with open("data/regime_switch.pkl", "rb") as f:
    data = pickle.load(f)
X, y = data["X"], data["y"]
feat_names = list(X.columns)

# Temporal split
n           = len(X)
split_train = int(n * 0.55)   # slightly earlier to capture regime in val
split_val   = int(n * 0.75)

X_train = X.iloc[:split_train]
y_train = y.iloc[:split_train]
X_val   = X.iloc[split_train:split_val]
y_val   = y.iloc[split_train:split_val]
X_test  = X.iloc[split_val:]
y_test  = y.iloc[split_val:]

# Key feature groups
signal_r1   = ["signal_0", "signal_1", "signal_2"]   # active in Regime 1
signal_r2   = ["signal_3", "signal_4"]                # active in Regime 2
signal_all  = signal_r1 + signal_r2

print(f"Data split: train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")
print(f"Regime 1 signal features: {signal_r1}")
print(f"Regime 2 signal features: {signal_r2}")
print()

# ── Run standard OrPSOC ───────────────────────────────────────────────────────
print("─" * 60)
print("Standard OrPSOC (fixed cr=0.6):")
print("─" * 60)
std_result = run_standard_orpsoc(
    X_train, y_train, X_val, y_val, feat_names,
    n_particles=20, max_iter=60, min_f=5, cr=0.6, verbose=True,
)
print()

# ── Run adaptive OrPSOC ───────────────────────────────────────────────────────
print("─" * 60)
print("Adaptive OrPSOC (your three-phase formulation):")
print("─" * 60)
adp_result = run_adaptive_orpsoc(
    X_train, y_train, X_val, y_val, feat_names,
    n_particles=20, max_iter=60, min_f=5,
    cr_low=0.3, cr_high=0.9,
    w_min=0.4,  w_max=0.9,
    N_explore=20, lam=0.1,
    detect_window=5, detect_threshold=0.04,
    verbose=True,
)
print()


# ── Compare what each selected ────────────────────────────────────────────────
def score_selection(selected, signal_r1, signal_r2):
    tp_r1 = [f for f in selected if f in signal_r1]
    tp_r2 = [f for f in selected if f in signal_r2]
    fp    = [f for f in selected if f not in signal_r1 + signal_r2]
    recall_r1 = len(tp_r1) / len(signal_r1)
    recall_r2 = len(tp_r2) / len(signal_r2)
    return recall_r1, recall_r2, tp_r1, tp_r2, fp

std_r1, std_r2, *_ = score_selection(std_result["selected"], signal_r1, signal_r2)
adp_r1, adp_r2, *_ = score_selection(adp_result["selected"], signal_r1, signal_r2)

print("=" * 60)
print("  COMPARISON")
print("=" * 60)
print()
print(f"  {'Metric':<35}  {'Standard':>10}  {'Adaptive':>10}")
print(f"  {'─'*57}")
print(f"  {'Regime 1 recall (signal_0,1,2)':<35}  {std_r1:>10.4f}  {adp_r1:>10.4f}")
print(f"  {'Regime 2 recall (signal_3,4)':<35}  {std_r2:>10.4f}  {adp_r2:>10.4f}")
print(f"  {'Combined recall':<35}  "
      f"{(std_r1+std_r2)/2:>10.4f}  {(adp_r1+adp_r2)/2:>10.4f}")
print(f"  {'Detections fired':<35}  {'N/A':>10}  "
      f"{len(adp_result['change_iters']):>10}")
print(f"  {'Runtime (s)':<35}  {std_result['runtime']:>10.1f}  "
      f"{adp_result['runtime']:>10.1f}")
print()
print(f"  Standard selected : {std_result['selected']}")
print(f"  Adaptive selected : {adp_result['selected']}")
print()

improvement = (adp_r1 + adp_r2) / 2 - (std_r1 + std_r2) / 2
if improvement > 0.05:
    print(f"  RESULT: Adaptive OrPSOC improved combined recall by {improvement:.4f}")
    print("  Your three-phase formulation is working.")
elif improvement > 0:
    print(f"  RESULT: Slight improvement ({improvement:.4f}). Try tuning N_explore or lambda.")
else:
    print(f"  RESULT: No improvement. Check detect_threshold — may need tuning.")
    print("  This is normal — the threshold and detector may need calibration.")
print()


# ── Visualise cr(t) and w(t) trajectories ────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 12))

iters = range(len(adp_result["fit_hist"]))

# Panel 1: Fitness comparison
ax = axes[0]
ax.plot(iters, adp_result["fit_hist"], "b-", linewidth=2, label="Adaptive OrPSOC")
ax.plot(iters, std_result["fit_hist"], "r--", linewidth=2, label="Standard OrPSOC (fixed cr=0.6)")
for ci in adp_result["change_iters"]:
    ax.axvline(ci, color="orange", linewidth=1.5, alpha=0.7, linestyle=":")
if adp_result["change_iters"]:
    ax.axvline(adp_result["change_iters"][0], color="orange",
               linewidth=1.5, alpha=0.7, linestyle=":", label="Regime change detected")
ax.set_title("Fitness Comparison: Standard vs Adaptive OrPSOC\n"
             "Adaptive should recover faster after regime change", fontsize=10)
ax.set_ylabel("Fitness (AUC - penalty)", fontsize=9)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 2: Adaptive cr(t) trajectory
ax = axes[1]
cr_iters = range(len(adp_result["cr_hist"]))
ax.plot(cr_iters, adp_result["cr_hist"], "b-", linewidth=2.5, label="cr(t) adaptive")
ax.axhline(0.6, color="red", linewidth=1.5, linestyle="--",
           alpha=0.7, label="Standard fixed cr = 0.6")
ax.axhline(0.3, color="gray", linewidth=1, linestyle=":",
           alpha=0.6, label="cr_low = 0.3 (stable phase)")
ax.axhline(0.9, color="green", linewidth=1, linestyle=":",
           alpha=0.6, label="cr_high = 0.9 (transition phase)")
for ci in adp_result["change_iters"]:
    ax.axvline(ci, color="orange", linewidth=1.5, linestyle=":", alpha=0.7)
ax.set_title("Crossover Rate cr(t) — Your Three-Phase Formulation\n"
             "Spikes to 0.9 on regime change, decays back to 0.3", fontsize=10)
ax.set_ylabel("Crossover Rate cr", fontsize=9)
ax.set_ylim([0.0, 1.05])
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# Panel 3: Adaptive w(t) trajectory
ax = axes[2]
w_iters = range(len(adp_result["w_hist"]))
ax.plot(w_iters, adp_result["w_hist"], "r-", linewidth=2.5, label="w(t) adaptive")
w_linear = [0.9 - (0.9 - 0.4) * (i / max(max(w_iters), 1)) for i in w_iters]
ax.plot(w_iters, w_linear, "b--", linewidth=1.5, alpha=0.7, label="Standard linear decay")
for ci in adp_result["change_iters"]:
    ax.axvline(ci, color="orange", linewidth=1.5, linestyle=":", alpha=0.7)
ax.set_title("Inertia Weight w(t) — Resets to w_max on Regime Change\n"
             "High w = explore, Low w = exploit", fontsize=10)
ax.set_ylabel("Inertia Weight w", fontsize=9)
ax.set_xlabel("Iteration", fontsize=9)
ax.set_ylim([0.2, 1.05])
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/step4_adaptive_vs_standard.png", dpi=150, bbox_inches="tight")
print("Saved: plots/step4_adaptive_vs_standard.png")

# Save results
with open("results/step4_adaptive_comparison.json", "w") as f:
    json.dump({
        "standard": {
            "selected": std_result["selected"],
            "recall_r1": std_r1, "recall_r2": std_r2,
            "runtime":   std_result["runtime"],
        },
        "adaptive": {
            "selected":      adp_result["selected"],
            "recall_r1":     adp_r1,
            "recall_r2":     adp_r2,
            "change_iters":  adp_result["change_iters"],
            "runtime":       adp_result["runtime"],
        },
        "improvement": improvement,
    }, f, indent=2)
print("Saved: results/step4_adaptive_comparison.json")
print()
print("=" * 60)
print("  STEP 4 COMPLETE")
print("=" * 60)
print()
print("The three plots tell the story:")
print()
print("  Plot 1 (Fitness): Does adaptive recover faster after the regime switch?")
print("  Plot 2 (cr):      Does cr spike on detection and decay smoothly?")
print("  Plot 3 (w):       Does w reset on detection and decay smoothly?")
print()
print("If Plots 2 and 3 show the correct three-phase shape but Plot 1")
print("shows no improvement — the detector threshold needs tuning.")
print("Try: detect_threshold=0.03 or detect_threshold=0.02")
print()
print("WRITE DOWN AND EMAIL TO INDU:")
print(f"  Standard recall: R1={std_r1:.3f}  R2={std_r2:.3f}")
print(f"  Adaptive recall: R1={adp_r1:.3f}  R2={adp_r2:.3f}")
print(f"  Improvement:     {improvement:+.4f}")
print(f"  Detections:      {adp_result['change_iters']}")
