"""
Why does Full Hybrid adapt at fold 7 when the detector fires correctly at fold 4?

WHAT IS ALREADY KNOWN
─────────────────────
  * The HMM fires at fold 4 in 30/30 seeds -- the earliest CAUSALLY possible
    fold (the break must enter the TRAINING window first). Detection is perfect.
  * Full Hybrid's recall of the post-switch signals does not rise until fold 7.
  * elite_frac is NOT the cause: 240/240 units across {0.2, 0.1, 0.05} leave
    adaptation_fold pinned at 7 with AUC unchanged (results/elite_frac_sweep.json).
  * The VELOCITY UPDATE is not the differential cause either, and this needs no
    new run to establish: +APSOLL and Full Hybrid both call run_hybrid_orpsoc,
    so they share the velocity update and the phase schedule exactly. +APSOLL
    adapts at fold 6, Full Hybrid at fold 7. A term common to both cannot
    explain a difference between them.

TWO SURVIVING SUSPECTS, both visible in the saved run
─────────────────────────────────────────────────────
1. WARM START ON NON-TRIGGER FOLDS.
   elite_frac only governs the `hmm_trigger=True` branch, and the trigger fires
   once (fold 4). On folds 5-7 the other branch runs:
       if not hmm_trigger: particles[0]["pos"] = ws
   so the carried old-regime gbest is re-seeded every fold after the break --
   exactly the folds the elite_frac sweep never touched.

2. THE PHASE 2 BURST IS THROTTLED AT THE CRITICAL FOLD.
       drift_strength = 1.0 if p_trans is None else clip(p_trans, 0, 1)
       cr_target = cr_low + (cr_high - cr_low) * drift_strength
       w_target  = w_min  + (w_max  - w_min)  * drift_strength
   +APSOLL passes p_trans=None and always gets a FULL burst. Full Hybrid passes
   the HMM's p_trans, which at fold 4 is 0.6902 -- so it receives 69% of the
   burst at the one fold where it most needs to move. Measured, not assumed.

DESIGN — 2x2 factorial, v2_regime_switch, 30 seeds
──────────────────────────────────────────────────
             burst=scaled (p_trans)      burst=full (p_trans=None)
  warm=on    CONTROL (must give fold 7)  isolates the burst throttle
  warm=off   isolates the warm anchor    both interventions together

No engine changes: "full burst" is obtained by passing p_trans=None, which is
precisely how +APSOLL already gets one, and "warm off" by passing
warm_start_pos=None. So this measures the existing code paths rather than a
modified algorithm.

PRIMARY OUTCOME (pre-registered): adaptation_fold = first POST fold whose mean
recall of the post-switch signals reaches RECALL_THRESH. Lower is better;
control must reproduce 7 or the harness is invalid. AUC is the guard -- earlier
adaptation that costs accuracy is not a win.

Run:  ORPSOC_N_JOBS=30 python experiments/lag_factorial.py
"""
import os
import sys
import json
import time
import pickle
import itertools

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

from orpsoc_runner import pin_threads, default_workers, provenance, CheckpointStore
pin_threads(1)

from joblib import Parallel, delayed
from orpsoc_utils import (walk_forward_folds, classify_folds,
                          run_hybrid_orpsoc, AdaptiveRegimeThreshold)

_s7 = open("step7_ablation.py").read().split("#  MASTER ABLATION LOOP")[0]
_ns = {}
exec(compile(_s7, "step7_ablation<head>", "exec"), _ns)
get_hmm_trigger = _ns["get_hmm_trigger"]

N_SEEDS, MAX_ITER, N_PARTICLES, N_SPLITS = 30, 60, 20, 8
THETA, GAP, MIN_TRAIN = 0.5, 5, 150
RECALL_THRESH = 1.0
LEVEL = "v2_regime_switch"
SWITCH = 500

# (warm_start, full_burst)
CELLS = [(True, False),    # control -- current Full Hybrid
         (True, True),     # burst throttle removed
         (False, False),   # warm anchor removed
         (False, True)]    # both
NAME = {(True, False): "warm=on  burst=scaled  [CONTROL]",
        (True, True):  "warm=on  burst=FULL",
        (False, False): "warm=off burst=scaled",
        (False, True):  "warm=off burst=FULL"}

RESULTS_PATH = "results/lag_factorial.json"


def run_one_seed(seed, warm_on, full_burst):
    with open(f"data/{LEVEL}.pkl", "rb") as f:
        d = pickle.load(f)
    X, y = d["X"], d["y"]
    feat = list(X.columns)
    folds = walk_forward_folds(X, y, n_splits=N_SPLITS, gap=GAP,
                               min_train=MIN_TRAIN)
    phases = [p["phase"] for p in classify_folds(folds, SWITCH)]
    r1 = [c for c in feat if c in ["signal_0", "signal_1", "signal_2"]]
    r2 = [c for c in feat if c in ["signal_3", "signal_4"]]

    hmm = AdaptiveRegimeThreshold(method="percentile", lookback=50,
                                  percentile_k=85.0)
    warm = None
    out = {"fold_aucs": [], "fold_r2_hits": [], "fold_r1_hits": [],
           "n_sel": [], "fold_triggered": [], "fold_p_trans": [],
           "_phases": phases}

    for fi, (X_tr, y_tr, X_te, y_te, _) in enumerate(folds):
        if len(y_te.unique()) < 2:
            continue
        trig, p_trans, _ = get_hmm_trigger(X_tr, feat_name=feat[0],
                                           threshold_obj=hmm)
        r = run_hybrid_orpsoc(
            X_tr, y_tr, X_te, y_te,
            hmm_trigger=trig,
            warm_start_pos=(warm if warm_on else None),
            # p_trans=None => drift_strength 1.0 => FULL burst, exactly how
            # +APSOLL already receives one.
            p_trans=(None if full_burst else p_trans),
            feat_names=feat, seed=seed + fi * 1000,
            n_particles=N_PARTICLES, max_iter=MAX_ITER, min_f=3, theta=THETA,
            cr_low=0.3, cr_high=0.8, w_max=0.9, w_min=0.4,
            N_explore=max(5, MAX_ITER // 4), lam=0.1)
        warm = r["gbest_pos"]
        sel = set(r["selected"])
        out["fold_aucs"].append(r["auc"])
        out["n_sel"].append(r["n_sel"])
        out["fold_r1_hits"].append(len(sel & set(r1)))
        out["fold_r2_hits"].append(len(sel & set(r2)))
        out["fold_triggered"].append(bool(trig))
        out["fold_p_trans"].append(float(p_trans))
    return out


def adaptation_fold(seeds):
    ph = seeds[0]["_phases"]
    for i in [j for j, p in enumerate(ph) if p == "post"]:
        v = [s["fold_r2_hits"][i] for s in seeds if len(s["fold_r2_hits"]) > i]
        if v and np.mean(v) >= RECALL_THRESH:
            return i
    return None


def main():
    n_jobs = default_workers()
    cfg = {"n_seeds": N_SEEDS, "max_iter": MAX_ITER, "n_particles": N_PARTICLES,
           "n_splits": N_SPLITS, "theta": THETA, "level": LEVEL,
           "cells": [list(c) for c in CELLS], "recall_thresh": RECALL_THRESH}
    prov = provenance(cfg, ["orpsoc_utils.py", "experiments/lag_factorial.py"])
    store = CheckpointStore("results/checkpoints", "lag_factorial", prov)

    print("=" * 78)
    print("  ADAPTATION-LAG FACTORIAL  (warm start x burst strength)")
    print(f"  level={LEVEL}  seeds={N_SEEDS}  provenance={prov['hash']}")
    print(f"  workers={n_jobs}", flush=True)

    results, t0 = {}, time.time()
    for warm_on, full_burst in CELLS:
        tag = f"warm{int(warm_on)}|burst{int(full_burst)}"
        units = [f"{tag}|s{s}" for s in range(N_SEEDS)]
        print(f"\n── {NAME[(warm_on, full_burst)]}\n   {store.summary(units)}",
              flush=True)
        todo = [s for s in range(N_SEEDS) if store.load(f"{tag}|s{s}") is None]
        if todo:
            fresh = Parallel(n_jobs=n_jobs)(
                delayed(run_one_seed)(s, warm_on, full_burst) for s in todo)
            for s, r in zip(todo, fresh):
                store.save(f"{tag}|s{s}", r)
        seeds = [store.load(f"{tag}|s{s}") for s in range(N_SEEDS)]
        results[tag] = seeds
        print(f"     AUC={np.mean([np.mean(s['fold_aucs']) for s in seeds]):.4f}  "
              f"k={np.mean([np.mean(s['n_sel']) for s in seeds]):4.1f}  "
              f"adaptation_fold={adaptation_fold(seeds)}", flush=True)

    with open(RESULTS_PATH, "w") as f:
        json.dump({"config": cfg, "provenance": prov, "results": results}, f)
    print(f"\nSaved: {RESULTS_PATH}  ({time.time()-t0:.0f}s)")
    report(results)


def report(results):
    from scipy import stats
    ctrl = "warm1|burst0"
    ph = results[ctrl][0]["_phases"]
    print()
    print("=" * 88)
    print("  ADAPTATION LAG FACTORIAL")
    print(f"  phases={ph}   detector fires at fold 4 (earliest causally possible)")
    print("=" * 88)
    print(f"  {'cell':<34}{'AUC':>9}{'d vs ctrl':>11}{'p':>9}{'k':>7}{'adapt fold':>12}")
    print("  " + "-" * 82)
    base = np.array([np.mean(s["fold_aucs"]) for s in results[ctrl]])
    for warm_on, full_burst in CELLS:
        tag = f"warm{int(warm_on)}|burst{int(full_burst)}"
        a = np.array([np.mean(s["fold_aucs"]) for s in results[tag]])
        if tag == ctrl:
            dtxt, ptxt = f"{'(control)':>11}", f"{'-':>9}"
        else:
            try:
                ptxt = f"{stats.wilcoxon(a, base).pvalue:>9.4f}"
            except Exception:
                ptxt = f"{'n/a':>9}"
            dtxt = f"{a.mean()-base.mean():>+11.4f}"
        k = np.mean([np.mean(s["n_sel"]) for s in results[tag]])
        print(f"  {NAME[(warm_on, full_burst)]:<34}{a.mean():>9.4f}{dtxt}{ptxt}"
              f"{k:>7.1f}{str(adaptation_fold(results[tag])):>12}")

    print()
    print("  per-fold recall of the POST-switch signals (r2):")
    print(f"    {'cell':<34}" + "".join(f"f{i}".rjust(7) for i in range(len(ph))))
    for warm_on, full_burst in CELLS:
        tag = f"warm{int(warm_on)}|burst{int(full_burst)}"
        tr = [np.mean([s["fold_r2_hits"][i] for s in results[tag]
                       if len(s["fold_r2_hits"]) > i]) for i in range(len(ph))]
        print(f"    {NAME[(warm_on, full_burst)]:<34}" +
              "".join(f"{v:>7.2f}" for v in tr))

    print()
    print("  HOW TO READ THIS")
    print("    control must give adaptation_fold=7, else the harness is invalid.")
    print("    If 'burst=FULL' alone moves it earlier -> the p_trans throttle is")
    print("      the cause (fold 4 currently gets 69% of the burst).")
    print("    If 'warm=off' alone moves it -> the per-fold re-seeding of the old")
    print("      gbest on NON-trigger folds is the cause.")
    print("    If only the both-cell moves it -> the two reinforce each other and")
    print("      neither is sufficient alone.")
    print("    If nothing moves it -> both suspects are cleared and the lag is in")
    print("      the Phase 2 -> 3 exit condition or the fitness landscape itself.")


if __name__ == "__main__":
    main()
