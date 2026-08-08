"""
Gated memory: keep population memory while the market is calm, quarantine it
after a detected break.

THE TRADE-OFF, both halves measured rather than assumed
───────────────────────────────────────────────────────
  PURE WARM  (current Full Hybrid)
      + stable subsets: mean Jaccard 0.416 vs 0.168 for cold conditions (2.48x)
      - adapts at fold 7, three folds after the detector correctly fires at 4
      - ratchets k downward every fold (9.4 vs OrPSOC's 11.1) because the size
        penalty only ever pushes down and nothing re-adds a dropped feature

  PURE COLD  (warm_start_pos=None everywhere)
      + adapts at fold 6, and is the only intervention with a significant AUC
        gain: +0.0199, p=0.0004 (results/lag_factorial.json)
      - churns: every fold re-rolls the dice, so the selected set flickers even
        when nothing about the market has changed. In a portfolio setting that
        is turnover paid for nothing.

Neither dominates. Hence GATED MEMORY: state-dependent, keyed to the detector.

  calm fold, no recent trigger  -> warm start ON   (lock the subset, no churn)
  trigger fold                  -> warm start ON   (elites + importance reinit;
                                                    the "smart restart" already
                                                    implemented -- and note the
                                                    elites COME FROM the warm
                                                    start, so it must stay on)
  within QUARANTINE folds after -> warm start OFF  (the fix: stop re-seeding the
                                                    stale pre-switch gbest on the
                                                    non-trigger folds 5-7)

Why the quarantine and not the trigger fold: elite_frac governs only the
`hmm_trigger=True` branch and changing it does nothing (240 units, adaptation
pinned at fold 7). The damage is in the OTHER branch,
`if not hmm_trigger: particles[0]["pos"] = ws`, which re-seeds the pre-switch
solution on every fold after the break.

ARMS (v2_regime_switch, 30 seeds)
  warm_always     control -- current Full Hybrid          expect fold 7, high J
  warm_never      pure cold                               expect fold 6, low J
  gated_q1        quarantine 1 fold after a trigger
  gated_q2        quarantine 2 folds after a trigger

PRIMARY OUTCOMES, both pre-registered — this is explicitly a two-objective test:
  adaptation_fold  first POST fold whose mean r2 recall reaches RECALL_THRESH
  pre_break_J      mean Jaccard over fold-pairs BEFORE the straddle fold
                   (the churn measure; higher = more stable while calm)
Gated memory succeeds only if it matches COLD on adaptation_fold AND WARM on
pre_break_J. Winning one at the cost of the other is not a win.
AUC is reported as a guard.

Run:  ORPSOC_N_JOBS=30 python experiments/gated_memory.py
"""
import os
import sys
import json
import time
import pickle

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
LEVEL, SWITCH = "v2_regime_switch", 500

# arm -> quarantine length. None = never warm (pure cold); -1 = always warm.
ARMS = {"warm_always": -1, "warm_never": None, "gated_q1": 1, "gated_q2": 2}
ORDER = ["warm_always", "warm_never", "gated_q1", "gated_q2"]
RESULTS_PATH = "results/gated_memory.json"


def use_warm(arm, folds_since_trigger, triggered_now):
    """State-dependent memory gate. Returns True if warm start should be used."""
    q = ARMS[arm]
    if q == -1:
        return True                      # always warm (control)
    if q is None:
        return False                     # never warm (pure cold)
    if triggered_now:
        return True                      # elites come FROM the warm start
    if folds_since_trigger is not None and folds_since_trigger <= q:
        return False                     # QUARANTINE
    return True                          # calm -> lock the subset


def jaccard_trace(sel_sets):
    out = []
    for i in range(len(sel_sets) - 1):
        a, b = set(sel_sets[i]), set(sel_sets[i + 1])
        u = a | b
        out.append(len(a & b) / len(u) if u else 1.0)
    return out


def run_one_seed(seed, arm):
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
    warm, since = None, None
    out = {"fold_aucs": [], "fold_r2_hits": [], "fold_r1_hits": [],
           "n_sel": [], "fold_selected": [], "fold_triggered": [],
           "fold_warm_used": [], "_phases": phases}

    for fi, (X_tr, y_tr, X_te, y_te, _) in enumerate(folds):
        if len(y_te.unique()) < 2:
            continue
        trig, p_trans, _ = get_hmm_trigger(X_tr, feat_name=feat[0],
                                           threshold_obj=hmm)
        w_on = use_warm(arm, since, trig)
        r = run_hybrid_orpsoc(
            X_tr, y_tr, X_te, y_te, hmm_trigger=trig,
            warm_start_pos=(warm if w_on else None), p_trans=p_trans,
            feat_names=feat, seed=seed + fi * 1000,
            n_particles=N_PARTICLES, max_iter=MAX_ITER, min_f=3, theta=THETA,
            cr_low=0.3, cr_high=0.8, w_max=0.9, w_min=0.4,
            N_explore=max(5, MAX_ITER // 4), lam=0.1)
        # gbest is always carried forward; the GATE decides whether it is USED.
        warm = r["gbest_pos"]
        since = 0 if trig else (None if since is None else since + 1)

        sel = set(r["selected"])
        out["fold_aucs"].append(r["auc"])
        out["n_sel"].append(r["n_sel"])
        out["fold_selected"].append(sorted(sel))
        out["fold_r1_hits"].append(len(sel & set(r1)))
        out["fold_r2_hits"].append(len(sel & set(r2)))
        out["fold_triggered"].append(bool(trig))
        out["fold_warm_used"].append(bool(w_on))
    out["jaccard"] = jaccard_trace(out["fold_selected"])
    return out


def adaptation_fold(seeds):
    ph = seeds[0]["_phases"]
    for i in [j for j, p in enumerate(ph) if p == "post"]:
        v = [s["fold_r2_hits"][i] for s in seeds if len(s["fold_r2_hits"]) > i]
        if v and np.mean(v) >= RECALL_THRESH:
            return i
    return None


def pre_break_jaccard(seeds):
    """Mean Jaccard over fold-pairs strictly BEFORE the straddle fold."""
    ph = seeds[0]["_phases"]
    strad = [i for i, p in enumerate(ph) if p == "straddle"]
    cut = strad[0] if strad else len(ph) // 2
    vals = [np.mean(s["jaccard"][:cut]) for s in seeds
            if len(s["jaccard"]) >= cut and cut > 0]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    n_jobs = default_workers()
    cfg = {"n_seeds": N_SEEDS, "max_iter": MAX_ITER, "n_particles": N_PARTICLES,
           "n_splits": N_SPLITS, "theta": THETA, "level": LEVEL,
           "arms": ARMS, "recall_thresh": RECALL_THRESH}
    prov = provenance(cfg, ["orpsoc_utils.py", "experiments/gated_memory.py"])
    store = CheckpointStore("results/checkpoints", "gated_memory", prov)

    print("=" * 82)
    print("  GATED MEMORY — state-dependent warm start")
    print(f"  level={LEVEL} seeds={N_SEEDS} provenance={prov['hash']} workers={n_jobs}")
    print("=" * 82, flush=True)

    results, t0 = {}, time.time()
    for arm in ORDER:
        units = [f"{arm}|s{s}" for s in range(N_SEEDS)]
        print(f"\n── {arm}\n   {store.summary(units)}", flush=True)
        todo = [s for s in range(N_SEEDS) if store.load(f"{arm}|s{s}") is None]
        if todo:
            fresh = Parallel(n_jobs=n_jobs)(
                delayed(run_one_seed)(s, arm) for s in todo)
            for s, r in zip(todo, fresh):
                store.save(f"{arm}|s{s}", r)
        seeds = [store.load(f"{arm}|s{s}") for s in range(N_SEEDS)]
        results[arm] = seeds
        print(f"     AUC={np.mean([np.mean(s['fold_aucs']) for s in seeds]):.4f}  "
              f"k={np.mean([np.mean(s['n_sel']) for s in seeds]):4.1f}  "
              f"adapt={adaptation_fold(seeds)}  "
              f"preJ={pre_break_jaccard(seeds):.3f}", flush=True)

    with open(RESULTS_PATH, "w") as f:
        json.dump({"config": cfg, "provenance": prov, "results": results}, f)
    print(f"\nSaved: {RESULTS_PATH}  ({time.time()-t0:.0f}s)")
    report(results)


def report(results):
    from scipy import stats
    ctrl = "warm_always"
    ph = results[ctrl][0]["_phases"]
    print()
    print("=" * 94)
    print("  GATED MEMORY — two objectives at once")
    print(f"  phases={ph}   detector fires at fold 4")
    print("  GOAL: match COLD on 'adapt' AND WARM on 'preJ'. One without the")
    print("        other is not a win.")
    print("=" * 94)
    base = np.array([np.mean(s["fold_aucs"]) for s in results[ctrl]])
    print(f"  {'arm':<14}{'AUC':>9}{'d vs ctrl':>11}{'p':>9}{'k':>7}"
          f"{'adapt fold':>12}{'pre-break J':>13}")
    print("  " + "-" * 75)
    for arm in ORDER:
        a = np.array([np.mean(s["fold_aucs"]) for s in results[arm]])
        if arm == ctrl:
            dtxt, ptxt = f"{'(control)':>11}", f"{'-':>9}"
        else:
            try:
                ptxt = f"{stats.wilcoxon(a, base).pvalue:>9.4f}"
            except Exception:
                ptxt = f"{'n/a':>9}"
            dtxt = f"{a.mean()-base.mean():>+11.4f}"
        k = np.mean([np.mean(s["n_sel"]) for s in results[arm]])
        print(f"  {arm:<14}{a.mean():>9.4f}{dtxt}{ptxt}{k:>7.1f}"
              f"{str(adaptation_fold(results[arm])):>12}"
              f"{pre_break_jaccard(results[arm]):>13.3f}")

    print()
    print("  per-fold Jaccard (churn while calm = low values on the left):")
    n = len(ph) - 1
    print(f"    {'arm':<14}" + "".join(f"{i}-{i+1}".rjust(8) for i in range(n)))
    for arm in ORDER:
        tr = [np.mean([s["jaccard"][i] for s in results[arm]
                       if len(s["jaccard"]) > i]) for i in range(n)]
        print(f"    {arm:<14}" + "".join(f"{v:>8.3f}" for v in tr))

    print()
    print("  per-fold r2 recall (adaptation speed):")
    print(f"    {'arm':<14}" + "".join(f"f{i}".rjust(8) for i in range(len(ph))))
    for arm in ORDER:
        tr = [np.mean([s["fold_r2_hits"][i] for s in results[arm]
                       if len(s["fold_r2_hits"]) > i]) for i in range(len(ph))]
        print(f"    {arm:<14}" + "".join(f"{v:>8.2f}" for v in tr))

    print()
    print("  warm start actually USED, per fold (gate behaviour check):")
    print(f"    {'arm':<14}" + "".join(f"f{i}".rjust(8) for i in range(len(ph))))
    for arm in ORDER:
        tr = [np.mean([s["fold_warm_used"][i] for s in results[arm]
                       if len(s["fold_warm_used"]) > i]) for i in range(len(ph))]
        print(f"    {arm:<14}" + "".join(f"{v:>8.2f}" for v in tr))


if __name__ == "__main__":
    main()
