"""
How much population memory is too much at a regime break?

THE QUESTION (and why it is NOT a rejection of the supervisor's advice)
───────────────────────────────────────────────────────────────────────
The supervisor proposed two changes, both already implemented in
orpsoc_utils.run_hybrid_orpsoc:

  #2 importance-guided reinit  — on trigger, the non-elite particles are seeded
     from windowed_feature_importance() on the MOST RECENT training window
     rather than a blind orthogonal draw. This demonstrably works: Full Hybrid
     beats the no-imp variant by +0.0717 (L1) and +0.0434 (L2), both p<1e-4.

  #3 elite preservation / partial restart — on trigger, elite_frac of the swarm
     is seeded from the carried gbest and reinjected alongside fresh particles.

Neither is in question here. What is unspecified is HOW MANY elites. With the
current settings the arithmetic is counter-intuitive:

    n_particles = 20, elite_frac = 0.2

    hmm_trigger = False  ->  particles[0] only          =  1 of 20 from old gbest
    hmm_trigger = True   ->  elite_k = round(0.2*20)    =  4 of 20 from old gbest

i.e. DETECTING a regime change QUADRUPLES the old-regime material in the swarm,
and each elite is installed with its best_fit evaluated, so they act as strong
attractors. Measured consequence (results/tier_a.json): warm-start conditions
are 2.48x more anchored to the previous fold's subset than cold ones, the HMM
fires correctly at fold 4 in 30/30 seeds, and yet Full Hybrid's recall of the
post-switch signals does not rise until fold 7 — three folds after detection.

So this sweep asks a parameter question INSIDE the partial-restart framework:
how much memory should survive a detected break? elite_frac = 0.0 is simply the
endpoint of the sweep (importance-reinit still on), not an argument against
population memory.

PRIMARY OUTCOME — declared before running:
    adaptation_fold = the first POST-switch fold at which mean recall of the
    post-switch signals (fold_r2_hits) exceeds RECALL_THRESH. Lower is better.
    Baseline for comparison: standard OrPSOC (cold) adapts at fold 6, Full
    Hybrid at fold 7. The hypothesis is that adaptation_fold falls as
    elite_frac falls. AUC is reported as a guard: a faster adaptation that
    costs accuracy is not a win.

Run:  ORPSOC_N_JOBS=30 python experiments/elite_frac_sweep.py
Writes results/elite_frac_sweep.json
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
from orpsoc_utils import (
    walk_forward_folds, classify_folds, run_hybrid_orpsoc,
    AdaptiveRegimeThreshold,
)

# step7_ablation.py is a script and now refuses to be imported; exec its head to
# reuse get_hmm_trigger as a single source of truth (same idiom as apsoll_sweep).
_s7 = open("step7_ablation.py").read().split("#  MASTER ABLATION LOOP")[0]
_ns = {}
exec(compile(_s7, "step7_ablation<head>", "exec"), _ns)
get_hmm_trigger = _ns["get_hmm_trigger"]

N_SEEDS, MAX_ITER, N_PARTICLES, N_SPLITS = 30, 60, 20, 8
THETA, GAP, MIN_TRAIN = 0.5, 5, 150
RECALL_THRESH = 1.0          # pre-registered: "has found >=1 new signal on average"

LEVELS = ["v2_drift", "v2_regime_switch"]
SWITCH_INDEX = {"v2_drift": None, "v2_regime_switch": 500}
CONDITIONS = ["full_hybrid", "full_hybrid_noimp"]   # only these carry elites
GRID = [0.2, 0.1, 0.05, 0.0]                        # 0.2 = current default

RESULTS_PATH = "results/elite_frac_sweep.json"


def run_one_seed(level_key, seed, elite_frac):
    with open(f"data/{level_key}.pkl", "rb") as f:
        data = pickle.load(f)
    X, y = data["X"], data["y"]
    feat_names = list(X.columns)
    folds = walk_forward_folds(X, y, n_splits=N_SPLITS, gap=GAP,
                               min_train=MIN_TRAIN)
    fold_phase = classify_folds(folds, SWITCH_INDEX.get(level_key))

    out = {c: {"fold_aucs": [], "fold_r2_hits": [], "fold_r1_hits": [],
               "n_sel": [], "fold_selected": []} for c in CONDITIONS}
    out["_phases"] = [p["phase"] for p in fold_phase]

    hmm = AdaptiveRegimeThreshold(method="percentile", lookback=50,
                                  percentile_k=85.0)
    warm = {c: None for c in CONDITIONS}

    # Ground truth, copied verbatim from step7_ablation.py:751-752 so the recall
    # numbers here are directly comparable with the paper run's fold_r2_hits.
    r1 = [c for c in feat_names if c in ["signal_0", "signal_1", "signal_2"]]
    r2 = [c for c in feat_names if c in ["signal_3", "signal_4"]]

    for fi, (X_tr, y_tr, X_te, y_te, _) in enumerate(folds):
        if len(y_te.unique()) < 2:
            continue
        pso_kw = dict(feat_names=feat_names, seed=seed + fi * 1000,
                      n_particles=N_PARTICLES, max_iter=MAX_ITER,
                      min_f=3, theta=THETA, cr_low=0.3, cr_high=0.8,
                      w_max=0.9, w_min=0.4, N_explore=max(5, MAX_ITER // 4),
                      lam=0.1, elite_frac=elite_frac)
        trig, p_trans, _ = get_hmm_trigger(X_tr, feat_name=feat_names[0],
                                           threshold_obj=hmm)
        for c in CONDITIONS:
            r = run_hybrid_orpsoc(
                X_tr, y_tr, X_te, y_te, hmm_trigger=trig,
                warm_start_pos=warm[c], p_trans=p_trans,
                use_importance_reinit=(c == "full_hybrid"), **pso_kw)
            warm[c] = r["gbest_pos"]
            sel = set(r["selected"])
            out[c]["fold_aucs"].append(r["auc"])
            out[c]["n_sel"].append(r["n_sel"])
            out[c]["fold_r1_hits"].append(len(sel & set(r1)))
            out[c]["fold_r2_hits"].append(len(sel & set(r2)))
            out[c]["fold_selected"].append(sorted(sel))
    return out


def adaptation_fold(seeds, cond, phases):
    """First POST fold whose mean r2 recall exceeds RECALL_THRESH, else None."""
    post = [i for i, p in enumerate(phases) if p == "post"]
    for i in post:
        vals = [s[cond]["fold_r2_hits"][i] for s in seeds
                if len(s[cond]["fold_r2_hits"]) > i]
        if vals and np.mean(vals) >= RECALL_THRESH:
            return i
    return None


def main():
    n_jobs = default_workers()
    cfg = {"n_seeds": N_SEEDS, "max_iter": MAX_ITER, "n_particles": N_PARTICLES,
           "n_splits": N_SPLITS, "theta": THETA, "levels": LEVELS,
           "conditions": CONDITIONS, "grid": GRID,
           "recall_thresh": RECALL_THRESH}
    prov = provenance(cfg, ["orpsoc_utils.py", "experiments/elite_frac_sweep.py"])

    print("=" * 76)
    print("  ELITE_FRAC SWEEP — how much population memory survives a break?")
    print("=" * 76)
    print(f"  seeds={N_SEEDS} iters={MAX_ITER} particles={N_PARTICLES} "
          f"folds={N_SPLITS}  grid={GRID}")
    print(f"  elite particles at trigger: " +
          ", ".join(f"{e}->{max(1, round(e*N_PARTICLES)) if e > 0 else 0}"
                    for e in GRID))
    print(f"  workers={n_jobs}  provenance={prov['hash']}", flush=True)

    store = CheckpointStore("results/checkpoints", "elite_frac", prov)
    results, t0 = {}, time.time()

    for lvl, ef in itertools.product(LEVELS, GRID):
        tag = f"{lvl}|ef{ef}"
        units = [f"{tag}|s{s}" for s in range(N_SEEDS)]
        print(f"\n── {tag}\n   {store.summary(units)}", flush=True)
        todo = [s for s in range(N_SEEDS) if store.load(f"{tag}|s{s}") is None]
        if todo:
            fresh = Parallel(n_jobs=n_jobs)(
                delayed(run_one_seed)(lvl, s, ef) for s in todo)
            for s, r in zip(todo, fresh):
                store.save(f"{tag}|s{s}", r)
        seeds = [store.load(f"{tag}|s{s}") for s in range(N_SEEDS)]
        results[tag] = seeds
        ph = seeds[0]["_phases"]
        for c in CONDITIONS:
            auc = np.mean([np.mean(s[c]["fold_aucs"]) for s in seeds])
            af = adaptation_fold(seeds, c, ph)
            k = np.mean([np.mean(s[c]["n_sel"]) for s in seeds])
            print(f"     {c:<20} AUC={auc:.4f}  k={k:4.1f}  "
                  f"adaptation_fold={af}", flush=True)

    with open(RESULTS_PATH, "w") as f:
        json.dump({"config": cfg, "provenance": prov, "results": results}, f)
    print(f"\nSaved: {RESULTS_PATH}  ({time.time()-t0:.0f}s)")
    report(results)


def report(results):
    from scipy import stats
    print()
    print("=" * 76)
    print("  RESULT — does adaptation arrive sooner as elite_frac falls?")
    print("=" * 76)
    for lvl in LEVELS:
        ctrl = f"{lvl}|ef0.2"
        if ctrl not in results:
            continue
        ph = results[ctrl][0]["_phases"]
        print(f"\n  {lvl}   phases={ph}")
        print(f"    {'elite_frac':<12}{'condition':<20}{'AUC':>9}{'d vs 0.2':>10}"
              f"{'p':>9}{'k':>7}{'adapt fold':>12}")
        print("    " + "-" * 79)
        for ef in GRID:
            tag = f"{lvl}|ef{ef}"
            if tag not in results:
                continue
            for c in CONDITIONS:
                a = np.array([np.mean(s[c]["fold_aucs"]) for s in results[tag]])
                b = np.array([np.mean(s[c]["fold_aucs"]) for s in results[ctrl]])
                if tag == ctrl:
                    dtxt, ptxt = f"{'(control)':>10}", f"{'-':>9}"
                else:
                    try:
                        p = stats.wilcoxon(a, b).pvalue
                        ptxt = f"{p:>9.4f}"
                    except Exception:
                        ptxt = f"{'n/a':>9}"
                    dtxt = f"{a.mean()-b.mean():>+10.4f}"
                k = np.mean([np.mean(s[c]["n_sel"]) for s in results[tag]])
                af = adaptation_fold(results[tag], c, ph)
                print(f"    {ef:<12}{c:<20}{a.mean():>9.4f}{dtxt}{ptxt}"
                      f"{k:>7.1f}{str(af):>12}")
    print()
    print("  READING THIS: adaptation_fold is the primary outcome (lower = the")
    print("  swarm finds the new regime's signals sooner). AUC is the guard --")
    print("  faster adaptation that loses accuracy is not an improvement. If")
    print("  adaptation_fold does not move, the elites are exonerated and the")
    print("  lag lives elsewhere (velocity update, or the Phase 2 burst itself).")


if __name__ == "__main__":
    main()
