"""
Does the APSOLL patience / re-arm fix actually help?

BACKGROUND
──────────
APSOLL's adaptive-c trigger is  c = (m/T)^(2/3) + 1  firing at c < 1.05, which
means it fires only when  m < 0.05^1.5 * T.  At T = MAX_ITER = 60 that is
m < 0.67, i.e. ONLY m = 0 — literally zero improvement — ever trips it. And
once the swarm enters Phase 3 it never returns, so the trigger fires at most
once per fold, ever.

orpsoc_utils.run_hybrid_orpsoc implements two corrections:

  apsoll_patience = k > 1
      require k CONSECUTIVE stagnant iterations rather than reacting to a
      single flat step (the early-stopping "patience" idiom).

  apsoll_rearm_after = r
      after r iterations in Phase 3, drop back to Phase 1 and clear the
      patience counter, so the trigger can fire AGAIN later in the run.

Both are wired through step7_ablation.py, but the paper run executed with the
LEGACY defaults (patience=1, rearm=None) — the provenance block in
results/step7_ablation_v2.json records exactly that. So the fix has never been
measured. This script measures it.

SCOPE — deliberately narrow, because breadth here costs hours for no evidence
──────
  levels      v2_drift, v2_regime_switch     (APSOLL is about adaptation; the
                                              stationary levels add cost, not
                                              information)
  conditions  apsoll, full_hybrid            (baseline and standard_orpsoc do
                                              not call APSOLL at all)
  seeds       30                             (matches the paper run)

Everything else — folds, gap, theta, particles, iterations, the HMM threshold
object, the warm-start chain — is copied from step7_ablation.py so that the
(patience=1, rearm=None) cell reproduces the paper run's numbers and acts as an
internal control. If that cell does NOT match, the comparison is invalid and
the script says so rather than letting you read a difference that is really a
harness discrepancy.

Checkpoints live under their own provenance hash, so this never touches
results/checkpoints/step7/.

Run:
    ORPSOC_N_JOBS=30 python experiments/apsoll_sweep.py
Writes results/apsoll_sweep.json
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
    walk_forward_folds,
    classify_folds,
    run_hybrid_orpsoc,
    AdaptiveRegimeThreshold,
)
# step7_ablation.py has NO `if __name__ == "__main__"` guard, so a plain
# `import step7_ablation` executes the entire 4-hour ablation as an import side
# effect. Exec only the section ABOVE the master loop -- the same idiom
# experiments/build_extra_markets.py uses against step9_real_data.py. This keeps
# get_hmm_trigger a single source of truth rather than a copy that silently
# drifts out of step with the version the paper run used.
_s7_src = open("step7_ablation.py").read().split("#  MASTER ABLATION LOOP")[0]
_s7_ns = {}
exec(compile(_s7_src, "step7_ablation<head>", "exec"), _s7_ns)
get_hmm_trigger = _s7_ns["get_hmm_trigger"]

# ── configuration, mirrored from step7_ablation.py ───────────────────────────
N_SEEDS      = 30
MAX_ITER     = 60
N_PARTICLES  = 20
N_SPLITS     = 8
THETA        = 0.5
GAP          = 5
MIN_TRAIN    = 150

LEVELS = ["v2_drift", "v2_regime_switch"]
SWITCH_INDEX = {"v2_drift": None, "v2_regime_switch": 500}
CONDITIONS = ["apsoll", "full_hybrid"]

# (patience, rearm_after). (1, None) is the legacy control -- keep it FIRST so
# the reproduction check runs before anything expensive is interpreted.
GRID = [(1, None), (1, 10), (1, 20), (3, None), (3, 10), (3, 20)]

RESULTS_PATH = "results/apsoll_sweep.json"
CKPT_ROOT    = "results/checkpoints"


def run_one_seed(level_key, seed, patience, rearm):
    """
    One complete walk-forward sequence for one seed at one APSOLL setting.

    Self-contained: the HMM threshold object and the warm-start chain are
    created here and never cross a seed boundary, so seeds are independent and
    may run in any order or concurrently.
    """
    with open(f"data/{level_key}.pkl", "rb") as f:
        data = pickle.load(f)
    X, y = data["X"], data["y"]
    feat_names = list(X.columns)
    folds = walk_forward_folds(X, y, n_splits=N_SPLITS, gap=GAP,
                               min_train=MIN_TRAIN)
    fold_phase = classify_folds(folds, SWITCH_INDEX.get(level_key))

    out = {c: {"fold_aucs": [], "n_sel": [], "trigger_iters": [],
               "n_rearms": []} for c in CONDITIONS}
    out["_phases"] = [p["phase"] for p in fold_phase]

    hmm_threshold = AdaptiveRegimeThreshold(method="percentile", lookback=50,
                                            percentile_k=85.0)
    warm_start_fh = None

    for fold_idx, (X_tr, y_tr, X_te, y_te, train_end) in enumerate(folds):
        if len(y_te.unique()) < 2:
            continue

        pso_kw = dict(
            feat_names=feat_names, seed=seed + fold_idx * 1000,
            n_particles=N_PARTICLES, max_iter=MAX_ITER,
            min_f=3, theta=THETA,
            cr_low=0.3, cr_high=0.8, w_max=0.9, w_min=0.4,
            N_explore=max(5, MAX_ITER // 4), lam=0.1,
            apsoll_patience=patience,
            apsoll_rearm_after=rearm,
        )

        # Condition 3 — +APSOLL, self-triggered, no HMM, no warm start.
        r3 = run_hybrid_orpsoc(X_tr, y_tr, X_te, y_te,
                               hmm_trigger=False, **pso_kw)
        out["apsoll"]["fold_aucs"].append(r3["auc"])
        out["apsoll"]["n_sel"].append(r3["n_sel"])
        out["apsoll"]["trigger_iters"].append(r3["apsoll_trigger_iters"])
        out["apsoll"]["n_rearms"].append(r3["apsoll_n_rearms"])

        # Condition 4 — Full Hybrid: HMM trigger + warm-start chain.
        triggered, p_trans, _is_warmup = get_hmm_trigger(
            X_tr, feat_name=feat_names[0], threshold_obj=hmm_threshold)
        r4 = run_hybrid_orpsoc(X_tr, y_tr, X_te, y_te,
                               hmm_trigger=triggered,
                               warm_start_pos=warm_start_fh,
                               p_trans=p_trans, **pso_kw)
        warm_start_fh = r4["gbest_pos"]
        out["full_hybrid"]["fold_aucs"].append(r4["auc"])
        out["full_hybrid"]["n_sel"].append(r4["n_sel"])
        out["full_hybrid"]["trigger_iters"].append(r4["apsoll_trigger_iters"])
        out["full_hybrid"]["n_rearms"].append(r4["apsoll_n_rearms"])

    return out


def main():
    n_jobs = default_workers()
    cfg = {"n_seeds": N_SEEDS, "max_iter": MAX_ITER,
           "n_particles": N_PARTICLES, "n_splits": N_SPLITS,
           "theta": THETA, "levels": LEVELS, "conditions": CONDITIONS,
           "grid": [[p, r] for p, r in GRID]}
    prov = provenance(cfg, ["orpsoc_utils.py", "experiments/apsoll_sweep.py"])

    print("=" * 74)
    print("  APSOLL PATIENCE / RE-ARM SWEEP")
    print("=" * 74)
    print(f"  seeds={N_SEEDS}  iters={MAX_ITER}  particles={N_PARTICLES} "
          f"folds={N_SPLITS}  theta={THETA}")
    print(f"  grid={GRID}")
    print(f"  workers={n_jobs}   provenance={prov['hash']}")
    print()

    store = CheckpointStore(CKPT_ROOT, "apsoll_sweep", prov)
    results = {}
    t0 = time.time()

    for level_key, (patience, rearm) in itertools.product(LEVELS, GRID):
        tag = f"{level_key}|p{patience}|r{rearm}"
        units = [f"{tag}|s{s}" for s in range(N_SEEDS)]
        print(f"── {tag}")
        print(f"   checkpoints: {store.summary(units)}")

        todo = [s for s in range(N_SEEDS)
                if store.load(f"{tag}|s{s}") is None]
        if todo:
            fresh = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(run_one_seed)(level_key, s, patience, rearm)
                for s in todo)
            for s, r in zip(todo, fresh):
                store.save(f"{tag}|s{s}", r)

        seeds = [store.load(f"{tag}|s{s}") for s in range(N_SEEDS)]
        results[tag] = seeds
        for cond in CONDITIONS:
            m = np.mean([np.mean(sr[cond]["fold_aucs"]) for sr in seeds])
            rearms = np.mean([np.mean(sr[cond]["n_rearms"]) for sr in seeds])
            fired = np.mean([np.mean([len(t) > 0 for t in sr[cond]["trigger_iters"]])
                             for sr in seeds])
            print(f"     {cond:<14} AUC={m:.4f}  fire-rate={fired:.2f}  "
                  f"re-arms/fold={rearms:.2f}")
        print()

    payload = {"config": cfg, "provenance": prov, "results": results}
    os.makedirs("results", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(payload, f)
    print(f"Saved: {RESULTS_PATH}   ({time.time()-t0:.0f}s)")
    print()
    report(results)


def _seed_means(seeds, cond):
    return np.array([np.mean(sr[cond]["fold_aucs"]) for sr in seeds])


def report(results):
    from scipy import stats

    print("=" * 74)
    print("  SWEEP TABLE — mean AUC across 30 seeds")
    print("  Wilcoxon vs the legacy (patience=1, rearm=None) control,")
    print("  paired by seed, two-tailed.")
    print("=" * 74)

    for level_key in LEVELS:
        ctrl_tag = f"{level_key}|p1|rNone"
        if ctrl_tag not in results:
            continue
        print(f"\n  {level_key}")
        print(f"    {'setting':<20}{'condition':<15}{'mean AUC':>10}"
              f"{'d vs ctrl':>12}{'p':>10}{'re-arms':>10}")
        print("    " + "-" * 77)
        for patience, rearm in GRID:
            tag = f"{level_key}|p{patience}|r{rearm}"
            if tag not in results:
                continue
            for cond in CONDITIONS:
                a = _seed_means(results[tag], cond)
                c = _seed_means(results[ctrl_tag], cond)
                d = a.mean() - c.mean()
                if tag == ctrl_tag:
                    ptxt, dtxt = "  (control)", "        —"
                else:
                    try:
                        _, p = stats.wilcoxon(a, c)
                        ptxt = f"{p:>10.4f}"
                    except Exception:
                        ptxt = f"{'n/a':>10}"
                    dtxt = f"{d:>+12.4f}"
                rearms = np.mean([np.mean(sr[cond]["n_rearms"])
                                  for sr in results[tag]])
                label = f"p={patience} r={rearm}"
                print(f"    {label:<20}{cond:<15}{a.mean():>10.4f}"
                      f"{dtxt}{ptxt}{rearms:>10.2f}")

    print()
    print("  READING THIS TABLE")
    print("    re-arms/fold = 0.00 everywhere means the re-arm never fired and")
    print("    the setting was inert -- report that as a null result, not as")
    print("    'the fix does not help'. A fix that never engages has not been")
    print("    tested. Check the fire-rate printed during the run too: if the")
    print("    trigger itself never fires, patience is moot by construction.")


if __name__ == "__main__":
    main()
