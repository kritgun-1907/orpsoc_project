"""
Phase 3 RERUN with median_k  (supervisor item 3)
================================================
The original Phase 3 used `min_k` as the "fixed" fitness criterion. Replicating
the criterion bake-off over 5 independent candidate pools showed that pick was a
single-draw artefact: min_k scores 30% +- 10.5 on sector ETF and 23% +- 11.6 on
v2 -- below the CURRENT criterion on v2. `median_k` is the replicated winner on
both datasets (33% +- 10.4 and 43% +- 12.3), so the LogReg-paradox comparison is
redone with it before the narrative is treated as settled.

ARMS (identical folds, identical test windows, matched final classifier)
  baseline          all features                              <- number to beat
  filter_frozen{k}  top-k univariate |AUC-0.5|, ranked ONCE on the FIRST fold's
                    training window, NEVER updated. Causal, non-adaptive.
  filter_refit{k}   same rule, re-ranked on EACH fold's training window.
                    Causal, adaptive. Isolates the value of ADAPTATION alone.
  orpsoc_current    OrPSOC with the existing trailing-window fitness
  orpsoc_mediank    OrPSOC with median_k, the replicated winner
  oracle_test{k}    top-k univariate ranked on the TEST fold. NON-CAUSAL.
                    An upper bound on UNIVARIATE FILTERING only -- not on
                    selection in general, and never a method.

Run:  python experiments/phase3_mediank.py        (writes results/phase3_mediank.json)
"""
import os
import sys
import json
import time
import pickle

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orpsoc_runner import pin_threads
pin_threads(1)
from joblib import Parallel, delayed
import orpsoc_utils as U
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

KS = [5, 10, 20]
N_SEEDS = 5
MK = {
    "lgbm":   lambda: LGBMClassifier(n_estimators=100, num_leaves=31,
                                     learning_rate=0.1, verbosity=-1,
                                     random_state=42, n_jobs=1),
    "logreg": lambda: LogisticRegression(max_iter=200),
}
# Search budget per classifier. A LightGBM fitness fit costs ~150ms on sector
# ETF's largest folds vs ~33ms for LogReg, and median_k multiplies that by K
# inner blocks, so the LightGBM arm runs at a reduced budget. Stated openly: a
# weaker search biases AGAINST the wrapper, so an OrPSOC loss there is
# ambiguous while an OrPSOC win would be strong. The LogReg arm -- which carries
# the LogReg-paradox result -- runs at the higher budget.
BUDGET = {"logreg": dict(max_iter=40, n_particles=16, N_explore=10),
          "lgbm":   dict(max_iter=20, n_particles=10, N_explore=5)}


def fit_score(X_tr, y_tr, X_te, y_te, cols, mk):
    p = Pipeline([("i", SimpleImputer(strategy="mean")),
                  ("s", StandardScaler()), ("m", mk())])
    p.fit(X_tr[cols], y_tr)
    return roc_auc_score(y_te, p.predict_proba(X_te[cols])[:, 1])


def univariate_rank(X, y, feat):
    """Top features by |AUC - 0.5|. Deterministic, no model fitting."""
    sc = {}
    for c in feat:
        try:
            sc[c] = abs(roc_auc_score(y, X[c]) - 0.5)
        except Exception:
            sc[c] = 0.0
    return [c for c, _ in sorted(sc.items(), key=lambda kv: -kv[1])]


def unit(ds, mt, seed):
    d = pickle.load(open(f"data/{ds}.pkl", "rb"))
    X, y = d["X"], d["y"]
    feat = list(X.columns)
    folds = U.walk_forward_folds(X, y, n_splits=8, gap=5, min_train=mt)
    frozen_rank = univariate_rank(folds[0][0], folds[0][1], feat)

    out = {}
    for fi, (X_tr, y_tr, X_te, y_te, _) in enumerate(folds):
        if y_te.nunique() < 2:
            continue
        refit_rank = univariate_rank(X_tr, y_tr, feat)
        oracle_rank = univariate_rank(X_te, y_te, feat)        # NON-CAUSAL

        for m, mk in MK.items():
            out.setdefault(f"baseline|{m}", []).append(
                fit_score(X_tr, y_tr, X_te, y_te, feat, mk))
            for k in KS:
                out.setdefault(f"filter_frozen{k}|{m}", []).append(
                    fit_score(X_tr, y_tr, X_te, y_te, frozen_rank[:k], mk))
                out.setdefault(f"filter_refit{k}|{m}", []).append(
                    fit_score(X_tr, y_tr, X_te, y_te, refit_rank[:k], mk))
                out.setdefault(f"oracle_test{k}|{m}", []).append(
                    fit_score(X_tr, y_tr, X_te, y_te, oracle_rank[:k], mk))

            kw = dict(feat_names=feat, seed=seed + fi * 1000, min_f=3,
                      theta=0.5, cr_low=0.3, cr_high=0.8, w_max=0.9,
                      w_min=0.4, lam=0.1, model_factory=mk, **BUDGET[m])
            r = U.run_standard_orpsoc(X_tr, y_tr, X_te, y_te, **kw)
            out.setdefault(f"orpsoc_current|{m}", []).append(r["auc"])
            out.setdefault(f"nsel_current|{m}", []).append(r["n_sel"])
            r = U.run_standard_orpsoc(X_tr, y_tr, X_te, y_te,
                                      criterion="median_k",
                                      criterion_kwargs=dict(k_blocks=4), **kw)
            out.setdefault(f"orpsoc_mediank|{m}", []).append(r["auc"])
            out.setdefault(f"nsel_mediank|{m}", []).append(r["n_sel"])
    return ds, seed, out


if __name__ == "__main__":
    tasks = [(ds, mt, sd) for ds, mt in (("v2_regime_switch", 150),
                                         ("sector_etf", 500))
             for sd in range(N_SEEDS)]
    t0 = time.time()
    res = [r for r in Parallel(n_jobs=6, backend="loky", verbose=5)(
        delayed(unit)(*t) for t in tasks) if r]
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)

    agg = {}
    for ds, sd, out in res:
        for k, v in out.items():
            agg.setdefault((ds, k), []).append(float(np.mean(v)))
    os.makedirs("results", exist_ok=True)
    json.dump({f"{a}|{b}": v for (a, b), v in agg.items()},
              open(os.environ.get("OUT", "results/phase3_mediank.json"), "w"),
              indent=2, default=float)

    from scipy import stats
    for ds in ("v2_regime_switch", "sector_etf"):
        for m in MK:
            base = np.mean(agg[(ds, f"baseline|{m}")])
            print(f"\n{'='*78}\n  {ds}  |  final model = {m}"
                  f"   TARGET = {base:.4f}\n{'='*78}")
            print(f"  {'arm':<24}{'AUC':>18}{'vs baseline':>13}{'n_sel':>8}")
            rows = [(f"baseline|{m}", 50)]
            for k in KS:
                rows += [(f"filter_frozen{k}|{m}", k), (f"filter_refit{k}|{m}", k)]
            rows += [(f"orpsoc_current|{m}", None), (f"orpsoc_mediank|{m}", None)]
            rows += [(f"oracle_test{k}|{m}", k) for k in KS]
            for key, kk in rows:
                v = np.array(agg[(ds, key)])
                if kk is None:
                    tag = key.split("|")[0].replace("orpsoc_", "nsel_")
                    kk = np.mean(agg[(ds, f"{tag}|{m}")])
                mark = "  <-- BEATS" if v.mean() > base else ""
                print(f"  {key.split('|')[0]:<24}{v.mean():>11.4f}±{v.std():.4f}"
                      f"{v.mean()-base:>+13.4f}{kk:>8.1f}{mark}")
            fr = np.mean(agg[(ds, f"filter_frozen5|{m}")])
            for arm in ("orpsoc_current", "orpsoc_mediank"):
                v = np.array(agg[(ds, f"{arm}|{m}")])
                t, p = stats.ttest_1samp(v, fr)
                print(f"    {arm:<18} vs frozen k=5: {v.mean()-fr:+.4f}"
                      f"   one-sample t={t:.2f}  p={p:.4f}")
