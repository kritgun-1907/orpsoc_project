"""
Professor item 1 — does "adaptation hurts" reproduce on a second market?

Frozen vs refitting univariate filter, identical rule, identical k, no search.
The claim under test: on real financial data the FROZEN filter beats the one
refitted on every fold's training window; on synthetic data the sign flips.

Established so far on sector ETF (6/6 cells frozen wins) and v2 synthetic
(6/6 refit wins). This adds Fama-French, and re-runs the other two in the same
process so every number in the table comes from one execution.
"""
import os, sys, json, time, pickle
import numpy as np

sys.path.insert(0, '/Users/kritgunsingh0719gmail.com/Documents/orpsoc_research')
os.chdir('/Users/kritgunsingh0719gmail.com/Documents/orpsoc_research')
from orpsoc_runner import pin_threads
pin_threads(1)
from joblib import Parallel, delayed
from orpsoc_utils import walk_forward_folds
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

KS = [5, 10, 20]
MODELS = {
    "LogReg":       lambda: LogisticRegression(max_iter=200),
    "LinearSVM":    lambda: LinearSVC(max_iter=5000, dual="auto"),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=200,
                                                   random_state=42, n_jobs=1),
    "LightGBM":     lambda: LGBMClassifier(n_estimators=100, num_leaves=31,
                                           learning_rate=0.1, verbosity=-1,
                                           random_state=42, n_jobs=1),
}
DATASETS = [("sector_etf", 500, "real-equity"),
            ("fama_french", 500, "real-equity"),
            ("bonds", 500, "real-rates"),
            ("commodities", 500, "real-commodity"),
            ("v2_regime_switch", 150, "synthetic")]


def score(X_tr, y_tr, X_te, y_te, cols, mk):
    p = Pipeline([("i", SimpleImputer(strategy="mean")),
                  ("s", StandardScaler()), ("m", mk())])
    p.fit(X_tr[cols], y_tr)
    try:
        s = p.predict_proba(X_te[cols])[:, 1]
    except AttributeError:
        s = p.decision_function(X_te[cols])
    return roc_auc_score(y_te, s)


def rank(X, y, feat):
    sc = {}
    for c in feat:
        try:
            sc[c] = abs(roc_auc_score(y, X[c]) - 0.5)
        except Exception:
            sc[c] = 0.0
    return [c for c, _ in sorted(sc.items(), key=lambda kv: -kv[1])]


def unit(ds, mt, kind, mname, mk):
    d = pickle.load(open(f"data/{ds}.pkl", "rb"))
    X, y = d["X"], d["y"]
    feat = list(X.columns)
    folds = walk_forward_folds(X, y, n_splits=8, gap=5, min_train=mt)
    frozen = rank(folds[0][0], folds[0][1], feat)     # ranked ONCE, never updated
    out = {}
    for X_tr, y_tr, X_te, y_te, _ in folds:
        if y_te.nunique() < 2:
            continue
        refit = rank(X_tr, y_tr, feat)                 # re-ranked every fold
        out.setdefault("all", []).append(score(X_tr, y_tr, X_te, y_te, feat, mk))
        for k in KS:
            out.setdefault(f"frozen{k}", []).append(
                score(X_tr, y_tr, X_te, y_te, frozen[:k], mk))
            out.setdefault(f"refit{k}", []).append(
                score(X_tr, y_tr, X_te, y_te, refit[:k], mk))
    return ds, kind, mname, {k: [float(x) for x in v] for k, v in out.items()}


if __name__ == "__main__":
    tasks = [(ds, mt, kind, n, f) for ds, mt, kind in DATASETS
             for n, f in MODELS.items()]
    t0 = time.time()
    res = Parallel(n_jobs=6, backend="loky", verbose=5)(
        delayed(unit)(*t) for t in tasks)
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)
    json.dump([{"ds": a, "kind": b, "model": c, "folds": d} for a, b, c, d in res],
              open(os.environ.get("OUT", "/tmp/ff.json"), "w"), indent=2)

    from scipy import stats
    print(f"\n{'='*92}")
    print("  REFIT - FROZEN  (negative = adapting HURTS)   paired across 8 folds")
    print(f"{'='*92}")
    print(f"  {'dataset':<18}{'model':<14}{'k=5':>9}{'k=10':>9}{'k=20':>9}"
          f"{'frozen wins':>13}{'p (k=5)':>10}")
    tally = {}
    for ds, kind, m, o in sorted(res, key=lambda r: (r[1], r[0], r[2])):
        row, wins = [], 0
        p5 = float("nan")
        for k in KS:
            fr = np.array(o[f"frozen{k}"]); rf = np.array(o[f"refit{k}"])
            d = rf.mean() - fr.mean(); row.append(d)
            wins += int(d < 0)
            if k == 5:
                try:
                    p5 = stats.wilcoxon(rf, fr).pvalue
                except Exception:
                    pass
        tally.setdefault(kind, [0, 0])
        tally[kind][0] += wins; tally[kind][1] += len(KS)
        print(f"  {ds:<18}{m:<14}" + "".join(f"{v:>+9.4f}" for v in row)
              + f"{wins:>10}/3{p5:>10.3f}")
    print()
    for kind, (w, n) in tally.items():
        print(f"  {kind:<10} frozen beats refit in {w}/{n} cells")
