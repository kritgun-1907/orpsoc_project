"""
Is the fitness criterion at its ceiling?
=======================================
median_k explains 16% of the variance in test AUC on real data, and even a
PERFECT search maximising it lands below the all-features baseline. Two
questions follow, and both are measurable.

Q1 — HOW MUCH OF TEST AUC IS PREDICTABLE AT ALL?
    Split the test fold into two disjoint halves and score every candidate
    subset on each. corr(AUC on half A, AUC on half B) measures how much of a
    subset's test performance is REPRODUCIBLE rather than sampling noise --
    using two samples from the *same* distribution, with no regime shift and no
    causality constraint. That is the most favourable possible case.

    No causal criterion can beat this. If the reliability ceiling is ~0.4 then
    median_k at 0.40 is already at the limit and further criterion work is
    wasted. If it is ~0.8 there is real headroom.

    Halving the test window halves the sample, so the raw split-half
    correlation understates full-fold reliability. The Spearman-Brown
    correction r_full = 2r/(1+r) adjusts for that.

Q2 — WOULD A FILTER-STYLE CRITERION DO BETTER?
    The frozen univariate filter beats every wrapper we have, and it never
    evaluates a subset at all -- it ranks individual features once and stops.
    So test a criterion built the same way: score a subset by the univariate
    strength of its members on the training window, with no subset-level
    performance estimate. This bridges the filter and wrapper paradigms and is
    nearly free (feature scores are computed once per fold).

Run:  python experiments/compass_ceiling.py
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
from orpsoc_utils import walk_forward_folds
from orpsoc_criteria import CriterionBank
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

N_SUBSETS = 80
CRITERIA = ["current", "median_k"]


def unit(ds, mt, fi):
    d = pickle.load(open(f"data/{ds}.pkl", "rb"))
    X, y = d["X"], d["y"]
    feat = list(X.columns)
    folds = walk_forward_folds(X, y, n_splits=8, gap=5, min_train=mt)
    X_tr, y_tr, X_te, y_te, _ = folds[fi]
    if y_te.nunique() < 2:
        return None

    bank = CriterionBank(X_tr, y_tr, feat, seed=fi)

    # Univariate feature strength on the TRAINING window only -- the same
    # quantity the frozen filter ranks on. Computed once per fold.
    uni = np.array([abs(roc_auc_score(y_tr, X_tr[c]) - 0.5) for c in feat])

    # Interleaved test halves (every other row) so both halves span the whole
    # test window. A contiguous split would put them in different sub-periods
    # and confound sampling noise with within-fold drift.
    n = len(X_te)
    ia, ib = np.arange(0, n, 2), np.arange(1, n, 2)

    rng = np.random.RandomState(11 + fi)
    rows = []
    for _ in range(N_SUBSETS):
        k = rng.randint(3, min(30, len(feat)))
        idx = np.sort(rng.choice(len(feat), size=k, replace=False))
        cols = [feat[i] for i in idx]
        p = Pipeline([("i", SimpleImputer(strategy="mean")),
                      ("s", StandardScaler()),
                      ("m", LGBMClassifier(n_estimators=100, num_leaves=31,
                                           learning_rate=0.1, verbosity=-1,
                                           random_state=42, n_jobs=1))])
        p.fit(X_tr[cols], y_tr)
        pr = p.predict_proba(X_te[cols])[:, 1]
        r = {"k": int(k), "test": roc_auc_score(y_te, pr)}
        ya, yb = y_te.iloc[ia], y_te.iloc[ib]
        r["half_a"] = roc_auc_score(ya, pr[ia]) if ya.nunique() > 1 else np.nan
        r["half_b"] = roc_auc_score(yb, pr[ib]) if yb.nunique() > 1 else np.nan
        for c in CRITERIA:
            r[c] = bank.score(idx, c)
        # NOTE: univ_sum is NOT a usable criterion -- it correlates +0.978 with
        # subset size and its argmax picks the largest subset available. Kept
        # only as a documented trap, alongside its size-blind twin.
        r["univ_sum"] = float(uni[idx].sum())
        r["univ_mean"] = float(uni[idx].mean())
        # THRESHOLD form: adding a feature above the fold's median univariate
        # strength helps, below it hurts, so the optimum is a well-defined
        # subset ("everything above the bar") rather than "as many as possible".
        r["univ_thresh"] = float((uni[idx] - np.median(uni)).sum())
        # sqrt-normalised: a compromise between sum (grows with k) and mean
        # (size-blind).
        r["univ_sqrt"] = float(uni[idx].sum() / np.sqrt(len(idx)))
        rows.append(r)
    return ds, fi, rows


if __name__ == "__main__":
    tasks = [(ds, mt, fi) for ds, mt in (("sector_etf", 500),
                                         ("v2_regime_switch", 150))
             for fi in (4, 5, 6)]
    t0 = time.time()
    res = [r for r in Parallel(n_jobs=6, backend="loky", verbose=5)(
        delayed(unit)(*t) for t in tasks) if r]
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)
    json.dump([{"ds": a, "fold": b, "rows": c} for a, b, c in res],
              open(os.environ.get("OUT", "results/compass_ceiling.json"), "w"),
              indent=2, default=float)

    ALL = CRITERIA + ["univ_sum", "univ_mean", "univ_thresh", "univ_sqrt"]
    for ds in ("sector_etf", "v2_regime_switch"):
        sub = [r for r in res if r[0] == ds]
        rel, corr = [], {c: [] for c in ALL}
        pick = {c: [] for c in ALL}
        rnd, best = [], []
        for _, _, rows in sub:
            a = np.array([r["half_a"] for r in rows], float)
            b = np.array([r["half_b"] for r in rows], float)
            t = np.array([r["test"] for r in rows], float)
            m = ~(np.isnan(a) | np.isnan(b) | np.isnan(t))
            rel.append(np.corrcoef(a[m], b[m])[0, 1])
            rnd.append(t[m].mean()); best.append(t[m].max())
            for c in ALL:
                s = np.array([r[c] for r in rows], float)
                mm = m & ~np.isnan(s)
                corr[c].append(np.corrcoef(s[mm], t[mm])[0, 1])
                pick[c].append(t[mm][np.argmax(s[mm])])
        r_half = np.mean(rel)
        r_full = 2 * r_half / (1 + r_half)
        rnd, best = np.mean(rnd), np.mean(best)
        print(f"\n{'='*74}\n  {ds}\n{'='*74}")
        print(f"  RELIABILITY CEILING")
        print(f"    split-half corr (same distribution, no shift) : {r_half:+.3f}")
        print(f"    Spearman-Brown corrected to full fold         : {r_full:+.3f}"
              f"   <- NO causal criterion can exceed this")
        print(f"\n  {'criterion':<12}{'corr w/ test':>14}{'% of ceiling':>14}"
              f"{'gap captured':>14}")
        for c in ALL:
            cc = np.mean(corr[c])
            gp = 100 * (np.mean(pick[c]) - rnd) / (best - rnd)
            print(f"  {c:<12}{cc:>+14.3f}{100*cc/r_full:>13.0f}%{gp:>13.0f}%")
