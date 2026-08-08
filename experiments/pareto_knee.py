"""
Elastic subset sizing: let the data choose k instead of fixing the price of a
feature in advance.

THE PROBLEM
───────────
The PSO objective is  theta*AUC + (1-theta)*(1 - k/N).  That IS a search over
subset size, but theta fixes the exchange rate BEFORE seeing the data: at
theta=0.5, N=50, an extra feature must buy 0.0100 AUC or it is rejected. On
Fama-French no marginal factor clears that bar, so every condition collapsed to
5-9 features against an intrinsic dimensionality of ~21 and discarded real
signal. On sector ETFs the same price happened to land near the right answer.
One global constant cannot be right for both.

Setting min_f = effective_rank is NOT the fix: effective rank is computed from
the correlation matrix of X alone and never looks at y. It is an unsupervised
guess at how many features you need to predict a target it has not seen.

THE IDEA UNDER TEST  (elastic / knee sizing)
────────────────────────────────────────────
Build the AUC-vs-k curve on the training window only, then pick the knee -- the
size past which adding features stops buying accuracy. Concretely, per fold:

  1. split the TRAINING window into inner-train / inner-val (75/25, causal,
     the same split the PSO uses for fitness -- the test fold is never touched)
  2. rank features on inner-train (univariate |AUC-0.5|, the same rule
     experiments/ff_adaptation.py uses, so ranking is not a confound)
  3. for k = 1..K_MAX: fit top-k on inner-train, score on inner-val -> AUC(k)
  4. pick k* from that curve by a KNEE rule (below)
  5. refit top-k* on the FULL training window, score on the TEST fold

Nothing at step 4 or 5 sees test data, so k* is chosen causally.

KNEE RULES compared (all pre-registered, none tuned on results):
  knee_kneedle  max distance from the chord joining (1, AUC(1)) to
                (K_MAX, AUC(K_MAX)) -- the standard Kneedle construction.
  knee_marginal smallest k whose forward marginal gain over the next
                PATIENCE steps stays below TOL. This is the "keep adding until
                it stops helping" rule directly.
  knee_1se      smallest k within 1 standard error of the best inner-val AUC
                (the lasso/CV convention: simplest model statistically
                indistinguishable from the best).

BASELINE ARMS for comparison:
  all           every feature (what the paper's baseline does)
  fixed5/10/20  the fixed sizes used in the frozen-vs-refit study
  effrank       k = effective rank of the training window (the unsupervised
                rule this experiment is meant to beat)

Run:  ORPSOC_N_JOBS=30 python experiments/pareto_knee.py
Writes results/pareto_knee.json and plots/pareto_knee_*.png
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

from orpsoc_runner import pin_threads, default_workers
pin_threads(1)

from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from orpsoc_utils import walk_forward_folds
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

K_MAX_CAP  = 40      # curves are built out to min(K_MAX_CAP, n_features)
TOL        = 0.002   # knee_marginal: gain below this counts as "stopped helping"
PATIENCE   = 3       # knee_marginal: sustained over this many steps
INNER_FRAC = 0.75    # inner-train fraction, matching the PSO fitness split

DATASETS = [("sector_etf", 500), ("fama_french", 500),
            ("bonds", 500), ("commodities", 500),
            ("v2_regime_switch", 150)]
MODELS = {
    "LogReg":   lambda: LogisticRegression(max_iter=200),
    "LightGBM": lambda: LGBMClassifier(n_estimators=100, num_leaves=31,
                                       learning_rate=0.1, verbosity=-1,
                                       random_state=42, n_jobs=1),
}


def _pipe(mk):
    return Pipeline([("i", SimpleImputer(strategy="mean")),
                     ("s", StandardScaler()), ("m", mk())])


def score(X_tr, y_tr, X_te, y_te, cols, mk):
    if len(cols) == 0 or y_te.nunique() < 2:
        return float("nan")
    p = _pipe(mk)
    p.fit(X_tr[cols], y_tr)
    try:
        s = p.predict_proba(X_te[cols])[:, 1]
    except AttributeError:
        s = p.decision_function(X_te[cols])
    return float(roc_auc_score(y_te, s))


def rank_features(X, y, feat):
    """Univariate ranking -- identical rule to experiments/ff_adaptation.py."""
    sc = {}
    for c in feat:
        try:
            sc[c] = abs(roc_auc_score(y, X[c]) - 0.5)
        except Exception:
            sc[c] = 0.0
    return [c for c, _ in sorted(sc.items(), key=lambda kv: -kv[1])]


def effective_rank(X):
    Xz = (X - np.nanmean(X, 0)) / (np.nanstd(X, 0) + 1e-12)
    Xz = np.nan_to_num(Xz)
    C = np.nan_to_num(np.corrcoef(Xz, rowvar=False))
    ev = np.clip(np.linalg.eigvalsh(C), 1e-12, None)
    p = ev / ev.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


# ── knee rules ───────────────────────────────────────────────────────────────
def knee_kneedle(ks, aucs):
    a = np.asarray(aucs, float)
    if np.all(np.isnan(a)) or len(a) < 3:
        return ks[0]
    x = np.asarray(ks, float)
    x0, x1, y0, y1 = x[0], x[-1], a[0], a[-1]
    denom = np.hypot(x1 - x0, y1 - y0) or 1.0
    dist = np.abs((y1 - y0) * x - (x1 - x0) * a + x1 * y0 - y1 * x0) / denom
    dist[np.isnan(a)] = -np.inf
    return int(x[int(np.argmax(dist))])


def knee_marginal(ks, aucs):
    """Smallest k after which the next PATIENCE gains all stay below TOL."""
    a = np.asarray(aucs, float)
    for i in range(len(a) - 1):
        nxt = a[i + 1:i + 1 + PATIENCE]
        if len(nxt) == 0:
            break
        if np.nanmax(nxt - a[i]) < TOL:
            return int(ks[i])
    return int(ks[int(np.nanargmax(a))])


def knee_1se(ks, aucs):
    a = np.asarray(aucs, float)
    if np.all(np.isnan(a)):
        return ks[0]
    best = np.nanmax(a)
    se = np.nanstd(a) / max(1.0, np.sqrt(np.sum(~np.isnan(a))))
    ok = np.where(a >= best - se)[0]
    return int(ks[int(ok[0])]) if len(ok) else int(ks[int(np.nanargmax(a))])


KNEES = {"knee_kneedle": knee_kneedle,
         "knee_marginal": knee_marginal,
         "knee_1se": knee_1se}


def run_one(ds, min_train, mname):
    d = pickle.load(open(f"data/{ds}.pkl", "rb"))
    X, y = d["X"], d["y"]
    feat = list(X.columns)
    mk = MODELS[mname]
    folds = walk_forward_folds(X, y, n_splits=8, gap=5, min_train=min_train)
    K_MAX = min(K_MAX_CAP, len(feat))
    ks = list(range(1, K_MAX + 1))

    out = {"ds": ds, "model": mname, "n_features": len(feat),
           "ks": ks, "folds": []}

    for fi, (X_tr, y_tr, X_te, y_te, _) in enumerate(folds):
        if y_te.nunique() < 2:
            continue
        # ── inner causal split of the TRAINING window only ──────────────────
        cut = int(len(X_tr) * INNER_FRAC)
        Xi, yi = X_tr.iloc[:cut], y_tr.iloc[:cut]
        Xv, yv = X_tr.iloc[cut:], y_tr.iloc[cut:]
        if yi.nunique() < 2 or yv.nunique() < 2:
            continue

        order = rank_features(Xi, yi, feat)
        curve = [score(Xi, yi, Xv, yv, order[:k], mk) for k in ks]

        er = effective_rank(X_tr.values)
        rec = {"fold": fi, "curve": curve, "eff_rank": er,
               "arms": {}, "chosen_k": {}}

        # knee arms: choose k on the curve, then evaluate on TEST
        for name, fn in KNEES.items():
            kstar = int(np.clip(fn(ks, curve), 1, K_MAX))
            rec["chosen_k"][name] = kstar
            rec["arms"][name] = score(X_tr, y_tr, X_te, y_te,
                                      order[:kstar], mk)
        # comparison arms
        rec["chosen_k"]["effrank"] = int(np.clip(round(er), 1, K_MAX))
        rec["arms"]["effrank"] = score(X_tr, y_tr, X_te, y_te,
                                       order[:rec["chosen_k"]["effrank"]], mk)
        rec["arms"]["all"] = score(X_tr, y_tr, X_te, y_te, feat, mk)
        rec["chosen_k"]["all"] = len(feat)
        for k in (5, 10, 20):
            rec["arms"][f"fixed{k}"] = score(X_tr, y_tr, X_te, y_te,
                                             order[:k], mk)
            rec["chosen_k"][f"fixed{k}"] = k
        out["folds"].append(rec)
    return out


def plot(results):
    os.makedirs("plots", exist_ok=True)
    by_ds = {}
    for r in results:
        by_ds.setdefault(r["ds"], []).append(r)
    for ds, rs in by_ds.items():
        fig, axes = plt.subplots(1, len(rs), figsize=(7 * len(rs), 4.6),
                                 squeeze=False)
        for ax, r in zip(axes[0], rs):
            ks = r["ks"]
            for rec in r["folds"]:
                ax.plot(ks, rec["curve"], lw=1, alpha=0.45)
            arr = np.array([rec["curve"] for rec in r["folds"]], float)
            mean = np.nanmean(arr, axis=0)
            ax.plot(ks, mean, lw=2.6, color="black", label="mean over folds")
            for name, col in (("knee_marginal", "#EF5350"),
                              ("knee_kneedle", "#42A5F5"),
                              ("knee_1se", "#66BB6A")):
                kk = np.mean([rec["chosen_k"][name] for rec in r["folds"]])
                ax.axvline(kk, color=col, ls="--", lw=1.8,
                           label=f"{name} (mean k={kk:.1f})")
            er = np.mean([rec["eff_rank"] for rec in r["folds"]])
            ax.axvline(er, color="#AB47BC", ls=":", lw=2.2,
                       label=f"effective rank ({er:.1f})")
            ax.set_title(f"{r['ds']} — {r['model']}")
            ax.set_xlabel("subset size k")
            ax.set_ylabel("inner-validation AUC")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.25)
        fig.suptitle(f"Elastic subset sizing — AUC vs k on the TRAINING window "
                     f"({ds}). Knee is chosen here, then scored on the held-out "
                     f"test fold.", fontsize=10)
        fig.tight_layout()
        p = f"plots/pareto_knee_{ds}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {p}")


def report(results):
    from scipy import stats
    ARMS = ["all", "fixed5", "fixed10", "fixed20", "effrank",
            "knee_kneedle", "knee_marginal", "knee_1se"]
    print()
    print("=" * 96)
    print("  TEST-FOLD AUC BY SIZING RULE   (k chosen causally on the training window)")
    print("  delta / p are vs the ALL-FEATURES baseline, paired across folds")
    print("=" * 96)
    for r in results:
        print(f"\n  {r['ds']} — {r['model']}   (N={r['n_features']}, "
              f"eff.rank={np.mean([f['eff_rank'] for f in r['folds']]):.1f})")
        print(f"    {'arm':<16}{'mean k':>8}{'test AUC':>10}{'vs all':>10}{'p':>9}")
        print("    " + "-" * 53)
        base = np.array([f["arms"]["all"] for f in r["folds"]], float)
        for a in ARMS:
            v = np.array([f["arms"].get(a, np.nan) for f in r["folds"]], float)
            k = np.mean([f["chosen_k"].get(a, np.nan) for f in r["folds"]])
            if a == "all":
                print(f"    {a:<16}{k:>8.1f}{np.nanmean(v):>10.4f}"
                      f"{'(base)':>10}{'-':>9}")
                continue
            try:
                p = stats.wilcoxon(v, base).pvalue
                ptxt = f"{p:>9.4f}"
            except Exception:
                ptxt = f"{'n/a':>9}"
            print(f"    {a:<16}{k:>8.1f}{np.nanmean(v):>10.4f}"
                  f"{np.nanmean(v)-np.nanmean(base):>+10.4f}{ptxt}")


def main():
    tasks = [(ds, mt, m) for ds, mt in DATASETS for m in MODELS]
    print("=" * 78)
    print("  ELASTIC / PARETO-KNEE SUBSET SIZING")
    print(f"  {len(tasks)} (dataset x model) cells, 8 folds each, "
          f"curves to k={K_MAX_CAP}")
    print(f"  workers={default_workers()}", flush=True)
    t0 = time.time()
    results = Parallel(n_jobs=default_workers(), verbose=5)(
        delayed(run_one)(*t) for t in tasks)
    print(f"\ndone in {(time.time()-t0)/60:.1f} min", flush=True)
    os.makedirs("results", exist_ok=True)
    with open("results/pareto_knee.json", "w") as f:
        json.dump(results, f)
    print("Saved: results/pareto_knee.json")
    plot(results)
    report(results)


if __name__ == "__main__":
    main()
