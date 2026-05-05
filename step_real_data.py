"""
STEP — Real Financial Data Validation (yfinance)
=================================================
Downloads real market data with known structural breaks and runs
the full OrPSOC ablation pipeline on it.

WHY REAL DATA?
──────────────
Synthetic results prove the algorithm works when we control the signal.
Real data proves it generalises.  Reviewers at quant journals require
at least one empirical dataset with a known structural break.

TWO DATASETS:
  Crisis 2008 : SPY daily 2005-01-01 → 2010-12-31
                True regime switch ≈ September 2008 (Lehman collapse)
  COVID 2020  : SPY daily 2018-01-01 → 2022-12-31
                True regime switch ≈ February 2020 (pandemic onset)

FEATURES ENGINEERED (all look-back only — no future leakage):
  log_return_1d   : daily log return
  log_return_5d   : 5-day log return (weekly)
  vol_20          : 20-day rolling std of log returns
  vol_60          : 60-day rolling std (quarterly)
  rsi_14          : 14-day RSI
  momentum_20     : 20-day price momentum (price/price[-20] - 1)
  momentum_60     : 60-day price momentum
  ma_ratio_20_60  : 20-day MA / 60-day MA (trend signal)
  skew_20         : 20-day rolling skewness of returns
  kurt_20         : 20-day rolling kurtosis of returns

BINARY TARGET:
  y = 1 if next-5-day log return > 0, else 0

Run with:
    pip install yfinance
    python step_real_data.py
"""

import numpy as np
import pandas as pd
import pickle
import json
import time
import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("data",    exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("plots",   exist_ok=True)

try:
    import yfinance as yf
except ImportError:
    raise SystemExit(
        "yfinance not installed.  Run:  pip install yfinance\n"
        "then re-run this script."
    )

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

from orpsoc_utils import (
    walk_forward_folds, feature_stability_ratio,
    AdaptiveRegimeThreshold,
    run_standard_orpsoc,
    run_hybrid_orpsoc,
)


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

FAST_MODE   = True
N_SEEDS     = 5  if FAST_MODE else 20
MAX_ITER    = 20 if FAST_MODE else 60
N_PARTICLES = 10 if FAST_MODE else 20
N_SPLITS    = 6  if FAST_MODE else 8

DATASETS = {
    "crisis_2008": {
        "ticker": "SPY",
        "start":  "2005-01-01",
        "end":    "2010-12-31",
        "switch_date": "2008-09-15",   # Lehman Brothers collapse
        "label":  "2008 Financial Crisis (SPY)",
    },
    "covid_2020": {
        "ticker": "SPY",
        "start":  "2018-01-01",
        "end":    "2022-12-31",
        "switch_date": "2020-02-20",   # COVID market crash onset
        "label":  "COVID Crash 2020 (SPY)",
    },
}

print("=" * 65)
print("  REAL DATA VALIDATION — yfinance")
print(f"  Mode: {'FAST (debug)' if FAST_MODE else 'FULL (paper)'}")
print(f"  Seeds={N_SEEDS}  MaxIter={MAX_ITER}  Particles={N_PARTICLES}")
print("=" * 65)
print()


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(prices: pd.Series) -> pd.DataFrame:
    """
    Build a look-ahead-free feature matrix from a daily close price series.
    All rolling windows use only past data (no shift(0) on future targets).
    """
    df = pd.DataFrame(index=prices.index)
    lr  = np.log(prices / prices.shift(1))

    df["log_return_1d"]  = lr
    df["log_return_5d"]  = np.log(prices / prices.shift(5))
    df["vol_20"]         = lr.rolling(20).std()
    df["vol_60"]         = lr.rolling(60).std()
    df["momentum_20"]    = prices / prices.shift(20) - 1
    df["momentum_60"]    = prices / prices.shift(60) - 1
    df["ma_ratio_20_60"] = prices.rolling(20).mean() / prices.rolling(60).mean()
    df["skew_20"]        = lr.rolling(20).skew()
    df["kurt_20"]        = lr.rolling(20).kurt()

    # RSI-14
    delta   = lr.copy()
    gain    = delta.clip(lower=0).rolling(14).mean()
    loss    = (-delta.clip(upper=0)).rolling(14).mean()
    rs      = gain / (loss + 1e-8)
    df["rsi_14"] = 100 - 100 / (1 + rs)

    # Normalise RSI to [0,1]
    df["rsi_14"] = df["rsi_14"] / 100.0

    return df


def make_target(prices: pd.Series, horizon: int = 5) -> pd.Series:
    """Binary: 1 if price is higher in `horizon` days, else 0."""
    fwd = np.log(prices.shift(-horizon) / prices)
    return (fwd > 0).astype(int)


# ══════════════════════════════════════════════════════════════════════════════
#  HMM (inlined — same as step7)
# ══════════════════════════════════════════════════════════════════════════════

class SimpleHMM:
    def __init__(self, n_iter=50, tol=1e-4):
        self.n_iter = n_iter; self.tol = tol; self.fitted = False
        self.pi = self.A = self.mu = self.sigma = None

    def _emission_prob(self, x):
        eps   = 1e-8
        probs = np.zeros((len(x), 2))
        for k in range(2):
            diff = x - self.mu[k]
            probs[:, k] = (np.exp(-0.5 * (diff / (self.sigma[k] + eps)) ** 2)
                           / (self.sigma[k] + eps + np.sqrt(2 * np.pi)))
        return np.clip(probs, 1e-300, None)

    def fit(self, x):
        T = len(x); sx = np.sort(x); mid = len(sx) // 2
        self.mu    = np.array([np.mean(sx[:mid]), np.mean(sx[mid:])])
        self.sigma = np.array([np.std(sx[:mid]) + 1e-4, np.std(sx[mid:]) + 1e-4])
        self.pi    = np.array([0.7, 0.3])
        self.A     = np.array([[0.95, 0.05], [0.10, 0.90]])
        ll_prev    = -np.inf
        for _ in range(self.n_iter):
            B = self._emission_prob(x)
            alpha = np.zeros((T, 2)); alpha[0] = self.pi * B[0]
            sc = np.zeros(T); sc[0] = alpha[0].sum(); alpha[0] /= sc[0] + 1e-300
            for t in range(1, T):
                alpha[t] = (alpha[t-1] @ self.A) * B[t]
                sc[t] = alpha[t].sum(); alpha[t] /= sc[t] + 1e-300
            beta = np.ones((T, 2)); beta[-1] = 1.0
            for t in range(T-2, -1, -1):
                beta[t] = self.A @ (B[t+1] * beta[t+1])
                beta[t] /= beta[t].sum() + 1e-300
            gamma = alpha * beta; gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300
            xi = np.zeros((T-1, 2, 2))
            for t in range(T-1):
                xi[t] = np.outer(alpha[t], beta[t+1] * B[t+1]) * self.A
                xi[t] /= xi[t].sum() + 1e-300
            self.pi = gamma[0]
            self.A  = xi.sum(0) / (xi.sum(0).sum(1, keepdims=True) + 1e-300)
            self.mu = (gamma * x[:,None]).sum(0) / (gamma.sum(0) + 1e-300)
            self.sigma = np.sqrt((gamma*(x[:,None]-self.mu)**2).sum(0)
                                 / (gamma.sum(0)+1e-300)) + 1e-4
            ll = np.sum(np.log(sc + 1e-300))
            if abs(ll - ll_prev) < self.tol: break
            ll_prev = ll
        self.fitted = True; return self

    def predict_proba(self, x):
        T = len(x); B = self._emission_prob(x)
        alpha = np.zeros((T, 2)); alpha[0] = self.pi * B[0]
        alpha[0] /= alpha[0].sum() + 1e-300
        for t in range(1, T):
            alpha[t] = (alpha[t-1] @ self.A) * B[t]
            alpha[t] /= alpha[t].sum() + 1e-300
        beta = np.ones((T, 2)); beta[-1] = 1.0
        for t in range(T-2, -1, -1):
            beta[t] = self.A @ (B[t+1] * beta[t+1])
            beta[t] /= beta[t].sum() + 1e-300
        g = alpha * beta; g /= g.sum(axis=1, keepdims=True) + 1e-300
        return g


def get_hmm_trigger(X_train, vol_col="vol_20",
                    threshold_obj=None):
    """
    Fit HMM on training volatility and return (triggered, p_trans).
    Uses pre-computed vol_20 column (no look-ahead).
    """
    obs = X_train[vol_col].bfill().values
    if len(obs) < 30 or np.isnan(obs).any():
        return False, 0.0
    hmm = SimpleHMM(n_iter=30, tol=1e-3)
    try:
        hmm.fit(obs)
        gamma   = hmm.predict_proba(obs)
        p_trans = float(gamma[-1, 1])
    except Exception:
        return False, 0.0
    triggered = threshold_obj.update(p_trans) if threshold_obj else (p_trans > 0.5)
    return triggered, p_trans


# ══════════════════════════════════════════════════════════════════════════════
#  BASELINE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_baseline(X_tr, y_tr, X_te, y_te, feat_names, seed):
    try:
        pipe = Pipeline([
            ("imp",    SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model",  XGBClassifier(n_estimators=80, max_depth=4,
                                     learning_rate=0.1, verbosity=0,
                                     random_state=seed))
        ])
        pipe.fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
    except Exception:
        auc = 0.5
    return {"auc": auc, "selected": list(feat_names)}


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

CONDITIONS = {
    "baseline":        "Baseline (all features)",
    "standard_orpsoc": "Standard OrPSOC",
    "apsoll":          "+APSOLL (no HMM)",
    "full_hybrid":     "Full Hybrid",
}

ALL_RESULTS = {}
t_global = time.time()

for ds_key, ds_cfg in DATASETS.items():
    print(f"\n{'─'*65}")
    print(f"  {ds_cfg['label']}")
    print(f"  Regime switch: {ds_cfg['switch_date']}")
    print(f"{'─'*65}")

    # ── Download data ─────────────────────────────────────────────────────────
    cache_path = f"data/real_{ds_key}.pkl"
    if os.path.exists(cache_path):
        print("  Loading cached data...")
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        X, y, prices = cached["X"], cached["y"], cached["prices"]
    else:
        print(f"  Downloading {ds_cfg['ticker']} from yfinance...")
        try:
            raw = yf.download(ds_cfg["ticker"],
                              start=ds_cfg["start"],
                              end=ds_cfg["end"],
                              progress=False, auto_adjust=True)
        except Exception as e:
            print(f"  ERROR downloading data: {e}")
            print("  Skipping this dataset.")
            continue

        if raw.empty:
            print("  No data returned. Skipping.")
            continue

        prices = raw["Close"].squeeze()
        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]

        X = engineer_features(prices)
        y = make_target(prices, horizon=5)

        # Align and drop NaN rows (from rolling windows and forward target)
        combined = pd.concat([X, y.rename("target")], axis=1).dropna()
        X = combined.drop(columns=["target"])
        y = combined["target"]

        with open(cache_path, "wb") as f:
            pickle.dump({"X": X, "y": y, "prices": prices}, f)
        print(f"  Cached to {cache_path}")

    feat_names  = list(X.columns)
    switch_date = pd.Timestamp(ds_cfg["switch_date"])
    switch_idx  = int(np.searchsorted(X.index, switch_date))

    print(f"  Rows: {len(X)}   Features: {len(feat_names)}")
    print(f"  Target: {int(y.sum())} UP ({100*y.mean():.1f}%)  "
          f"| Switch at index {switch_idx} ({ds_cfg['switch_date']})")
    print()

    folds = walk_forward_folds(X, y, n_splits=N_SPLITS, gap=5, min_train=150)
    if not folds:
        print("  Not enough data for walk-forward folds. Skipping.")
        continue

    level_results = {cond: [] for cond in CONDITIONS}

    for seed in range(N_SEEDS):
        seed_results = {cond: {"fold_aucs": [], "fold_selected": [],
                                "runtimes": []}
                        for cond in CONDITIONS}

        hmm_threshold = AdaptiveRegimeThreshold(method="percentile",
                                                lookback=50,
                                                percentile_k=85.0)

        for fold_idx, (X_tr, y_tr, X_te, y_te, train_end) in enumerate(folds):
            if len(y_te.unique()) < 2:
                continue

            pso_kw = dict(
                feat_names=feat_names, seed=seed + fold_idx * 1000,
                n_particles=N_PARTICLES, max_iter=MAX_ITER,
                min_f=2, theta=0.7,
                cr_low=0.3, cr_high=0.8, w_max=0.9, w_min=0.4,
                N_explore=max(5, MAX_ITER // 4), lam=0.1,
            )

            # Condition 1: Baseline
            t0 = time.time()
            r1 = run_baseline(X_tr, y_tr, X_te, y_te, feat_names,
                              seed + fold_idx * 1000)
            seed_results["baseline"]["fold_aucs"].append(r1["auc"])
            seed_results["baseline"]["fold_selected"].append(r1["selected"])
            seed_results["baseline"]["runtimes"].append(time.time() - t0)

            # Condition 2: Standard OrPSOC
            r2 = run_standard_orpsoc(X_tr, y_tr, X_te, y_te, **pso_kw)
            seed_results["standard_orpsoc"]["fold_aucs"].append(r2["auc"])
            seed_results["standard_orpsoc"]["fold_selected"].append(r2["selected"])
            seed_results["standard_orpsoc"]["runtimes"].append(r2["runtime"])

            # Condition 3: +APSOLL, no HMM
            r3 = run_hybrid_orpsoc(X_tr, y_tr, X_te, y_te,
                                   hmm_trigger=False, **pso_kw)
            seed_results["apsoll"]["fold_aucs"].append(r3["auc"])
            seed_results["apsoll"]["fold_selected"].append(r3["selected"])
            seed_results["apsoll"]["runtimes"].append(r3["runtime"])

            # Condition 4: Full Hybrid with delayed HMM trigger
            triggered, p_trans = get_hmm_trigger(X_tr, vol_col="vol_20",
                                                  threshold_obj=hmm_threshold)
            r4 = run_hybrid_orpsoc(X_tr, y_tr, X_te, y_te,
                                   hmm_trigger=triggered, **pso_kw)
            seed_results["full_hybrid"]["fold_aucs"].append(r4["auc"])
            seed_results["full_hybrid"]["fold_selected"].append(r4["selected"])
            seed_results["full_hybrid"]["runtimes"].append(r4["runtime"])

        for cond in CONDITIONS:
            jac = feature_stability_ratio(seed_results[cond]["fold_selected"])
            seed_results[cond]["jaccard"] = jac
            level_results[cond].append(seed_results[cond])

        auc_str = "  ".join(
            f"{cond[:3]}={np.mean(seed_results[cond]['fold_aucs']):.3f}"
            for cond in CONDITIONS
        )
        print(f"  seed={seed+1:2d}/{N_SEEDS}  {auc_str}")

    ALL_RESULTS[ds_key] = {
        "label":      ds_cfg["label"],
        "switch_date": ds_cfg["switch_date"],
        "switch_idx":  switch_idx,
        "n_rows":      len(X),
        "n_features":  len(feat_names),
        "feat_names":  feat_names,
        "conditions":  level_results,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

def to_json(obj):
    if isinstance(obj, (np.floating, np.integer)): return float(obj)
    if isinstance(obj, np.ndarray):                return obj.tolist()
    if isinstance(obj, dict):  return {k: to_json(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [to_json(v)    for v in obj]
    return obj


print("\n" + "=" * 65)
print("  SUMMARY — Real Data Validation")
print("=" * 65)
cond_labels = ["Baseline", "OrPSOC", "+APSOLL", "Full Hybrid"]
print(f"{'Dataset':<28}" + "".join(f"{lb:>15}" for lb in cond_labels))

real_summary = {}
for ds_key, rdata in ALL_RESULTS.items():
    row   = f"{rdata['label'][:27]:<28}"
    real_summary[ds_key] = {}
    for cond in CONDITIONS:
        seed_aucs = [np.mean(sr["fold_aucs"])
                     for sr in rdata["conditions"][cond]]
        m = float(np.mean(seed_aucs)); s = float(np.std(seed_aucs))
        row += f"{m:.3f}±{s:.3f}   "
        real_summary[ds_key][cond] = {"mean": m, "std": s,
                                       "seed_aucs": [float(v) for v in seed_aucs]}
    print(row)

print()


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT — fold-by-fold AUC for each real dataset
# ══════════════════════════════════════════════════════════════════════════════

COND_KEYS   = list(CONDITIONS.keys())
COND_COLORS = ["#78909C", "#42A5F5", "#66BB6A", "#EF5350"]

if ALL_RESULTS:
    n_ds  = len(ALL_RESULTS)
    fig, axes = plt.subplots(1, n_ds, figsize=(8 * n_ds, 5), squeeze=False)
    fig.suptitle("Real Data Validation — AUC per Walk-Forward Fold\n"
                 "(Dashed vertical line = known regime switch date)",
                 fontsize=12, fontweight="bold")

    for ax, (ds_key, rdata) in zip(axes[0], ALL_RESULTS.items()):
        folds_data = rdata["conditions"]["baseline"]
        n_folds_actual = min(len(sr["fold_aucs"]) for sr in folds_data)
        fold_x = list(range(1, n_folds_actual + 1))
        switch_fold_approx = int(rdata["switch_idx"] /
                                 max(rdata["n_rows"], 1) * n_folds_actual)

        for ck, col, cl in zip(COND_KEYS, COND_COLORS, cond_labels):
            sr_list   = rdata["conditions"][ck]
            auc_mat   = np.array([sr["fold_aucs"][:n_folds_actual]
                                  for sr in sr_list])
            m = auc_mat.mean(axis=0)
            s = auc_mat.std(axis=0)
            ax.plot(fold_x, m, "o-", color=col, linewidth=2, markersize=5, label=cl)
            ax.fill_between(fold_x, m - s, m + s, alpha=0.12, color=col)

        ax.axvline(switch_fold_approx + 0.5, color="red", linewidth=2,
                   linestyle="--", alpha=0.8,
                   label=f"Regime switch ≈ fold {switch_fold_approx}")
        ax.axhline(0.5, color="gray", linewidth=1, linestyle=":", alpha=0.4)
        ax.set_title(rdata["label"], fontsize=10, fontweight="bold")
        ax.set_xlabel("Walk-Forward Fold", fontsize=9)
        ax.set_ylabel("AUC-ROC", fontsize=9)
        ax.set_ylim([0.3, 1.05])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig("plots/real_data_fold_auc.png", dpi=150, bbox_inches="tight")
    print("Saved: plots/real_data_fold_auc.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════

save = {
    "config": {
        "fast_mode":   FAST_MODE,
        "n_seeds":     N_SEEDS,
        "max_iter":    MAX_ITER,
        "n_particles": N_PARTICLES,
        "n_splits":    N_SPLITS,
    },
    "summary":      real_summary,
    "full_results": to_json(ALL_RESULTS),
}

with open("results/real_data_validation.json", "w") as f:
    json.dump(save, f, indent=2)

print("Saved: results/real_data_validation.json")
print(f"Total time: {(time.time()-t_global)/60:.1f} min")
print()
print("=" * 65)
print("  REAL DATA VALIDATION COMPLETE")
print("=" * 65)
print()
print("What to check in your paper write-up:")
print("  1. Full Hybrid AUC vs +APSOLL on the post-switch folds")
print("     (after hmm_trigger_delay fix, Full Hybrid should recover faster)")
print("  2. Feature stability Jaccard dip around the switch fold")
print("  3. Compare real-data AUC levels with synthetic Level 4 results")
print("     (real markets are noisier; lower AUC is expected)")
