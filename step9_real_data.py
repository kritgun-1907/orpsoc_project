"""
STEP 9 — Real Financial Data Validation (sector ETFs + Fama-French)
===================================================================
Implements the supervisor's data-side request: move from synthetic
Level-4 (regime switch) to REAL datasets with documented structural
breaks, and run the SAME ablation pipeline (with the new transition
-smoothing fixes) on them.

TWO DATASETS
────────────
1. SECTOR ETFS  — the nine original SPDR Select Sector ETFs
   (XLB XLE XLF XLI XLK XLP XLU XLV XLY), daily Adj-Close from 2000.
   Rolling statistics per sector + cross-sectional features give a
   50+ feature matrix.  Ground-truth structural breaks:
       2001-09  dot-com unwind / 9-11
       2008-09  Lehman / GFC
       2020-02  COVID crash
2. FAMA-FRENCH  — the five-factor daily series (Mkt-RF, SMB, HML,
   RMW, CMA) from Kenneth French's data library at Dartmouth.
   Rolling statistics across multiple windows give 50 features.
   Standard academic benchmark; the direct analogue of the synthetic
   Level-4 setup.

WHAT THIS PRODUCES
──────────────────
  data/sector_etf.pkl      {"X", "y", "base"}  ← same format as step1
  data/fama_french.pkl     {"X", "y", "base"}
  results/step9_real_data.json
  plots/step9_<dataset>_recovery.png   (per-fold AUC, breaks marked)

Because the .pkl files use the identical format to the synthetic
datasets, you can ALSO drop them into step7_ablation.py's LEVELS dict
to run them through the exact same harness as Levels 1-4.

CONFIG FLAGS (top of file)
  PREP_ONLY = True   → only download + build + save the .pkl datasets,
                        skip the (slow) ablation.  Good for a quick
                        data-prep run.
  FAST_MODE          → reduced seeds/iters for a quick directional read.

Run with:
    pip install yfinance
    python step9_real_data.py
"""

import os
# Pin BLAS/OpenMP BEFORE numpy and lightgbm are imported -- they read these at
# load time. See orpsoc_runner.pin_threads().
from orpsoc_runner import (pin_threads, default_workers, provenance,
                           CheckpointStore)
pin_threads(1)

import numpy as np
import pandas as pd
import pickle
import json
import time
import io
import re
import ssl
import zipfile
import urllib.request
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier
from scipy.stats import wilcoxon   # importance-reinit ablation test
from joblib import Parallel, delayed

from orpsoc_utils import (
    walk_forward_folds, feature_stability_ratio,
    AdaptiveRegimeThreshold,
    run_standard_orpsoc,
    run_hybrid_orpsoc,
)

os.makedirs("data",    exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("plots",   exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

PREP_ONLY   = False    # True = only build+save .pkl datasets, skip ablation
FAST_MODE   = False     # True = quick directional read; False = paper run

N_SEEDS     = 5  if FAST_MODE else 20
MAX_ITER    = 20 if FAST_MODE else 60
N_PARTICLES = 10 if FAST_MODE else 20
N_SPLITS    = 6  if FAST_MODE else 8

# ── Execution controls (do NOT affect any computed number) ───────────────────
# N_JOBS parallelises over SEEDS ONLY. Folds always run sequentially, in
# chronological order, inside a single worker. Override with ORPSOC_N_JOBS.
THETA       = 0.5   # compactness weight; see step7_ablation.py for the frontier
N_JOBS      = default_workers()
# Per-(dataset, seed) checkpointing, scoped by a hash of CONFIG + the source of
# orpsoc_utils.py and this file. NOTE: this REPLACES the old
# "checkpoint_{key}_{fast|full}_v3.pkl" scheme, which keyed only on FAST_MODE
# and would therefore silently reload results produced under a different
# MAX_ITER / N_PARTICLES / N_SEEDS / N_SPLITS or a different code version --
# a direct violation of guardrail G3.
USE_CHECKPOINTS = True

CONFIG = {"fast_mode": FAST_MODE, "n_seeds": N_SEEDS, "max_iter": MAX_ITER,
          "n_particles": N_PARTICLES, "n_splits": N_SPLITS,
          "theta": THETA}
PROV   = provenance(CONFIG, ["orpsoc_utils.py", "step9_real_data.py"])

SECTOR_TICKERS = ["XLB", "XLE", "XLF", "XLI", "XLK",
                  "XLP", "XLU", "XLV", "XLY"]
START_DATE     = "2000-01-01"

# Documented structural breaks used as ground-truth regime change points.
BREAKS = {
    "sector_etf":  ["2001-09-17", "2008-09-15", "2020-02-20"],
    "fama_french": ["2001-09-17", "2008-09-15", "2020-02-20"],
}

FF_5FACTOR_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/"
                  "ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")

print("=" * 68)
print("  STEP 9: Real Financial Data Validation")
print(f"  Mode: {'PREP-ONLY' if PREP_ONLY else ('FAST (debug)' if FAST_MODE else 'FULL (paper)')}")
if not PREP_ONLY:
    print(f"  Seeds={N_SEEDS}  MaxIter={MAX_ITER}  Particles={N_PARTICLES}  Splits={N_SPLITS}")
    print(f"  Workers={N_JOBS} (seeds in parallel; folds always sequential)")
    print(f"  Provenance={PROV['hash']}  "
          f"checkpoints={'on' if USE_CHECKPOINTS else 'off'}")
print("=" * 68, flush=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA DOWNLOAD  (cached to data/raw_*.pkl so re-runs don't re-download)
# ══════════════════════════════════════════════════════════════════════════════

def download_sector_etfs() -> pd.DataFrame:
    """Daily adjusted-close prices for the nine SPDR sector ETFs."""
    cache = "data/raw_sector_prices.pkl"
    if os.path.exists(cache):
        print(f"  [cache] sector prices ← {cache}", flush=True)
        return pd.read_pickle(cache)

    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit("yfinance not installed.  Run:  pip install yfinance")

    print(f"  Downloading {len(SECTOR_TICKERS)} sector ETFs from Yahoo "
          f"({START_DATE}→today) …", flush=True)
    raw = yf.download(SECTOR_TICKERS, start=START_DATE, auto_adjust=True,
                      progress=False, threads=True)

    # yfinance returns a column MultiIndex (field, ticker) for multi-ticker pulls.
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0)
        if "Close" in lvl0:
            prices = raw["Close"].copy()
        else:                                  # (ticker, field) ordering
            prices = raw.xs("Close", axis=1, level=1)
    else:
        prices = raw[["Close"]].copy()

    prices = prices[[t for t in SECTOR_TICKERS if t in prices.columns]]
    prices = prices.dropna(how="all").ffill().dropna()
    if prices.shape[1] < len(SECTOR_TICKERS):
        print(f"  WARNING: only {prices.shape[1]}/{len(SECTOR_TICKERS)} "
              f"tickers returned data.", flush=True)
    prices.to_pickle(cache)
    print(f"  Saved raw prices → {cache}  shape={prices.shape}", flush=True)
    return prices


def _http_get_bytes(url: str, timeout: int = 60) -> bytes:
    """
    Fetch bytes robustly across environments. Tries `requests` first (it ships
    its own certifi CA bundle, sidestepping the macOS Python.framework
    "certificate verify failed" issue), then falls back to urllib with a
    certifi-backed SSL context, then the system default context.
    """
    headers = {"User-Agent": "Mozilla/5.0 (research; orpsoc)"}
    try:
        import requests
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.content
    except Exception:
        pass
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(req, timeout=timeout, context=ctx).read()


def download_fama_french() -> pd.DataFrame:
    """Daily Fama-French 5-factor series (decimal returns) from Dartmouth."""
    cache = "data/raw_fama_french.pkl"
    if os.path.exists(cache):
        print(f"  [cache] Fama-French ← {cache}", flush=True)
        return pd.read_pickle(cache)

    print(f"  Downloading Fama-French 5-factor daily from Dartmouth …", flush=True)
    raw = _http_get_bytes(FF_5FACTOR_URL, timeout=60)
    zf  = zipfile.ZipFile(io.BytesIO(raw))
    txt = zf.read(zf.namelist()[0]).decode("latin-1")
    lines = txt.splitlines()

    # The CSV has a copyright preamble; the data block is the run of rows that
    # start with an 8-digit date (YYYYMMDD). The header row contains 'Mkt-RF'.
    hdr   = next(i for i, l in enumerate(lines) if "Mkt-RF" in l)
    is_data = lambda l: re.match(r"^\s*\d{8}\s*,", l) is not None
    start = next(i for i in range(hdr, len(lines)) if is_data(lines[i]))
    end   = start
    while end < len(lines) and is_data(lines[end]):
        end += 1

    block = "\n".join([lines[hdr]] + lines[start:end])
    ff = pd.read_csv(io.StringIO(block))
    ff.columns = ["Date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    ff["Date"] = pd.to_datetime(ff["Date"].astype(int).astype(str), format="%Y%m%d")
    ff = ff.set_index("Date").astype(float) / 100.0          # percent → decimal
    ff = ff[ff.index >= pd.Timestamp(START_DATE)]
    ff.to_pickle(cache)
    print(f"  Saved raw factors → {cache}  shape={ff.shape}", flush=True)
    return ff


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING  (all look-back only — no future leakage)
# ══════════════════════════════════════════════════════════════════════════════

def _rsi(returns: pd.Series, window: int = 14) -> pd.Series:
    gain = returns.clip(lower=0).rolling(window).mean()
    loss = (-returns.clip(upper=0)).rolling(window).mean()
    rs   = gain / (loss + 1e-8)
    return (100 - 100 / (1 + rs)) / 100.0


def build_sector_etf_dataset(prices: pd.DataFrame, horizon: int = 5):
    """
    9 sectors × 6 per-sector features + 4 market/cross-sectional features
    → 58 features.  Target = direction of the next `horizon`-day equal-weight
    market return.  Column 0 (mkt_ret_1d) is a return series so the HMM's
    rolling-std volatility detector keys off it.
    """
    lr      = np.log(prices / prices.shift(1))
    mkt_ret = lr.mean(axis=1)                       # equal-weight market return

    feats = {}
    feats["mkt_ret_1d"]   = mkt_ret                 # <- col 0 (HMM driver)
    feats["mkt_vol_20"]   = mkt_ret.rolling(20).std()
    feats["xsec_disp_20"] = lr.std(axis=1).rolling(20).mean()   # dispersion
    feats["breadth_20"]   = (lr.rolling(20).mean() > 0).mean(axis=1)

    for tk in prices.columns:
        p, r = prices[tk], lr[tk]
        feats[f"ret_1d_{tk}"]   = r
        feats[f"vol_20_{tk}"]   = r.rolling(20).std()
        feats[f"mom_20_{tk}"]   = p / p.shift(20) - 1
        feats[f"mom_60_{tk}"]   = p / p.shift(60) - 1
        feats[f"ma_ratio_{tk}"] = p.rolling(20).mean() / p.rolling(60).mean()
        feats[f"rsi_14_{tk}"]   = _rsi(r, 14)

    X = pd.DataFrame(feats, index=prices.index)

    # TARGET: "will realized vol exceed its 1-year median?" — directly tests
    # what the HMM detects (variance breaks) and is genuinely forecastable.
    rolling_vol = mkt_ret.rolling(20).std()
    future_vol  = rolling_vol.shift(-horizon)
    median_vol  = rolling_vol.rolling(252).median()
    y    = (future_vol > median_vol).astype(int)
    base = mkt_ret

    X, y, base, dates = _align(X, y, base, warmup=120, trim=horizon)
    return X, y, base, dates


def build_fama_french_dataset(ff: pd.DataFrame, horizon: int = 5):
    """
    5 factors × 10 rolling features = 50 features.  Target = direction of the
    next `horizon`-day cumulative market factor (Mkt-RF).  Column 0
    (Mkt-RF_lvl) is the market return so the HMM detector keys off it.
    """
    factors = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    feats = {}
    for f in factors:
        s = ff[f]
        feats[f"{f}_lvl"]    = s
        feats[f"{f}_vol20"]  = s.rolling(20).std()
        feats[f"{f}_vol60"]  = s.rolling(60).std()
        feats[f"{f}_mom20"]  = s.rolling(20).sum()
        feats[f"{f}_mom60"]  = s.rolling(60).sum()
        feats[f"{f}_ma20"]   = s.rolling(20).mean()
        feats[f"{f}_ma60"]   = s.rolling(60).mean()
        feats[f"{f}_z20"]    = (s - s.rolling(20).mean()) / (s.rolling(20).std() + 1e-8)
        feats[f"{f}_skew60"] = s.rolling(60).skew()
        feats[f"{f}_kurt60"] = s.rolling(60).kurt()

    X = pd.DataFrame(feats, index=ff.index)
    # Mkt-RF_lvl first so step7's get_hmm_trigger(feat_names[0]) sees a return.
    X = X[["Mkt-RF_lvl"] + [c for c in X.columns if c != "Mkt-RF_lvl"]]

    mkt  = ff["Mkt-RF"]
    # TARGET: "will realized vol exceed its 1-year median?" — same logic as
    # sector_etf; aligns with what the HMM actually detects (variance breaks).
    rolling_vol = mkt.rolling(20).std()
    future_vol  = rolling_vol.shift(-horizon)
    median_vol  = rolling_vol.rolling(252).median()
    y    = (future_vol > median_vol).astype(int)
    base = mkt

    X, y, base, dates = _align(X, y, base, warmup=120, trim=horizon)
    return X, y, base, dates


def _align(X, y, base, warmup, trim):
    """Drop rolling-window warm-up rows and target-trim tail; align indices.
    Preserves the original DatetimeIndex as a separate 'dates' Series so
    break_folds() can map integer fold indices back to calendar dates reliably,
    even after NaN rows are dropped from the middle of the dataset.
    """
    X = X.iloc[warmup:-trim] if trim > 0 else X.iloc[warmup:]
    y = y.reindex(X.index)
    base = base.reindex(X.index)
    # Preserve original date index BEFORE reset so break mapping is accurate.
    dates = pd.Series(X.index, index=X.index) if hasattr(X.index, 'date') \
            else pd.Series(X.index)
    keep = y.notna() & np.isfinite(X.to_numpy()).all(axis=1)
    dates_kept = dates[keep].reset_index(drop=True)
    return (X[keep].reset_index(drop=True),
            y[keep].astype(int).reset_index(drop=True),
            base[keep].reset_index(drop=True),
            dates_kept)


# ══════════════════════════════════════════════════════════════════════════════
#  HMM REGIME DETECTOR  (inlined — identical maths to step7)
# ══════════════════════════════════════════════════════════════════════════════

class SimpleHMM:
    def __init__(self, n_iter=30, tol=1e-3):
        self.n_iter = n_iter; self.tol = tol
        self.pi = self.A = self.mu = self.sigma = None

    def _emission_prob(self, x):
        eps = 1e-8
        probs = np.zeros((len(x), 2))
        for k in range(2):
            diff = x - self.mu[k]
            sig = self.sigma[k] + eps
            # BUGFIX: this was "/ (sig + sqrt(2*pi))" -- an ADDITIVE constant,
            # not the correct Gaussian normalizer "sig * sqrt(2*pi)". For a
            # small sig (a tight, confident state), sqrt(2*pi)~2.51 dominates
            # the sum and the denominator barely changes with sig, which
            # suppresses the whole reason a tight-fitting state should be far
            # more informative than a fat one for points near its mean. This
            # was the DOMINANT cause of P(Transition) sticking at ~1.0
            # forever once any crisis entered the training window -- the
            # emission evidence for "we're back to quiet" was never strong
            # enough to overcome the transition-matrix prior. Confirmed by
            # direct test: same data, this one-symbol fix took P(Trans) from
            # "stuck at 1.0 on every post-crisis fold" to correctly reading
            # ~0.0 after each crisis ends and ~1.0 while inside one.
            probs[:, k] = np.exp(-0.5 * (diff / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
        return np.clip(probs, 1e-300, None)

    def fit(self, x):
        T = len(x); sx = np.sort(x); mid = len(sx) // 2
        # BUGFIX: a bare "+1e-4" additive floor is negligible against real
        # rolling-volatility scale, so the low-vol state's sigma can collapse
        # toward zero on a quiet historical stretch. Once that happens, any
        # later moderately-elevated volatility sits many std-devs from the
        # low state and the posterior saturates to ~1.0 for the rest of the
        # series -- confirmed on real data: trigger fire-rate was 1.0 on 7-8
        # of 8 folds on both real datasets. Floor sigma at 10% of the SERIES'
        # OWN std instead of an absolute constant, so it can't collapse below
        # a meaningful fraction of the data's actual scale.
        sigma_floor = max(1e-4, 0.1 * float(np.std(x)))
        self.mu    = np.array([np.mean(sx[:mid]), np.mean(sx[mid:])])
        self.sigma = np.maximum(
            np.array([np.std(sx[:mid]), np.std(sx[mid:])]), sigma_floor)
        self.pi    = np.array([0.7, 0.3])
        self.A     = np.array([[0.95, 0.05], [0.10, 0.90]])
        ll_prev = -np.inf
        for _ in range(self.n_iter):
            B = self._emission_prob(x)
            alpha = np.zeros((T, 2)); alpha[0] = self.pi * B[0]
            sc = np.zeros(T); sc[0] = alpha[0].sum(); alpha[0] /= sc[0] + 1e-300
            for t in range(1, T):
                alpha[t] = (alpha[t-1] @ self.A) * B[t]
                sc[t] = alpha[t].sum(); alpha[t] /= sc[t] + 1e-300
            beta = np.ones((T, 2))
            for t in range(T-2, -1, -1):
                beta[t] = self.A @ (B[t+1] * beta[t+1]); beta[t] /= beta[t].sum() + 1e-300
            gamma = alpha * beta; gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300
            xi = np.zeros((T-1, 2, 2))
            for t in range(T-1):
                xi[t] = np.outer(alpha[t], beta[t+1] * B[t+1]) * self.A
                xi[t] /= xi[t].sum() + 1e-300
            self.pi = gamma[0]
            self.A  = xi.sum(0) / (xi.sum(0).sum(1, keepdims=True) + 1e-300)
            self.mu = (gamma * x[:, None]).sum(0) / (gamma.sum(0) + 1e-300)
            self.sigma = np.maximum(
                np.sqrt((gamma * (x[:, None] - self.mu) ** 2).sum(0)
                       / (gamma.sum(0) + 1e-300)), sigma_floor)
            ll = np.sum(np.log(sc + 1e-300))
            if abs(ll - ll_prev) < self.tol:
                break
            ll_prev = ll
        return self

    def predict_proba(self, x):
        T = len(x); B = self._emission_prob(x)
        alpha = np.zeros((T, 2)); alpha[0] = self.pi * B[0]; alpha[0] /= alpha[0].sum() + 1e-300
        for t in range(1, T):
            alpha[t] = (alpha[t-1] @ self.A) * B[t]; alpha[t] /= alpha[t].sum() + 1e-300
        beta = np.ones((T, 2))
        for t in range(T-2, -1, -1):
            beta[t] = self.A @ (B[t+1] * beta[t+1]); beta[t] /= beta[t].sum() + 1e-300
        g = alpha * beta; g /= g.sum(axis=1, keepdims=True) + 1e-300
        return g


def get_hmm_trigger(X_train, feat_name, rolling_window=20, threshold_obj=None,
                    warmup_min_obs=150):
    """
    Fit HMM on the rolling volatility of feat_name; return
    (triggered: bool, p_trans: float, is_warmup: bool).

    BUGFIX (warm-start problem): with too little training data, a 2-state
    HMM can't separate states reliably and produces an unstable, sometimes
    falsely-confident p_trans -- confirmed directly: ~300 quiet observations
    with ZERO actual regime change produced p_trans=0.76, borderline enough
    to spuriously trigger. On a real run this showed up as fold 1 (the
    smallest training window) firing when it shouldn't have, which then
    ate the cooldown budget and masked detection at the REAL first break.
    Fix: don't let the detector act at all below `warmup_min_obs` valid
    rolling-vol observations. p_trans is still computed and returned for
    diagnostics, but `threshold_obj.update()` is not called during warmup,
    so an unreliable fold-1 reading can never consume cooldown or pollute
    the percentile threshold's history.
    """
    obs = pd.Series(X_train[feat_name].values).rolling(rolling_window).std().bfill().values
    if len(obs) < 30 or np.isnan(obs).any():
        return False, 0.0, True
    is_warmup = len(obs) < warmup_min_obs
    try:
        hmm = SimpleHMM(n_iter=30, tol=1e-3).fit(obs)
        p_trans = float(hmm.predict_proba(obs)[-1, 1])
    except Exception:
        return False, 0.0, is_warmup
    if is_warmup:
        return False, p_trans, True   # diagnostic only -- not acted on
    triggered = threshold_obj.update(p_trans) if threshold_obj else (p_trans > 0.5)
    return triggered, p_trans, False


def run_baseline(X_tr, y_tr, X_te, y_te, feat_names, seed):
    """No feature selection — all features. LightGBM, matching step7."""
    try:
        pipe = Pipeline([
            ("imp",    SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model",  LGBMClassifier(n_estimators=100, num_leaves=31,
                                      learning_rate=0.1, verbosity=-1,
                                      random_state=seed, n_jobs=1))
        ])
        pipe.fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
    except Exception as e:
        print(f"  [baseline] fold failed: {e}", flush=True)
        auc = 0.5
    return {"auc": auc, "selected": list(feat_names), "n_sel": len(feat_names)}


# ══════════════════════════════════════════════════════════════════════════════
#  ABLATION LOOP  (mirrors step7 conditions, uses the improved hybrid runner)
# ══════════════════════════════════════════════════════════════════════════════

CONDITIONS = {
    "baseline":         "Baseline (all features)",
    "standard_orpsoc":  "Standard OrPSOC",
    "apsoll":           "+APSOLL (no HMM)",
    "full_hybrid":      "Full Hybrid",
    "full_hybrid_noimp":"Full Hybrid (no imp-reinit)",
}


def run_one_seed(X, y, ds_key, seed):
    """
    Run ONE complete walk-forward sequence for one seed of one dataset.

    Unit of parallelism and of checkpointing. This is the body of the former
    `for seed in range(N_SEEDS)` loop, moved verbatim into a function — the
    fold loop inside is unchanged and still runs fold 1 -> 2 -> ... -> N in
    chronological order.

    All stateful objects (`hmm_threshold`, `warm_start_fh`,
    `warm_start_fh_noimp`) are constructed HERE, which is exactly where they
    were constructed before: at the top of the seed loop. Nothing crosses a
    seed boundary, and every RNG is seeded from `seed + fi * 1000`, so seeds
    may run in any order or concurrently without changing any result.
    """
    feat_names = list(X.columns)
    folds = walk_forward_folds(X, y, n_splits=N_SPLITS, gap=5, min_train=500)
    t_seed = time.time()

    sr = {c: {"fold_aucs": [], "fold_selected": []} for c in CONDITIONS}
    hmm_threshold = AdaptiveRegimeThreshold(method="percentile",
                                            lookback=50, percentile_k=85.0)
    warm_start_fh = None
    warm_start_fh_noimp = None      # own chain for the no-imp ablation

    for fi, (X_tr, y_tr, X_te, y_te, _) in enumerate(folds):
        if y_te.nunique() < 2:           # skip degenerate test folds
            continue
        pso_kw = dict(
            feat_names=feat_names, seed=seed + fi * 1000,
            n_particles=N_PARTICLES, max_iter=MAX_ITER, min_f=3, theta=THETA,
            cr_low=0.3, cr_high=0.8, w_max=0.9, w_min=0.4,
            N_explore=max(5, MAX_ITER // 4), lam=0.1,
        )

        r1 = run_baseline(X_tr, y_tr, X_te, y_te, feat_names, seed + fi * 1000)
        r2 = run_standard_orpsoc(X_tr, y_tr, X_te, y_te, **pso_kw)
        r3 = run_hybrid_orpsoc(X_tr, y_tr, X_te, y_te,
                               hmm_trigger=False, **pso_kw)

        triggered, p_trans, is_warmup = get_hmm_trigger(
            X_tr, feat_name=feat_names[0], threshold_obj=hmm_threshold)
        r4 = run_hybrid_orpsoc(X_tr, y_tr, X_te, y_te,
                               hmm_trigger=triggered,
                               warm_start_pos=warm_start_fh,
                               p_trans=p_trans, **pso_kw)
        warm_start_fh = r4["gbest_pos"]

        # Ablation twin: identical but importance-guided reinit OFF.
        r5 = run_hybrid_orpsoc(X_tr, y_tr, X_te, y_te,
                               hmm_trigger=triggered,
                               warm_start_pos=warm_start_fh_noimp,
                               p_trans=p_trans,
                               use_importance_reinit=False, **pso_kw)
        warm_start_fh_noimp = r5["gbest_pos"]

        # Log the FULL diagnostic picture, not just the gated boolean:
        # raw_fire = did the underlying signal cross the bar BEFORE
        # cooldown/warmup gating (None during warmup, since it was never
        # evaluated); is_warmup = was this fold too early to trust at all.
        # Without these, "triggered=False" is ambiguous between "no
        # signal" and "signal present but suppressed" -- exactly the
        # ambiguity that made the previous run impossible to diagnose.
        raw_fire = (None if is_warmup else
                   bool(getattr(hmm_threshold, "last_raw_fire", False)))
        sr["full_hybrid"].setdefault("fold_triggered", []).append(bool(triggered))
        sr["full_hybrid"].setdefault("fold_p_trans", []).append(float(p_trans))
        sr["full_hybrid"].setdefault("fold_raw_fire", []).append(raw_fire)
        sr["full_hybrid"].setdefault("fold_is_warmup", []).append(bool(is_warmup))

        for c, r in zip(CONDITIONS, (r1, r2, r3, r4, r5)):
            sr[c]["fold_aucs"].append(r["auc"])
            sr[c]["fold_selected"].append(r["selected"])


    for c in CONDITIONS:
        sr[c]["jaccard"] = feature_stability_ratio(sr[c]["fold_selected"])

    mean_auc = {c: np.mean(sr[c]["fold_aucs"]) for c in CONDITIONS}
    print(f"  [{ds_key}] seed {seed + 1}/{N_SEEDS}  " +
          "  ".join(f"{c[:4]}={mean_auc[c]:.3f}" for c in CONDITIONS) +
          f"  ({time.time() - t_seed:.0f}s)", flush=True)
    return sr


STORE = CheckpointStore("results/checkpoints", "step9", PROV,
                        enabled=USE_CHECKPOINTS)


def _seed_or_checkpoint(X, y, ds_key, seed):
    """Return a cached seed result if one exists, else compute and cache it."""
    unit = f"{ds_key}_seed{seed:03d}"
    cached = STORE.load(unit)
    if cached is not None:
        return cached
    result = run_one_seed(X, y, ds_key, seed)
    STORE.save(unit, result)
    return result


def run_ablation(X, y, ds_key):
    """
    Drive one dataset: seeds in parallel, folds sequential inside each seed.

    joblib returns results in SUBMISSION order, so level_results is in seed
    order exactly as the old sequential append produced it.
    """
    folds = walk_forward_folds(X, y, n_splits=N_SPLITS, gap=5, min_train=500)

    units = [f"{ds_key}_seed{s:03d}" for s in range(N_SEEDS)]
    if USE_CHECKPOINTS:
        print(f"  [{ds_key}] checkpoints: {STORE.summary(units)}", flush=True)

    seed_results_list = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_seed_or_checkpoint)(X, y, ds_key, seed)
        for seed in range(N_SEEDS)
    )

    level_results = {c: [sr[c] for sr in seed_results_list] for c in CONDITIONS}
    return level_results, folds



def break_folds(folds, X_index_dates, break_dates):
    """Return the fold indices whose TEST window contains each break date."""
    out = []
    for bd in break_dates:
        bd = pd.Timestamp(bd)
        for fi, (_, _, _, _, train_end) in enumerate(folds):
            # folds were built on a reset index; map via the date-indexed series
            t0 = X_index_dates.iloc[train_end + 5] if train_end + 5 < len(X_index_dates) else None
            t1 = X_index_dates.iloc[min(train_end + 5 + (len(X_index_dates)//N_SPLITS),
                                        len(X_index_dates) - 1)]
            if t0 is not None and t0 <= bd <= t1:
                out.append(fi)
                break
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  CHECKPOINTING
# ══════════════════════════════════════════════════════════════════════════════
#  REMOVED: the old dataset-level `_save_checkpoint` / `_load_checkpoint` pair,
#  which wrote `data/checkpoint_{key}_{fast|full}_v3.pkl`. That key encoded
#  FAST_MODE and NOTHING ELSE, so changing MAX_ITER, N_PARTICLES, N_SEEDS,
#  N_SPLITS, or any pipeline code and re-running would silently reload the
#  previous results and report them as new -- a direct violation of guardrail
#  G3 ("two numbers produced under different configs are not comparable"), with
#  nothing in the output to reveal it.
#
#  Replaced by the per-(dataset, seed) CheckpointStore defined above, which:
#    * scopes checkpoints by a hash of CONFIG + the source of orpsoc_utils.py
#      and this file, so a config or code change gets a NEW directory and can
#      never reuse stale units (old ones are kept, not destroyed, so re-running
#      at a previous config still resumes);
#    * stores the provenance record inside every file and re-verifies on load;
#    * checkpoints at seed granularity, so a crash costs one seed, not a whole
#      dataset;
#    * writes atomically (tmp + os.replace), so a kill mid-write cannot leave a
#      half-written checkpoint that would load as valid.
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def fold_auc_curve(level_results, cond):
    """Mean ± std AUC per fold across seeds."""
    aucs = [sr["fold_aucs"] for sr in level_results[cond]]
    m = min(len(a) for a in aucs)
    arr = np.array([a[:m] for a in aucs])
    return arr.mean(0), arr.std(0)


def plot_recovery(ds_key, level_results, brk_folds):
    plt.figure(figsize=(9, 5))
    colors = {"baseline": "gray", "standard_orpsoc": "tab:blue",
              "apsoll": "tab:green", "full_hybrid": "tab:red",
              "full_hybrid_noimp": "tab:purple"}
    for c in CONDITIONS:
        m, s = fold_auc_curve(level_results, c)
        x = np.arange(1, len(m) + 1)
        plt.plot(x, m, "-o", color=colors[c], label=CONDITIONS[c])
        plt.fill_between(x, m - s, m + s, color=colors[c], alpha=0.12)
    for bf in brk_folds:
        plt.axvline(bf + 1, color="red", ls="--", alpha=0.6)
    plt.axhline(0.5, color="k", ls=":", alpha=0.4)
    plt.xlabel("Walk-Forward Fold (chronological →)")
    plt.ylabel("AUC-ROC (mean ± std across seeds)")
    plt.title(f"Step 9 — Recovery on Real Data: {ds_key}\n"
              f"(red dashed = documented structural break)")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    path = f"plots/step9_{ds_key}_recovery.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}", flush=True)


def main():
    builders = {
        "sector_etf":  lambda: build_sector_etf_dataset(download_sector_etfs()),
        "fama_french": lambda: build_fama_french_dataset(download_fama_french()),
    }

    # ── 1. Build + save datasets ────────────────────────────────────────────
    print("\n── Building datasets ──────────────────────────────────────────", flush=True)
    datasets = {}
    for key, build in builders.items():
        try:
            X, y, base, dates = build()
        except Exception as e:
            print(f"  SKIP {key}: {e}", flush=True)
            continue
        with open(f"data/{key}.pkl", "wb") as f:
            pickle.dump({"X": X, "y": y, "base": base}, f)
        datasets[key] = (X, y, dates)
        print(f"  {key:12s}  X={X.shape}  y-balance={y.mean():.3f}  "
              f"→ data/{key}.pkl", flush=True)

    if not datasets:
        raise SystemExit("No datasets could be built (check network / yfinance).")

    if PREP_ONLY:
        print("\nPREP_ONLY=True — datasets saved, skipping ablation.", flush=True)
        return

    # ── 2. Run ablation on each ─────────────────────────────────────────────
    out = {"config": dict(CONFIG), "provenance": PROV, "datasets": {}}
    t0 = time.time()
    for key, (X, y, dates) in datasets.items():
        print(f"\n── Ablation: {key} ────────────────────────────────────────", flush=True)

        # Seeds run in parallel; completed seeds are resumed from the
        # provenance-scoped checkpoint store (see CHECKPOINTING above).
        level_results, folds = run_ablation(X, y, key)

        # dates comes directly from _align — no post-hoc reconstruction needed.
        brk = break_folds(folds, dates, BREAKS.get(key, []))

        summary = {}
        for c in CONDITIONS:
            m, s = fold_auc_curve(level_results, c)
            summary[c] = {
                "mean_auc": float(np.mean(m)),
                "fold_auc_mean": [float(v) for v in m],
                "fold_auc_std":  [float(v) for v in s],
                "mean_jaccard": float(np.mean(
                    [np.mean(sr["jaccard"]["per_fold_jaccard"])
                    if isinstance(sr["jaccard"], dict) and sr["jaccard"]["per_fold_jaccard"]
                    else 1.0
                    for sr in level_results[c]])),
            }
        out["datasets"][key] = {"break_folds": brk, "conditions": summary}
        plot_recovery(key, level_results, brk)

        # ── Importance-reinit ablation: Full Hybrid vs no-imp twin ───────────
        def _seed_means(cond):
            return np.array([np.mean(sr["fold_aucs"]) for sr in level_results[cond]])
        imp_means   = _seed_means("full_hybrid")
        noimp_means = _seed_means("full_hybrid_noimp")
        delta = float(imp_means.mean() - noimp_means.mean())
        p_val = float("nan")
        try:
            if len(imp_means) == len(noimp_means) and np.any(imp_means - noimp_means != 0):
                _, p_val = wilcoxon(imp_means, noimp_means, alternative="greater")
        except Exception:
            pass
        # Post-break folds only (where imp-reinit is designed to act)
        def _postbreak_mean(cond):
            vals = []
            for sr in level_results[cond]:
                aucs = sr["fold_aucs"]
                post = [aucs[i] for i in range(len(aucs))
                        if any(i >= bf for bf in brk)]
                if post:
                    vals.append(np.mean(post))
            return float(np.mean(vals)) if vals else float("nan")
        out["datasets"][key]["importance_ablation"] = {
            "full_hybrid_mean":       float(imp_means.mean()),
            "full_hybrid_noimp_mean": float(noimp_means.mean()),
            "delta_auc":              delta,
            "wilcoxon_p_greater":     float(p_val),
            "postbreak_full_hybrid":       _postbreak_mean("full_hybrid"),
            "postbreak_full_hybrid_noimp": _postbreak_mean("full_hybrid_noimp"),
        }
        # HMM trigger diagnostics per fold. p_trans/raw_fire/is_warmup do NOT
        # depend on the PSO seed (they're a pure function of X_train), so
        # take seed 0's arrays directly rather than averaging -- averaging
        # a list containing None (warmup folds) doesn't work anyway, and
        # averaging would hide exactly the ambiguity we're trying to remove.
        fh = level_results["full_hybrid"]
        s0 = fh[0] if fh else {}
        p_trans_arr  = s0.get("fold_p_trans", [])
        raw_fire_arr = s0.get("fold_raw_fire", [])
        warmup_arr   = s0.get("fold_is_warmup", [])
        trig_arr     = s0.get("fold_triggered", [])
        if trig_arr:
            out["datasets"][key]["trigger_fire_rate"] = [float(v) for v in trig_arr]
            out["datasets"][key]["trigger_p_trans"]   = [float(v) for v in p_trans_arr]
            out["datasets"][key]["trigger_raw_fire"]  = raw_fire_arr
            out["datasets"][key]["trigger_is_warmup"] = warmup_arr

        ia = out["datasets"][key]["importance_ablation"]
        print(f"  [{key}] imp-reinit ablation: FullHybrid={ia['full_hybrid_mean']:.4f}  "
              f"NoImp={ia['full_hybrid_noimp_mean']:.4f}  Δ={ia['delta_auc']:+.4f}  "
              f"p={ia['wilcoxon_p_greater']:.4f}", flush=True)
        if trig_arr:
            def _fmt_raw(v, w):
                if w: return "  W"    # warm-up: not evaluated at all
                return "  1" if v else "  0"
            print(f"  [{key}] break_folds={brk}", flush=True)
            print(f"    fold      : " + " ".join(f"{i+1:>4}" for i in range(len(trig_arr))), flush=True)
            print(f"    p_trans   : " + " ".join(f"{v:>4.2f}" for v in p_trans_arr), flush=True)
            print(f"    raw_fire  : " + " ".join(f"{_fmt_raw(v,w):>4}" for v, w in
                                                  zip(raw_fire_arr, warmup_arr)), flush=True)
            print(f"    final trig: " + " ".join(f"{'  1' if v else '  0':>4}" for v in trig_arr), flush=True)
            print(f"    (W = warm-up fold, not enough training data to trust the "
                  f"detector -- not evaluated, not counted as no-signal)", flush=True)


        print(f"  {key} mean AUC:  " +
              "  ".join(f"{c[:4]}={summary[c]['mean_auc']:.4f}" for c in CONDITIONS),
              flush=True)

    with open("results/step9_real_data.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: results/step9_real_data.json   ({time.time()-t0:.0f}s)", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()