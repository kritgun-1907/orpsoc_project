"""
Replace the HMM: twin-window KS and BOCPD vs the incumbent, on the noise null control.

WHY BYPASS THE HMM ENTIRELY
───────────────────────────
Three statistics built on the HMM posteriors were tested against a noise null
control on v2_regime_switch (break enters TRAINING at fold 4):

    statistic                hit@4   FP   NULL
    level (incumbent)          2/5    15    13
    deviation                  0/5     8     6
    tail (T=60, k=90)          2/5     1     2
    occupancy (k=85)           0/5     0     2

None achieves fold-4 detection at zero false alarms. The reason is upstream of
all of them: roughly three quarters of the `gamma` values are pinned at 0.000 or
1.000 -- the saturation AdaptiveRegimeThreshold itself warns about ("this usually
means the underlying P(Transition) signal is saturated ... Check HMM state
separation"). Every statistic above is a summary of those posteriors, so none can
recover information the posteriors already destroyed.

KS and BOCPD do not use the HMM at all. They act directly on the rolling
volatility series, so they are not subject to that failure mode.

THE TWO ALTERNATIVES
────────────────────
  ks_twin   Two-sample Kolmogorov-Smirnov between the recent T observations and
            the preceding T. Distribution-free, and the D statistic is a bounded
            effect size in [0,1] that does NOT saturate the way a p-value does.
            This is the classic twin-window change detector.

  bocpd     Bayesian Online Changepoint Detection (Adams & MacKay 2007) with a
            Normal-Inverse-Gamma conjugate prior and constant hazard 1/LAMBDA.
            Maintains the full run-length posterior; the signal is
            P(run length <= L), i.e. "how much belief is on a changepoint having
            occurred recently". Principled, online, and makes no two-state
            assumption at all -- which matters because the two-state assumption
            is exactly what the saturated HMM is failing to satisfy.

All statistics feed the SAME AdaptiveRegimeThreshold, so differences are
attributable to the detector rather than the gate.

PRE-REGISTERED SCORING (identical to the previous three experiments)
    hit@4    fires at fold 4, the earliest CAUSALLY possible fold
    hit@>=4  fires at any post-break fold (correct, possibly late)
    FP       any fire before fold 4, on any feature      <- must be 0
    NULL     any fire at all on a noise_* feature        <- must be ~0

SUCCESS BAR, fixed before running: beat `occupancy` k=85 (0/5 @ fold4, 3/5 late,
FP=0, NULL=2) by achieving hit@4 >= 2/5 with FP = 0 and NULL <= 2.

Run:  python experiments/detector_alternatives.py
"""
import os
import sys
import pickle

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

import pandas as pd
from scipy import stats as sstats
from orpsoc_utils import (walk_forward_folds, classify_folds,
                          AdaptiveRegimeThreshold)

_s7 = open("step7_ablation.py").read().split("#  MASTER ABLATION LOOP")[0]
_ns = {}
exec(compile(_s7, "step7_ablation<head>", "exec"), _ns)
SimpleHMM = _ns["SimpleHMM"]

# Dataset is an argument so the four detectors can be re-run on a HELD-OUT draw.
# Four experiments selected operating points from the seed-42 draw; re-running on
# a fresh seed is what converts that tuning into a genuine out-of-sample check.
#     python experiments/detector_alternatives.py v2_regime_switch_holdout
LEVEL = sys.argv[1] if len(sys.argv) > 1 else "v2_regime_switch"
SWITCH, ROLLING, WARMUP = 500, 20, 150
SIGNALS = ["signal_0", "signal_1", "signal_2", "signal_3", "signal_4"]
NOISE = ["noise_0", "noise_1", "noise_2", "noise_3", "noise_4"]

KS_T = 100          # twin-window half-width
BOCPD_LAMBDA = 250  # prior mean run length
BOCPD_L = 50        # "recent" = run length <= this


def vol_series(X_train, feat):
    return pd.Series(X_train[feat].values).rolling(ROLLING).std().bfill().values


# ── incumbent + best HMM statistic, for side-by-side comparison ──────────────
def hmm_gamma(obs):
    try:
        h = SimpleHMM(n_iter=30, tol=1e-3)
        h.fit(obs)
        return h.predict_proba(obs)
    except Exception:
        return None


def stat_hmm_level(obs):
    g = hmm_gamma(obs)
    return 0.0 if g is None else float(g[-1, 1])


def stat_hmm_occupancy(obs):
    g = hmm_gamma(obs)
    if g is None:
        return 0.0
    h = len(g) // 2
    if h < 5:
        return 0.0
    return float(np.clip(g[h:, 1].mean() - g[:h, 1].mean(), 0.0, 1.0))


# ── alternative 1: twin-window KS ────────────────────────────────────────────
def stat_ks_twin(obs):
    """Two-sample KS D statistic between the last KS_T points and the previous KS_T."""
    T = min(KS_T, len(obs) // 2)
    if T < 20:
        return 0.0
    recent, earlier = obs[-T:], obs[-2 * T:-T]
    if len(earlier) < T:
        return 0.0
    try:
        return float(sstats.ks_2samp(recent, earlier).statistic)
    except Exception:
        return 0.0


# ── alternative 2: BOCPD (Adams & MacKay 2007), NIG conjugate ────────────────
def stat_bocpd(obs):
    """
    P(run length <= BOCPD_L) at the final timestep.

    Standard recursion: grow each run length by the predictive probability of the
    new datum, and accumulate the changepoint mass into run length 0. Predictive
    is Student-t from the Normal-Inverse-Gamma posterior of each run.
    """
    x = np.asarray(obs, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 30:
        return 0.0
    # Standardise so the prior is scale-free across features.
    sd = x.std()
    if sd <= 0:
        return 0.0
    x = (x - x.mean()) / sd

    H = 1.0 / BOCPD_LAMBDA
    mu = np.array([0.0]); kap = np.array([1.0])
    alp = np.array([1.0]); bet = np.array([1.0])
    R = np.array([1.0])                      # run-length posterior

    for t in range(n):
        # Student-t predictive for each run length
        df = 2 * alp
        scale = np.sqrt(bet * (kap + 1.0) / (alp * kap))
        pred = sstats.t.pdf(x[t], df=df, loc=mu, scale=scale)
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

        growth = R * pred * (1.0 - H)        # run continues
        cp = float(np.sum(R * pred * H))     # changepoint here
        R = np.concatenate(([cp], growth))
        s = R.sum()
        if not np.isfinite(s) or s <= 0:
            return 0.0
        R /= s

        # NIG updates, prepending the prior for the new run length 0
        mu = np.concatenate(([0.0], (kap * mu + x[t]) / (kap + 1.0)))
        bet = np.concatenate(([1.0],
                              bet + (kap * (x[t] - mu[1:]) ** 2) / (2.0 * (kap + 1.0))))
        alp = np.concatenate(([1.0], alp + 0.5))
        kap = np.concatenate(([1.0], kap + 1.0))

        # Truncate the tail to keep this O(n * K) rather than O(n^2)
        if len(R) > 400:
            R, mu, kap, alp, bet = R[:400], mu[:400], kap[:400], alp[:400], bet[:400]
            R /= R.sum()
    return float(np.sum(R[:min(BOCPD_L + 1, len(R))]))


STATS = {"hmm_level [incumbent]": stat_hmm_level,
         "hmm_occupancy": stat_hmm_occupancy,
         "ks_twin": stat_ks_twin,
         "bocpd": stat_bocpd}


def run(fn, feat, folds, k, cache):
    thr = AdaptiveRegimeThreshold(method="percentile", lookback=50,
                                  percentile_k=k, cooldown=1)
    vals, fired = [], []
    for fi, (X_tr, *_r) in enumerate(folds):
        key = (id(fn), feat, fi)
        if key not in cache:
            obs = vol_series(X_tr, feat)
            cache[key] = (fn(obs), len(obs) < WARMUP)
        p, warm = cache[key]
        vals.append(p)
        fired.append(False if warm else bool(thr.update(p)))
    return vals, [i for i, v in enumerate(fired) if v]


def main():
    d = pickle.load(open(f"data/{LEVEL}.pkl", "rb"))
    X, y = d["X"], d["y"]
    feats = list(X.columns)
    folds = walk_forward_folds(X, y, n_splits=8, gap=5, min_train=150)
    ph = [p["phase"] for p in classify_folds(folds, SWITCH)]
    post0 = ph.index("post")
    sig = [c for c in SIGNALS if c in feats]
    noi = [c for c in NOISE if c in feats]
    cache = {}
    KS_GRID = [75, 85, 90]

    print("=" * 98)
    print("  HMM vs TWIN-WINDOW KS vs BOCPD  —  noise null control")
    print(f"  {LEVEL}   phases={ph}   earliest causally possible fire = fold {post0}")
    print(f"  bar: beat occupancy (hit@4 0/5, FP 0, NULL 2) with hit@4>=2/5, FP=0, NULL<=2")
    print("=" * 98)

    rows = []
    for sname, fn in STATS.items():
        for k in KS_GRID:
            h4 = hge = fp = nl = 0
            det = []
            for c in sig:
                _v, f = run(fn, c, folds, k, cache)
                h4 += int(post0 in f)
                hge += int(any(i >= post0 for i in f))
                fp += sum(1 for i in f if i < post0)
                det.append(f"{c[-1]}:{f if f else '-'}")
            for c in noi:
                _v, f = run(fn, c, folds, k, cache)
                fp += sum(1 for i in f if i < post0)
                nl += len(f)
            rows.append((sname, k, h4, hge, fp, nl, " ".join(det)))

    print(f"  {'statistic':<24}{'k':<5}{'hit@4':>8}{'hit@>=4':>10}{'FP':>6}{'NULL':>7}"
          f"   signal fire folds")
    print("  " + "-" * 94)
    for sname, k, h4, hge, fp, nl, det in rows:
        flag = "  <-- FP=0" if fp == 0 else ""
        print(f"  {sname:<24}{k:<5}{f'{h4}/{len(sig)}':>8}{f'{hge}/{len(sig)}':>10}"
              f"{fp:>6}{nl:>7}   {det}{flag}")

    clean = [r for r in rows if r[4] == 0]
    print()
    print("=" * 98)
    if not clean:
        print("  No configuration achieves FP = 0.")
    else:
        best = max(clean, key=lambda r: (r[2], r[3], -r[5]))
        sname, k, h4, hge, fp, nl, _d = best
        print(f"  BEST ZERO-FALSE-POSITIVE: {sname}  k={k}")
        print(f"    hit@4={h4}/{len(sig)}  hit@>=4={hge}/{len(sig)}  NULL={nl}")
        passed = h4 >= 2 and nl <= 2
        print(f"    pre-registered bar (hit@4>=2/5, FP=0, NULL<=2): "
              f"{'PASSED' if passed else 'NOT MET'}")
    print()
    print("  CAVEAT: one benchmark draw, one break. Four detector experiments have")
    print("  now selected operating points from THIS draw -- any winner must be")
    print("  confirmed on a fresh draw before it goes in the paper.")


if __name__ == "__main__":
    main()
