"""
Recent-tail change statistic: the middle ground between `deviation` and `occupancy`.

THE AXIS
────────
The three statistics tested so far differ in ONE thing -- how many observations
the estimate rests on -- and that single axis explains all their behaviour:

    statistic    rests on            FP   NULL   detects
    level        1 point (endpoint)  15     13   fold 4  (but unusable)
    deviation    1 point, rescaled    8      6   never
    occupancy    ~half the window     0      2   fold 5  (one fold late)

More averaging -> fewer false alarms -> more latency. `occupancy` is late
because at fold 4 the break has only just entered the window and sits at its very
end, so a half-window mean dilutes it to nothing.

THE HYPOTHESIS
──────────────
Average over the RECENT TAIL only:

    p = clip( mean(gamma[-T:, 1]) - mean(gamma[:-T, 1]), 0, 1 )

With T ~ the rolling-volatility window, the newly-arrived post-break data
DOMINATES the tail instead of being diluted, while T > 1 still buys the variance
reduction that a single endpoint cannot. T=1 degenerates to `deviation`;
T = n//2 degenerates to `occupancy`. So this sweep interpolates between two
measured endpoints rather than proposing something unrelated.

PRE-REGISTERED SCORING (identical to the previous two experiments)
    hit@4     fires at fold 4, the earliest CAUSALLY possible fold
    hit@>=4   fires at any post-break fold (correct, possibly late)
    FP        any fire before fold 4, on any feature       <- must be 0
    NULL      any fire at all on a noise_* feature         <- must be ~0

SUCCESS BAR, fixed before running: some (T, percentile_k) achieves hit@4 >= 3/5
with FP = 0 and NULL <= 2. Anything less and the tail idea is not better than
`occupancy` at k=85 (3/5 at fold 5, FP=0, NULL=2), which stays the incumbent.

Run:  python experiments/detector_tail.py
"""
import os
import sys
import pickle

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

import pandas as pd
from orpsoc_utils import (walk_forward_folds, classify_folds,
                          AdaptiveRegimeThreshold)

_s7 = open("step7_ablation.py").read().split("#  MASTER ABLATION LOOP")[0]
_ns = {}
exec(compile(_s7, "step7_ablation<head>", "exec"), _ns)
SimpleHMM = _ns["SimpleHMM"]

LEVEL, SWITCH, ROLLING, WARMUP = "v2_regime_switch", 500, 20, 150
SIGNALS = ["signal_0", "signal_1", "signal_2", "signal_3", "signal_4"]
NOISE = ["noise_0", "noise_1", "noise_2", "noise_3", "noise_4"]
TAILS = [10, 20, 30, 40, 60]          # T=20 matches the rolling window
KS = [65, 75, 80, 85, 90]


def gamma_of(X_train, feat_name, cache):
    key = (feat_name, len(X_train))
    if key in cache:
        return cache[key]
    obs = pd.Series(X_train[feat_name].values).rolling(ROLLING).std().bfill().values
    out = (None, True)
    if len(obs) >= 30:
        warm = len(obs) < WARMUP
        try:
            hmm = SimpleHMM(n_iter=30, tol=1e-3)
            hmm.fit(obs)
            out = (hmm.predict_proba(obs), warm)
        except Exception:
            out = (None, warm)
    cache[key] = out
    return out


def tail_stat(g, T):
    if g is None or len(g) < T + 10:
        return 0.0
    return float(np.clip(g[-T:, 1].mean() - g[:-T, 1].mean(), 0.0, 1.0))


def run(feat, folds, T, k, cache):
    thr = AdaptiveRegimeThreshold(method="percentile", lookback=50,
                                  percentile_k=k, cooldown=1)
    fired, vals = [], []
    for (X_tr, *_r) in folds:
        g, warm = gamma_of(X_tr, feat, cache)
        p = tail_stat(g, T)
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

    print("=" * 96)
    print("  RECENT-TAIL CHANGE STATISTIC — sweep over tail length T and gate k")
    print(f"  {LEVEL}   phases={ph}   earliest causally possible fire = fold {post0}")
    print(f"  bar: hit@4 >= 3/5 with FP=0 and NULL<=2  (else `occupancy` k=85 stays incumbent)")
    print("=" * 96)
    print(f"  {'T':<6}{'pct_k':<8}{'hit@4':>8}{'hit@>=4':>10}{'FP':>6}{'NULL':>7}"
          f"   signal fire folds")
    print("  " + "-" * 92)

    best = None
    for T in TAILS:
        for k in KS:
            h4 = hge = fp = nl = 0
            det = []
            for c in sig:
                _v, f = run(c, folds, T, k, cache)
                h4 += int(post0 in f)
                hge += int(any(i >= post0 for i in f))
                fp += sum(1 for i in f if i < post0)
                det.append(f"{c[-1]}:{f if f else '-'}")
            for c in noi:
                _v, f = run(c, folds, T, k, cache)
                fp += sum(1 for i in f if i < post0)
                nl += len(f)
            flag = ""
            if fp == 0 and (best is None or (h4, hge, -nl) > best[0]):
                best = ((h4, hge, -nl), T, k, h4, hge, nl)
                flag = "  <-- best FP=0"
            print(f"  {T:<6}{k:<8}{f'{h4}/{len(sig)}':>8}{f'{hge}/{len(sig)}':>10}"
                  f"{fp:>6}{nl:>7}   {' '.join(det)}{flag}")

    print()
    print("=" * 96)
    if best is None:
        print("  NO (T, k) reaches FP = 0. The tail statistic does not beat `occupancy`.")
    else:
        _s, T, k, h4, hge, nl = best
        passed = (h4 >= 3) and (nl <= 2)
        print(f"  BEST ZERO-FALSE-POSITIVE POINT: T={T}, percentile_k={k}")
        print(f"    hit@4 = {h4}/{len(sig)}   hit@>=4 = {hge}/{len(sig)}   NULL = {nl}")
        print()
        print(f"  Pre-registered bar (hit@4>=3/5, FP=0, NULL<=2): "
              f"{'PASSED' if passed else 'NOT MET'}")
        if not passed:
            print("  -> `occupancy` at k=85 (3/5 at fold 5, FP=0, NULL=2) remains the")
            print("     best available detector. The tail idea is not an improvement.")
    print()
    print("  CAVEAT: one benchmark draw, one break. Any operating point chosen here")
    print("  must be confirmed on a fresh draw before it goes in the paper --")
    print("  picking (T, k) off this table IS tuning on the test.")


if __name__ == "__main__":
    main()
