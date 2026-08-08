"""
Can the `occupancy` change-statistic reach fold-4 detection at zero false alarms?

CONTEXT
───────
experiments/detector_statistics.py compared three statistics against a noise
null control on v2_regime_switch (break enters TRAINING at fold 4):

    statistic        hits@f4   false pos   NULL fires
    level (current)      2/5          15           13
    deviation            0/5           8            6
    occupancy            0/5           0            2

`occupancy` -- mean P(high-vol) over the recent half of the window minus the
earlier half -- removed every pre-break false alarm and nearly every noise fire.
It detected the break on 3 of 5 signals but at fold 5, one fold LATE, because at
percentile_k=85 its fold-4 values (0.031 / 0.155 / 0.230) sit below its own
threshold.

It is therefore spending a ZERO false-positive budget. This sweep asks whether
loosening the gate converts that unused budget into earlier detection before
false alarms reappear.

SCORING (pre-registered, and separating latency from correctness -- the previous
scorecard conflated them and mislabelled `occupancy` as "not better")
    hit@4     fires at fold 4 exactly (the earliest CAUSALLY possible fold)
    hit@>=4   fires at any post-break fold: correct but possibly late
    FP        any fire before fold 4, on any feature   <- must stay 0
    NULL      any fire at all on a noise_* feature     <- must stay ~0

The operating point of interest is the loosest gate that still holds FP = 0.

Run:  python experiments/detector_threshold_sweep.py
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
KS = [50, 55, 60, 65, 70, 75, 80, 85]
COOLDOWNS = [0, 1]


def occupancy(X_train, feat_name):
    obs = pd.Series(X_train[feat_name].values).rolling(ROLLING).std().bfill().values
    if len(obs) < 30:
        return 0.0, True
    warm = len(obs) < WARMUP
    try:
        hmm = SimpleHMM(n_iter=30, tol=1e-3)
        hmm.fit(obs)
        g = hmm.predict_proba(obs)
    except Exception:
        return 0.0, warm
    h = len(g) // 2
    if h < 5:
        return 0.0, warm
    return float(np.clip(g[h:, 1].mean() - g[:h, 1].mean(), 0.0, 1.0)), warm


def run(feat_name, folds, k, cooldown, cache):
    thr = AdaptiveRegimeThreshold(method="percentile", lookback=50,
                                  percentile_k=k, cooldown=cooldown)
    fired = []
    for fi, (X_tr, *_rest) in enumerate(folds):
        key = (feat_name, fi)
        if key not in cache:
            cache[key] = occupancy(X_tr, feat_name)
        p, warm = cache[key]
        fired.append(False if warm else bool(thr.update(p)))
    return [i for i, v in enumerate(fired) if v]


def main():
    d = pickle.load(open(f"data/{LEVEL}.pkl", "rb"))
    X, y = d["X"], d["y"]
    feats = list(X.columns)
    folds = walk_forward_folds(X, y, n_splits=8, gap=5, min_train=150)
    ph = [p["phase"] for p in classify_folds(folds, SWITCH)]
    post0 = ph.index("post")
    signals = [c for c in SIGNALS if c in feats]
    noise = [c for c in NOISE if c in feats]
    cache = {}          # HMM fits are the expensive part; reuse across k

    print("=" * 92)
    print("  OCCUPANCY STATISTIC — threshold sweep vs noise null control")
    print(f"  {LEVEL}   phases={ph}   earliest causally possible fire = fold {post0}")
    print("  target operating point: the LOOSEST gate that still holds FP = 0")
    print("=" * 92)
    print(f"  {'cooldown':<10}{'pct_k':<8}{'hit@4':>8}{'hit@>=4':>10}"
          f"{'FP':>6}{'NULL':>7}   signal fire folds")
    print("  " + "-" * 88)

    best = None
    for cd in COOLDOWNS:
        for k in KS:
            h4 = hge = fp = nl = 0
            detail = []
            for c in signals:
                f = run(c, folds, k, cd, cache)
                h4 += int(post0 in f)
                hge += int(any(i >= post0 for i in f))
                fp += sum(1 for i in f if i < post0)
                detail.append(f"{c[-1]}:{f if f else '-'}")
            for c in noise:
                f = run(c, folds, k, cd, cache)
                fp += sum(1 for i in f if i < post0)
                nl += len(f)
            flag = ""
            if fp == 0:
                score = (h4, hge, -nl)
                if best is None or score > best[0]:
                    best = (score, cd, k, h4, hge, nl)
                    flag = "  <-- best FP=0 so far"
            print(f"  {cd:<10}{k:<8}{f'{h4}/{len(signals)}':>8}"
                  f"{f'{hge}/{len(signals)}':>10}{fp:>6}{nl:>7}   "
                  f"{' '.join(detail)}{flag}")

    print()
    print("=" * 92)
    if best is None:
        print("  NO operating point achieves FP = 0. The statistic cannot be gated")
        print("  into usefulness on this draw; the observable or the HMM itself is")
        print("  the remaining suspect.")
    else:
        _s, cd, k, h4, hge, nl = best
        print(f"  BEST ZERO-FALSE-POSITIVE OPERATING POINT: percentile_k={k}, cooldown={cd}")
        print(f"    detects at fold {post0} on {h4}/{len(signals)} signals")
        print(f"    detects at some post-break fold on {hge}/{len(signals)} signals")
        print(f"    fires on noise {nl} time(s)")
        print()
        print(f"  Incumbent for comparison (level statistic, k=85, cooldown=1):")
        print(f"    2/5 at fold 4, but 15 pre-break false positives and 13 noise fires.")
    print()
    print("  CAVEAT: one benchmark draw, one break. Confirm on a second draw")
    print("  before treating any operating point as settled.")


if __name__ == "__main__":
    main()
