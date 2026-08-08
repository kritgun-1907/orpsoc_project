"""
Multi-draw detector evaluation: are these detectors distinguishable at all?

WHY THIS EXISTS
───────────────
Four experiments compared regime-change statistics on a SINGLE benchmark draw
(seed 42) and produced a confident ranking. A held-out draw (seed 1234)
overturned it:

                        seed 42            seed 1234
    statistic       hit@4  FP  NULL    hit@4  FP  NULL
    hmm_level         2/5  15    13      4/5  11     9
    hmm_occupancy     0/5   0     2      0/5   3     0
    ks_twin           1/5   3     4      0/5   8     6
    bocpd             5/5  20    15      4/5  20    18

`hmm_occupancy`'s zero false positives -- the entire basis for preferring it --
did not replicate. The incumbent's hit rate doubled. Conclusion: the differences
between these detectors may be smaller than the variance between draws, in which
case every ranking reported so far is a property of one 1000-row sample.

This script settles that by evaluating all four across N_DRAWS independent draws
and reporting the SPREAD, not just the mean. Seeds are averaged over data
stochasticity, which the 30-seed PSO runs elsewhere do NOT control for -- those
average over swarm randomness on a fixed dataset.

WHAT WOULD MAKE A DETECTOR USABLE
─────────────────────────────────
  * fires at the earliest causally possible fold on most draws, AND
  * essentially never fires before the break, AND
  * essentially never fires on noise features,
  * with all three STABLE across draws (small spread).
`frac_FP0` -- the fraction of draws achieving zero false positives -- is the
headline number: a detector that manages it on 1 draw in 8 is not a detector.

Run:  python experiments/detector_multidraw.py [n_draws]
Writes results/detector_multidraw.json
"""
import os
import sys
import json
import pickle
import importlib

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

from orpsoc_runner import pin_threads, default_workers
pin_threads(1)

from joblib import Parallel, delayed
from orpsoc_utils import (walk_forward_folds, classify_folds,
                          AdaptiveRegimeThreshold)

N_DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
BASE_SEED = 9000            # distinct from 42 (tuned on) and 1234 (held out)
PCT_K = 85                  # one gate for all; comparing statistics, not gates
SIGNALS = ["signal_0", "signal_1", "signal_2", "signal_3", "signal_4"]
NOISE = ["noise_0", "noise_1", "noise_2", "noise_3", "noise_4"]
SWITCH, ROLLING, WARMUP = 500, 20, 150


def _stats_module():
    """Reuse the statistic implementations rather than re-deriving them."""
    import experiments.detector_alternatives as da
    return da


def one_draw(seed):
    import pandas as pd
    da = _stats_module()
    m = importlib.import_module("make_benchmark_v2")
    X, y, _base = m.generate_v2("regime_switch", seed=seed)
    feats = list(X.columns)
    folds = walk_forward_folds(X, y, n_splits=8, gap=5, min_train=150)
    ph = [p["phase"] for p in classify_folds(folds, SWITCH)]
    post0 = ph.index("post")
    sig = [c for c in SIGNALS if c in feats]
    noi = [c for c in NOISE if c in feats]

    fns = {"hmm_level": da.stat_hmm_level,
           "hmm_occupancy": da.stat_hmm_occupancy,
           "ks_twin": da.stat_ks_twin,
           "bocpd": da.stat_bocpd}

    # Volatility series per (feature, fold), computed once and shared.
    vol = {}
    for c in sig + noi:
        for fi, (X_tr, *_r) in enumerate(folds):
            s = pd.Series(X_tr[c].values).rolling(ROLLING).std().bfill().values
            vol[(c, fi)] = (s, len(s) < WARMUP)

    out = {}
    for sname, fn in fns.items():
        h4 = hge = fp = nl = 0
        for c in sig + noi:
            thr = AdaptiveRegimeThreshold(method="percentile", lookback=50,
                                          percentile_k=PCT_K, cooldown=1)
            fires = []
            for fi in range(len(folds)):
                s, warm = vol[(c, fi)]
                p = fn(s)
                if not warm and thr.update(p):
                    fires.append(fi)
            if c in sig:
                h4 += int(post0 in fires)
                hge += int(any(i >= post0 for i in fires))
            fp += sum(1 for i in fires if i < post0)
            if c in noi:
                nl += len(fires)
        out[sname] = {"hit4": h4, "hitge": hge, "fp": fp, "null": nl}
    return seed, out


def main():
    seeds = [BASE_SEED + i for i in range(N_DRAWS)]
    print("=" * 92)
    print(f"  MULTI-DRAW DETECTOR EVALUATION — {N_DRAWS} independent draws")
    print(f"  seeds {seeds[0]}..{seeds[-1]}  (disjoint from 42 and 1234)")
    print(f"  gate: percentile_k={PCT_K}, cooldown=1 for every statistic")
    print("=" * 92, flush=True)

    res = Parallel(n_jobs=default_workers(), verbose=5)(
        delayed(one_draw)(s) for s in seeds)
    per = {}
    for _s, o in res:
        for k, v in o.items():
            per.setdefault(k, []).append(v)

    with open("results/detector_multidraw.json", "w") as f:
        json.dump({"n_draws": N_DRAWS, "seeds": seeds, "pct_k": PCT_K,
                   "per_statistic": per}, f, indent=2)

    print()
    print("=" * 92)
    print(f"  RESULTS over {N_DRAWS} draws   (mean +- sd, [min, max])")
    print("=" * 92)
    print(f"  {'statistic':<16}{'hit@4 /5':>18}{'FP':>18}{'NULL fires':>18}"
          f"{'frac FP=0':>12}")
    print("  " + "-" * 84)
    summary = {}
    for sname, rows in per.items():
        h4 = np.array([r["hit4"] for r in rows], float)
        fp = np.array([r["fp"] for r in rows], float)
        nl = np.array([r["null"] for r in rows], float)
        f0 = float(np.mean(fp == 0))
        summary[sname] = {"hit4_mean": h4.mean(), "hit4_sd": h4.std(),
                          "fp_mean": fp.mean(), "fp_sd": fp.std(),
                          "null_mean": nl.mean(), "null_sd": nl.std(),
                          "frac_fp0": f0}
        print(f"  {sname:<16}"
              f"{f'{h4.mean():.2f}+-{h4.std():.2f} [{h4.min():.0f},{h4.max():.0f}]':>18}"
              f"{f'{fp.mean():.1f}+-{fp.std():.1f} [{fp.min():.0f},{fp.max():.0f}]':>18}"
              f"{f'{nl.mean():.1f}+-{nl.std():.1f} [{nl.min():.0f},{nl.max():.0f}]':>18}"
              f"{f0:>12.2f}")

    print()
    print("  INTERPRETATION")
    best_fp0 = max(summary.items(), key=lambda kv: kv[1]["frac_fp0"])
    print(f"    highest frac(FP=0): {best_fp0[0]} at {best_fp0[1]['frac_fp0']:.2f}")
    if best_fp0[1]["frac_fp0"] < 0.5:
        print("    NO statistic achieves zero false positives on even half the")
        print("    draws. None of these detectors is usable as a regime gate, and")
        print("    single-draw rankings between them are not meaningful.")
    lv, oc = summary.get("hmm_level"), summary.get("hmm_occupancy")
    if lv and oc:
        sep = abs(lv["fp_mean"] - oc["fp_mean"])
        pooled = np.hypot(lv["fp_sd"], oc["fp_sd"])
        print(f"    level vs occupancy on FP: separation {sep:.1f}, "
              f"pooled sd {pooled:.1f} -> "
              f"{'separable' if sep > 2 * pooled else 'NOT separable at 2 sd'}")
    print()
    print("  Every earlier detector conclusion was drawn from ONE draw. Treat this")
    print("  table, not those, as the reportable result.")


if __name__ == "__main__":
    main()
