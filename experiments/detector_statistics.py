"""
Three regime-change statistics, judged against a NOISE NULL CONTROL.

WHY
───
The current detector computes

    p_trans = gamma[-1, 1]

the posterior probability that the LAST observation of the training window sits
in the high-volatility state. That is a LEVEL statistic, not a CHANGE statistic:
it asks "is it turbulent right now?", never "did something change?". Any
volatility cluster landing at the end of a window fires it.

Measured consequence (experiments/detector_statistics.py run on v2_regime_switch,
break at fold 4): the detector fires on noise_0, noise_1 and noise_2 --- features
with no relationship to y and no regime structure whatsoever --- as readily as on
the real signals, and its false alarm at fold 2 (p_trans=0.9356) is MORE confident
than the genuine break at fold 4 (0.6902). Roughly three quarters of all p_trans
values are pinned at 0.000 or 1.000, which is the saturation the code's own
AdaptiveRegimeThreshold warning describes.

THE NULL CONTROL
────────────────
noise_* features are pure nuisance. A correct change statistic must fire on
signal_* at the first causally-possible fold and essentially NEVER on noise_*.
That is a falsifiable bar; the current statistic fails it outright.

STATISTICS COMPARED
───────────────────
  level      p = gamma[-1, 1]
             the incumbent. Level, not change.

  deviation  p = clip(gamma[-1,1] - mean(gamma[:,1]), 0, 1)
             current state MINUS the window's own baseline occupancy. A window
             that is turbulent throughout no longer fires, because the last
             observation is unremarkable relative to its own history.

  occupancy  p = clip(mean(gamma[H:,1]) - mean(gamma[:H,1]), 0, 1),  H = n//2
             how much MORE of the recent half sits in the high-vol state than
             the earlier half. Averaging over halves rather than reading a single
             endpoint makes it far less sensitive to where a cluster happens to
             land -- which is precisely the failure mode of `level`.

All three feed the SAME AdaptiveRegimeThreshold, so any difference is
attributable to the statistic rather than to the gating.

PRE-REGISTERED SCORING
──────────────────────
  hit  : fires at the earliest causally possible fold (first POST fold) --- a
         walk-forward learner cannot react before the break enters TRAINING.
  FP   : any fire before that fold, on ANY feature.
  null : any fire at all on a noise_* feature, at any fold.
A statistic is better only if it raises hits while lowering BOTH FP and null.

Run:  python experiments/detector_statistics.py
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

# step7_ablation.py is a script and refuses import; exec its head for SimpleHMM.
_s7 = open("step7_ablation.py").read().split("#  MASTER ABLATION LOOP")[0]
_ns = {}
exec(compile(_s7, "step7_ablation<head>", "exec"), _ns)
SimpleHMM = _ns["SimpleHMM"]

LEVEL, SWITCH = "v2_regime_switch", 500
ROLLING = 20
WARMUP_MIN_OBS = 150
SIGNALS = ["signal_0", "signal_1", "signal_2", "signal_3", "signal_4"]
NOISE = ["noise_0", "noise_1", "noise_2", "noise_3", "noise_4"]


def gamma_for(X_train, feat_name):
    """Posterior state occupancies for one feature's rolling-volatility series."""
    obs = pd.Series(X_train[feat_name].values).rolling(ROLLING).std().bfill().values
    if len(obs) < 30:
        return None, True
    is_warmup = len(obs) < WARMUP_MIN_OBS
    hmm = SimpleHMM(n_iter=30, tol=1e-3)
    try:
        hmm.fit(obs)
        return hmm.predict_proba(obs), is_warmup
    except Exception:
        return None, is_warmup


def stat_level(g):
    return float(g[-1, 1])


def stat_deviation(g):
    return float(np.clip(g[-1, 1] - g[:, 1].mean(), 0.0, 1.0))


def stat_occupancy(g):
    h = len(g) // 2
    if h < 5:
        return 0.0
    return float(np.clip(g[h:, 1].mean() - g[:h, 1].mean(), 0.0, 1.0))


STATS = {"level": stat_level,           # incumbent
         "deviation": stat_deviation,
         "occupancy": stat_occupancy}


def run(stat_name, feat_name, folds):
    thr = AdaptiveRegimeThreshold(method="percentile", lookback=50,
                                  percentile_k=85.0)
    fn = STATS[stat_name]
    ps, fired = [], []
    for (X_tr, _y_tr, _X_te, _y_te, _te) in folds:
        g, is_warm = gamma_for(X_tr, feat_name)
        if g is None:
            ps.append(0.0); fired.append(False); continue
        p = fn(g)
        ps.append(p)
        # Same warm-up guard the production detector uses: compute for
        # diagnostics but do not act on it.
        fired.append(False if is_warm else bool(thr.update(p)))
    return ps, fired


def main():
    d = pickle.load(open(f"data/{LEVEL}.pkl", "rb"))
    X, y = d["X"], d["y"]
    feats = list(X.columns)
    folds = walk_forward_folds(X, y, n_splits=8, gap=5, min_train=150)
    ph = [p["phase"] for p in classify_folds(folds, SWITCH)]
    post0 = ph.index("post")

    signals = [c for c in SIGNALS if c in feats]
    noise = [c for c in NOISE if c in feats]

    print("=" * 100)
    print("  REGIME-CHANGE STATISTICS vs NOISE NULL CONTROL")
    print(f"  {LEVEL}   phases={ph}")
    print(f"  earliest causally possible fire = fold {post0}")
    print("  * = triggered")
    print("=" * 100)

    summary = {}
    for sname in STATS:
        print(f"\n── statistic: {sname}"
              f"{'   [INCUMBENT]' if sname == 'level' else ''}")
        print(f"   {'observable':<12}{'role':<8}"
              + "".join(f"f{i}".rjust(8) for i in range(len(ph)))
              + "    fires")
        print("   " + "-" * 92)
        hits = fps = nulls = 0
        for c in signals + noise:
            ps, fired = run(sname, c, folds)
            fires = [i for i, v in enumerate(fired) if v]
            is_noise = c in noise
            if not is_noise:
                hits += int(post0 in fires)
            fps += sum(1 for i in fires if i < post0)
            if is_noise:
                nulls += len(fires)
            cells = "".join(f"{p:>7.3f}*" if t else f"{p:>8.3f}"
                            for p, t in zip(ps, fired))
            print(f"   {c:<12}{'noise' if is_noise else 'signal':<8}{cells}    {fires}")
        summary[sname] = (hits, fps, nulls, len(signals), len(noise))

    print()
    print("=" * 100)
    print("  SCORECARD   (hit = fires at fold %d on a signal; FP = any fire before"
          " it; null = ANY fire on noise)" % post0)
    print("=" * 100)
    print(f"  {'statistic':<14}{'hits':>12}{'false pos':>12}{'NULL fires':>14}"
          f"   verdict")
    print("  " + "-" * 84)
    base = summary["level"]
    for sname, (h, f, n, ns_, nn) in summary.items():
        if sname == "level":
            verdict = "incumbent"
        else:
            better = (h >= base[0]) and (f <= base[1]) and (n < base[2])
            verdict = ("BETTER on all three" if better and n == 0 else
                       "better" if better else
                       "not better")
        print(f"  {sname:<14}{f'{h}/{ns_}':>12}{f:>12}{f'{n}':>14}   {verdict}")
    print()
    print("  A statistic that fires on noise is not detecting regime change --")
    print("  it is detecting volatility clustering, which every autocorrelated")
    print("  series has. NULL fires must go to ~0 for the detector to be sound.")


if __name__ == "__main__":
    main()
