"""
verify_data.py — integrity gate for every dataset the pipeline will consume
===========================================================================
Motivated by a real failure: data/fama_french.pkl sat on disk since June built
against an OLDER target definition (return direction) while the code had since
moved to a volatility-regime target. Nothing detected it. Its stored labels
agreed with a target rebuilt from its own `base` series only 47% of the time --
worse than a coin flip -- and every experiment that read that file silently
measured the wrong thing.

The check that would have caught it in one second: rebuild the target from the
dataset's OWN stored `base` using the documented construction, and confirm it
reproduces the stored `y`. A dataset that cannot regenerate its own labels is
stale or corrupt, and nothing downstream of it is trustworthy.

Exit code 0 = all datasets sound, 1 = at least one failed. Wire it in front of
any long run.

Run:  python experiments/verify_data.py [--strict]
"""
from __future__ import annotations

import glob
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import roc_auc_score

# Datasets whose label is the documented volatility-regime target:
#   y = 1{ rolling_vol.shift(-horizon) > rolling_vol.rolling(252).median() }
# with rolling_vol = base.rolling(20).std(). These can regenerate their labels.
REAL_VOL_TARGET = {"sector_etf", "fama_french", "bonds", "commodities"}
HORIZON, VOL_WIN, MED_WIN = 5, 20, 252

# Minimum agreement before a dataset is considered self-consistent. The check is
# exact in principle; the tolerance only absorbs boundary rows lost to the
# align/trim step.
AGREE_MIN = 0.99


def check_real(name, X, y, base):
    """Rebuild the vol-regime target from `base` and compare to stored `y`."""
    b = pd.Series(np.asarray(base, dtype=float))
    rv = b.rolling(VOL_WIN).std()
    rebuilt = (rv.shift(-HORIZON) > rv.rolling(MED_WIN).median()).astype(int)
    m = rv.shift(-HORIZON).notna() & rv.rolling(MED_WIN).median().notna()
    if m.sum() < 100:
        return False, "too few comparable rows to verify"
    agree = float((rebuilt[m].values == pd.Series(np.asarray(y))[m].values).mean())
    ok = agree >= AGREE_MIN
    detail = f"labels reproduce from own base: {agree:6.1%}"
    if not ok:
        detail += ("  <-- STALE: the stored labels do not match the target the "
                   "code builds. Regenerate this dataset.")
    return ok, detail


def check_common(name, X, y):
    """Structural checks that apply to every dataset."""
    problems = []
    if X.isna().to_numpy().any():
        problems.append(f"{int(X.isna().to_numpy().sum())} NaNs in X")
    allnan = [c for c in X.columns if X[c].isna().all()]
    if allnan:
        problems.append(f"all-NaN columns: {allnan}")
    if len(X) != len(y):
        problems.append(f"X/y length mismatch {len(X)} vs {len(y)}")
    if pd.Series(y).nunique() < 2:
        problems.append("y is single-class")
    bal = float(np.mean(y))
    if not 0.2 <= bal <= 0.8:
        problems.append(f"extreme class balance {bal:.3f}")
    return problems


def main(strict=False):
    paths = sorted(p for p in glob.glob("data/*.pkl")
                   if not os.path.basename(p).startswith("raw_")
                   and "checkpoint" not in os.path.basename(p)
                   and "STALE" not in os.path.basename(p))
    if not paths:
        print("no datasets found in data/ -- nothing to verify")
        return 1

    print("=" * 86)
    print("  DATASET INTEGRITY")
    print("=" * 86)
    print(f"  {'dataset':<26}{'shape':>13}{'y-mean':>9}{'max|corr|':>11}  status")
    failures = []
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        try:
            d = pickle.load(open(p, "rb"))
            X, y, base = d["X"], d["y"], d.get("base")
        except Exception as e:
            print(f"  {name:<26}{'—':>13}{'—':>9}{'—':>11}  UNREADABLE: {e}")
            failures.append(name)
            continue

        problems = check_common(name, X, y)
        note = ""
        if name in REAL_VOL_TARGET and base is not None:
            ok, detail = check_real(name, X, y, base)
            note = detail
            if not ok:
                problems.append("label/base mismatch")

        cors = [abs(roc_auc_score(y, X[c]) - 0.5) * 2 for c in X.columns]
        status = "OK" if not problems else "FAIL: " + "; ".join(problems)
        if problems:
            failures.append(name)
        print(f"  {name:<26}{str(X.shape):>13}{np.mean(y):>9.3f}"
              f"{max(cors):>11.3f}  {status}")
        if note:
            print(f"  {'':<26}{'':>33}  {note}")

    print("\n" + "=" * 86)
    if failures:
        print(f"  {len(failures)} DATASET(S) FAILED: {', '.join(failures)}")
        print("  Do not start a long run against these.")
        return 1
    print(f"  all {len(paths)} datasets sound")
    return 0


if __name__ == "__main__":
    sys.exit(main(strict="--strict" in sys.argv))
