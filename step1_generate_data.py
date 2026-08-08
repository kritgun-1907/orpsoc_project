"""
STEP 1 — Generate Synthetic Data
===================================
This file generates the four complexity levels.

WHY SYNTHETIC DATA FIRST?
──────────────────────────
With real financial data, you never know the ground truth.
You cannot tell whether a bad result is OrPSOC's fault or
just inherent market noise.

With synthetic data YOU CONTROL everything:
  - Which features are truly predictive
  - Exactly when the regime changes
  - How much noise surrounds the signal

So when OrPSOC fails or succeeds, you know EXACTLY why.
That is what makes it a scientific experiment.

THE FOUR LEVELS:
  Level 1 — White Noise    : No temporal structure at all
  Level 2 — AR(1)          : Temporal memory, but stable (stationary)
  Level 3 — Drift          : Non-stationary mean (slowly shifting)
  Level 4 — Regime Switch  : Two distinct regimes, abrupt change at t=500

Run with:
    python step1_generate_data.py
"""

# ── import guard ─────────────────────────────────────────────────────────────
# This file is a SCRIPT, not a module. It executes its whole pipeline at module
# level, so `import step1_generate_data` runs the entire thing as a side effect --
# for step7_ablation that is a 4-hour ablation triggered by an innocent-looking
# import. Fail loudly instead.
#
# `globals().get("__name__", ...)` rather than a bare `__name__`: helpers are
# reused by exec'ing the section above a marker into a fresh namespace (see
# experiments/apsoll_sweep.py), and that namespace has no __name__ at all.
if globals().get("__name__", "__main__") != "__main__":
    raise ImportError(
        "step1_generate_data.py is a script, not an importable module -- importing it would "
        "execute the full pipeline. To reuse a helper, exec the section above "
        "the main loop into a fresh namespace (see experiments/apsoll_sweep.py)."
    )
# ─────────────────────────────────────────────────────────────────────────────


import numpy as np
import pandas as pd
import pickle
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.stats import levene
import os

os.makedirs("data",    exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("plots",   exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate(
    level:           str,
    n_steps:         int   = 1000,
    n_features:      int   = 50,
    n_signal:        int   = 5,
    ar_coef:         float = 0.70,
    drift_rate:      float = 0.0003,
    regime_switch:   int   = 500,
    seed:            int   = 42,
):
    """
    Generate one synthetic dataset at a given complexity level.

    PARAMETERS
    ──────────
    level         : "white_noise" | "ar1" | "drift" | "regime_switch"
    n_steps       : How many time steps (rows)
    n_features    : Total features (signal + noise)
    n_signal      : How many features are truly predictive
    ar_coef       : AR(1) coefficient. Must be |phi| < 1 for stationarity.
    drift_rate    : How fast the mean drifts upward (Level 3)
    regime_switch : When the regime changes (Level 4)
    seed          : Random seed for reproducibility

    RETURNS
    ───────
    X    : pd.DataFrame, shape (n_steps, n_features)
    y    : pd.Series, shape (n_steps,), binary 0/1
    base : np.ndarray, shape (n_steps,), the hidden signal
    """

    rng  = np.random.RandomState(seed)
    base = np.zeros(n_steps)
    base[0] = rng.randn()

    if level == "white_noise":
        base = rng.randn(n_steps)

    elif level == "ar1":
        for t in range(1, n_steps):
            base[t] = ar_coef * base[t - 1] + rng.randn()

    elif level == "drift":
        for t in range(1, n_steps):
            base[t] = drift_rate * t + ar_coef * base[t - 1] + rng.randn()

    elif level == "regime_switch":
        for t in range(1, n_steps):
            if t < regime_switch:
                base[t] = ar_coef  * base[t - 1] + 0.5 * rng.randn()
            else:
                base[t] = 0.3 * base[t - 1] + 2.0 * rng.randn()

    else:
        raise ValueError(f"Unknown level: {level}. "
                         f"Choose: white_noise | ar1 | drift | regime_switch")

    # ── Build feature matrix ───────────────────────────────────────────────────
    feature_data = {}

    for i in range(n_signal):
        if level == "regime_switch":
            sig = np.zeros(n_steps)
            if i < 3:
                sig[:regime_switch] = base[:regime_switch] + 0.3 * rng.randn(regime_switch)
                sig[regime_switch:] = rng.randn(n_steps - regime_switch)
            else:
                sig[:regime_switch] = rng.randn(regime_switch)
                sig[regime_switch:] = base[regime_switch:] + 0.3 * rng.randn(n_steps - regime_switch)
        else:
            sig = base + 0.3 * rng.randn(n_steps)
        feature_data[f"signal_{i}"] = sig

    for i in range(n_features - n_signal):
        feature_data[f"noise_{i}"] = rng.randn(n_steps)

    X = pd.DataFrame(feature_data)
    y = pd.Series((base > 0).astype(int), name="target")

    return X, y, base


# ══════════════════════════════════════════════════════════════════════════════
#  STATIONARITY TEST
# ══════════════════════════════════════════════════════════════════════════════

def test_stationarity(series: np.ndarray, name: str,
                      method: str = "adf",
                      split_at: int | None = None) -> dict:
    """
    Stationarity test.

    method="adf"            : Augmented Dickey-Fuller. Null = unit root.
                              p < 0.05 → STATIONARY.
    method="kpss"           : Kwiatkowski-Phillips-Schmidt-Shin (regression="c").
                              Null = stationary. p < 0.05 → NON-STATIONARY.
    method="variance_break" : Levene's test for equal variance across two
                              halves (or around `split_at`).  Null = equal
                              variance.  p < 0.05 → variance-non-stationary.

    Why three methods:
      - ADF is the default for mean/unit-root non-stationarity.
      - KPSS catches mild trends that ADF misses (used for Level 3 drift).
      - Levene's catches variance regime shifts (used for Level 4): the
        regime_switch series is mean-zero on both sides, so ADF and KPSS
        see a stationary series even though the variance jumps from 0.5
        to 2.0 at t=split_at.  The variance shift IS the non-stationarity,
        and Levene's is the textbook test for it.
    """
    if method == "kpss":
        # regression="c" tests stationarity around a constant only.
        # "ct" would first detrend the series, so a trended series would
        # always report stationary — useless for catching the drift we
        # intentionally injected at Level 3.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat, p_value, _, _ = kpss(series, regression="c", nlags="auto")
        is_stationary = p_value >= 0.05
        label = "KPSS"
    elif method == "variance_break":
        cut = split_at if split_at is not None else len(series) // 2
        stat, p_value = levene(series[:cut], series[cut:], center="median")
        is_stationary = p_value >= 0.05
        label = "Levene"
    else:
        adf_stat, p_value, _, _, critical_values, _ = adfuller(series)
        is_stationary = p_value < 0.05
        label = "ADF"
    status = "STATIONARY" if is_stationary else "NON-STATIONARY"
    print(f"  {name:<40}  [{label}] p={p_value:.5f}  →  {status}")
    return {"p": p_value, "stationary": is_stationary, "method": label}


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — Generate, test, plot, save
# ══════════════════════════════════════════════════════════════════════════════

LEVELS = {
    "white_noise":    "Level 1 — White Noise",
    "ar1":            "Level 2 — AR(1) Stationary",
    "drift":          "Level 3 — Drift (Non-Stationary)",
    "regime_switch":  "Level 4 — Regime Switch",
}

print("=" * 55)
print("  STEP 1: Generate Synthetic Data")
print("=" * 55)
print()

datasets = {}
for key, name in LEVELS.items():
    X, y, base = generate(key)
    datasets[key] = {"X": X, "y": y, "base": base, "name": name}
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    print(f"  Generated: {name}")
    print(f"    Rows: {len(X)}   Features: {len(X.columns)}")
    print(f"    Target: {n_pos} UP ({n_pos/len(y)*100:.1f}%)  "
          f"{n_neg} DOWN ({n_neg/len(y)*100:.1f}%)")
    print()

print("Stationarity Tests:")
print("  ADF    (p < 0.05 → STATIONARY)      for white_noise, ar1")
print("  KPSS   (p < 0.05 → NON-STATIONARY)  for drift")
print("  Levene (p < 0.05 → NON-STATIONARY)  for regime_switch (variance break)")
print()
STATIONARITY_METHOD = {
    "white_noise":   "adf",
    "ar1":           "adf",
    "drift":         "kpss",
    "regime_switch": "variance_break",
}
adf_results = {}
for key, data in datasets.items():
    # Pass the known switch point for variance_break so the two halves
    # align with the actual regime boundary.
    split_at = 500 if key == "regime_switch" else None
    r = test_stationarity(data["base"], data["name"],
                          method=STATIONARITY_METHOD[key],
                          split_at=split_at)
    adf_results[key] = r

print()
print("Expected:")
print("  Level 1 (white_noise)   → STATIONARY")
print("  Level 2 (ar1)           → STATIONARY")
print("  Level 3 (drift)         → NON-STATIONARY")
print("  Level 4 (regime_switch) → NON-STATIONARY")
print()

all_ok = True
expectations = {
    "white_noise":   True,
    "ar1":           True,
    "drift":         False,
    "regime_switch": False,
}
for key, expected_stationary in expectations.items():
    actual = adf_results[key]["stationary"]
    ok     = (actual == expected_stationary)
    symbol = "PASS" if ok else "UNEXPECTED"
    print(f"  {symbol}  {datasets[key]['name']}")
    if not ok:
        all_ok = False

print()
if all_ok:
    print("  All stationarity results match expectations.")
else:
    print("  WARNING: Some results unexpected. Check data generator.")
print()

print("Generating plots...")

fig, axes = plt.subplots(4, 2, figsize=(14, 16))
COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

for idx, (key, data) in enumerate(datasets.items()):
    base = data["base"]
    name = data["name"]
    adf  = adf_results[key]
    col  = COLORS[idx]

    ax = axes[idx, 0]
    ax.plot(base, color=col, linewidth=0.8, alpha=0.9)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    if key == "regime_switch":
        ax.axvline(500, color="red", linewidth=1.5, linestyle="--",
                   alpha=0.8, label="Regime switch at t=500")
        ax.legend(fontsize=7, loc="upper right")
    status = "STATIONARY" if adf["stationary"] else "NON-STATIONARY"
    ax.set_title(f"{name}\np={adf['p']:.4f}  {status}", fontsize=9, pad=4)
    ax.set_xlabel("Time step", fontsize=8)
    ax.set_ylabel("Value",     fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)

    ax2 = axes[idx, 1]
    s  = pd.Series(base)
    rm = s.rolling(50).mean()
    rs = s.rolling(50).std()
    ax2.plot(rm, color=col, linewidth=1.2, label="Rolling mean (w=50)")
    ax2.fill_between(range(len(base)), rm - rs, rm + rs,
                     alpha=0.25, color=col, label="±1 std")
    ax2.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    if key == "regime_switch":
        ax2.axvline(500, color="red", linewidth=1.5, linestyle="--", alpha=0.7)
    ax2.set_title("Rolling statistics (flat = stationary)", fontsize=9, pad=4)
    ax2.set_xlabel("Time step", fontsize=8)
    ax2.legend(fontsize=7)
    ax2.tick_params(labelsize=7)
    ax2.grid(True, alpha=0.2)

fig.suptitle(
    "Synthetic Time Series — 4 Complexity Levels\n"
    "LEFT: Raw series   RIGHT: Rolling mean ± std\n"
    "Flat rolling statistics = stationary   |   Changing = non-stationary",
    fontsize=10, fontweight="bold"
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("plots/step1_data_inspection.png", dpi=150, bbox_inches="tight")
print("  Saved: plots/step1_data_inspection.png")

for key, data in datasets.items():
    path = f"data/{key}.pkl"
    with open(path, "wb") as f:
        pickle.dump({"X": data["X"], "y": data["y"], "base": data["base"]}, f)
    print(f"  Saved: {path}")

print()
print("=" * 55)
print("  STEP 1 COMPLETE")
print("=" * 55)
print()
print("What you just built:")
print("  4 synthetic datasets, each 1000 steps x 50 features")
print("  5 signal features (truly predictive)")
print("  45 noise features (pure random)")
print("  Stationarity tests confirm the data is correct")
print()
print("Next: run step2_baseline.py")