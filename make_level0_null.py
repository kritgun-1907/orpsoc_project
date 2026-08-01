"""
make_level0_null.py — Level 0: a TRUE null benchmark
=====================================================
Writes data/null.pkl in the same {"X", "y", "base"} format as step1's four
levels, so it drops straight into step7_ablation.py's LEVELS dict.

WHY THIS LEVEL DOES NOT EXIST YET
──────────────────────────────────
Level 1 is called "white noise", but that refers to the TEMPORAL process of the
latent `base` (i.i.d. across time). The labels are still y = 1{base > 0} and
every signal feature is base + 0.3·noise, so |corr(signal_i, y)| ≈ 0.76 and the
all-features baseline scores AUC ≈ 0.993. L1 is a stationarity control, not a
null. Nothing in the current suite asks the question a null asks:

    When there is NOTHING to find, does the selector correctly find nothing,
    or does it manufacture a subset and report confidence in it?

CONSTRUCTION
────────────
    base_y ~ N(0,1) i.i.d.          latent used ONLY to build the label
    y      = 1{base_y > 0}
    X      ~ N(0,1) i.i.d.,          50 columns, ALL independent of base_y

The first five columns keep the names `signal_0..signal_4` purely so the
existing harness (recall, r1/r2 hit counts, signal_r1/signal_r2 splits) runs
without modification. ON THIS LEVEL THEY ARE DECOYS — they carry no signal.
The `fold_recall` metric therefore becomes a FALSE-DISCOVERY RATE:

    E[recall | k features chosen at random from 50] = k / 50

so a selector picking k=3 should score ≈0.06. Anything materially above that is
the selector latching onto validation-split noise.

WHAT TO EXPECT (predictions, written down BEFORE running — see guardrail G4)
────────────────────────────────────────────────────────────────────────────
  1. Baseline test AUC ≈ 0.50. If it is not, the generator is broken.
  2. Selection-variant test AUC ≤ 0.50. PSO maximises fitness on the inner
     validation split; with no real signal it can only fit that split's noise,
     which does not transfer. Below-chance is the expected outcome, not a bug.
  3. Subset size collapses to min_f = 3. This is the sharpest test of the
     objective-function analysis: fitness = θ·AUC + (1-θ)·(1 - k/N) with AUC
     flat at ~0.5 for every subset leaves ONLY the compactness term, whose
     optimum is the smallest legal subset. If k does not collapse to 3, the
     compactness term is not driving subset size and that analysis is wrong.

Run with:
    python make_level0_null.py
"""

import os
import pickle

import numpy as np
import pandas as pd

N_STEPS = 1000
N_FEATURES = 50
N_SIGNAL = 5          # decoy names only, kept for harness compatibility
SEED = 42

os.makedirs("data", exist_ok=True)


def generate_null(n_steps=N_STEPS, n_features=N_FEATURES,
                  n_signal=N_SIGNAL, seed=SEED):
    """Return (X, y, base) with y provably independent of every column of X."""
    rng = np.random.RandomState(seed)

    # Label latent. Drawn FIRST and never reused, so no feature can encode it.
    base = rng.randn(n_steps)
    y = pd.Series((base > 0).astype(int), name="target")

    # Features. Drawn from an independent stream; never a function of `base`.
    cols = {}
    for i in range(n_signal):
        cols[f"signal_{i}"] = rng.randn(n_steps)       # DECOY — no signal
    for i in range(n_features - n_signal):
        cols[f"noise_{i}"] = rng.randn(n_steps)
    X = pd.DataFrame(cols)

    return X, y, base


if __name__ == "__main__":
    X, y, base = generate_null()

    print("=" * 68)
    print("  LEVEL 0 — True Null Benchmark")
    print("=" * 68)
    print(f"  X shape      : {X.shape}")
    print(f"  y balance    : {y.mean():.4f}   (should be ~0.50)")
    print(f"  NaNs         : {int(X.isna().sum().sum())}")

    cors = np.array([abs(np.corrcoef(X[c], y)[0, 1]) for c in X.columns])
    print(f"  |corr(x_j,y)|: max={cors.max():.4f}  mean={cors.mean():.4f}")
    print(f"                 (max is order 1/sqrt(n) = {1/np.sqrt(len(X)):.4f}; "
          f"pure sampling noise)")

    # Contrast with L1, which is often mistaken for a null.
    try:
        with open("data/white_noise.pkl", "rb") as f:
            wn = pickle.load(f)
        wc = np.array([abs(np.corrcoef(wn["X"][c], wn["y"])[0, 1])
                       for c in wn["X"].columns])
        print(f"\n  For contrast, Level 1 'white_noise': "
              f"max |corr| = {wc.max():.4f}  (NOT a null)")
    except FileNotFoundError:
        pass

    with open("data/null.pkl", "wb") as f:
        pickle.dump({"X": X, "y": y, "base": base}, f)
    print("\n  Saved: data/null.pkl")
    print("  Add to step7_ablation.py LEVELS as:  \"null\": \"Level 0 — True Null\"")
