"""
STEP 2 — Walk-Forward Baseline
=================================
Before running OrPSOC, establish what a naive model achieves.

WHY A BASELINE FIRST?
──────────────────────
Without a baseline you cannot claim OrPSOC adds value.
If OrPSOC gives AUC = 0.72, is that good? You need to know
what AUC all-features achieves first.

WHAT YOU ARE MEASURING
────────────────────────
For each of the 4 levels:
  1. Run walk-forward CV using ALL 50 features
  2. Plot AUC per fold (chronologically)
  3. Observe whether AUC drops in later folds

The fold-by-fold AUC plot IS the scientific finding about
non-stationarity — AUC stays stable on Levels 1-2, drops on
Levels 3-4. That degradation is the problem OrPSOC must fix.

Run with:
    python step2_baseline.py
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")
import json
import os
os.makedirs("plots",   exist_ok=True)
os.makedirs("results", exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  WALK-FORWARD CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def walk_forward_cv(
    X:            pd.DataFrame,
    y:            pd.Series,
    feature_cols: list  = None,
    n_splits:     int   = 8,
    gap:          int   = 5,
    min_train:    int   = 150,
) -> dict:
    """
    Walk-forward cross-validation for time series.

    FOLD STRUCTURE (n=1000, min_train=150, gap=5, n_splits=8):
      Fold 1: train [0..149]   gap [150..154]   test [155..259]
      Fold 2: train [0..259]   gap [260..264]   test [265..369]
      ...

    Gap prevents rolling-window feature leakage across the boundary.
    """
    if feature_cols is None:
        feature_cols = list(X.columns)

    n         = len(X)
    fold_size = (n - min_train - gap) // n_splits

    fold_aucs = []
    fold_ends = []

    for fold in range(n_splits):
        train_end  = min_train + fold * fold_size
        test_start = train_end + gap
        test_end   = test_start + fold_size

        if test_end > n:
            break

        X_tr = X[feature_cols].iloc[:train_end]
        y_tr = y.iloc[:train_end]
        X_te = X[feature_cols].iloc[test_start:test_end]
        y_te = y.iloc[test_start:test_end]

        if len(y_te.unique()) < 2:
            continue

        pipe = Pipeline([
            ("imp",    SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model",  XGBClassifier(
                n_estimators=80, max_depth=4,
                learning_rate=0.1, verbosity=0, random_state=42
            ))
        ])
        pipe.fit(X_tr, y_tr)
        prob = pipe.predict_proba(X_te)[:, 1]
        auc  = roc_auc_score(y_te, prob)

        fold_aucs.append(float(auc))
        fold_ends.append(train_end)

    mean_auc = float(np.mean(fold_aucs)) if fold_aucs else 0.0
    std_auc  = float(np.std(fold_aucs))  if fold_aucs else 0.0

    return {
        "fold_aucs": fold_aucs,
        "fold_ends": fold_ends,
        "mean_auc":  mean_auc,
        "std_auc":   std_auc,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — Run baseline on all four levels
# ══════════════════════════════════════════════════════════════════════════════

LEVELS = {
    "white_noise":   "Level 1 — White Noise",
    "ar1":           "Level 2 — AR(1) Stationary",
    "drift":         "Level 3 — Drift (Non-Stationary)",
    "regime_switch": "Level 4 — Regime Switch",
}

print("=" * 55)
print("  STEP 2: Walk-Forward Baseline (All Features)")
print("=" * 55)
print()
print("Using all 50 features. No OrPSOC yet.")
print("Purpose: establish baseline AUC and observe degradation.")
print()

all_results = {}

for key, name in LEVELS.items():
    with open(f"data/{key}.pkl", "rb") as f:
        data = pickle.load(f)
    X, y = data["X"], data["y"]

    print(f"Running: {name}")
    r = walk_forward_cv(X, y, n_splits=8)
    all_results[key] = {"name": name, "r": r}

    aucs  = r["fold_aucs"]
    early = np.mean(aucs[:4]) if len(aucs) >= 4 else np.mean(aucs)
    late  = np.mean(aucs[4:]) if len(aucs) >= 5 else (aucs[-1] if aucs else 0.0)
    drop  = early - late

    print(f"  Overall mean AUC : {r['mean_auc']:.4f}")
    print(f"  Std  AUC         : {r['std_auc']:.4f}")
    print(f"  Early folds avg  : {early:.4f}")
    print(f"  Late  folds avg  : {late:.4f}")

    if drop > 0.03:
        print(f"  AUC drop         : {drop:.4f}  <<< DEGRADATION DETECTED")
    else:
        print(f"  AUC drop         : {drop:.4f}  (stable)")
    print()

print("=" * 55)
print("  INTERPRETATION")
print("=" * 55)
print()
print("  Level 1 (White Noise):   Mean AUC ≈ 0.50  (random guessing)")
print("  Level 2 (AR1 Stat.):     Mean AUC > 0.60  (stable signal)")
print("  Level 3 (Drift):         AUC drops in later folds")
print("  Level 4 (Regime Switch): AUC collapses after fold 4-5")
print()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes  = axes.flatten()
COLS  = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

for idx, (key, data) in enumerate(all_results.items()):
    ax   = axes[idx]
    r    = data["r"]
    aucs = r["fold_aucs"]
    folds = list(range(1, len(aucs) + 1))
    col  = COLS[idx]

    ax.plot(folds, aucs, "o-", color=col, linewidth=2,
            markersize=8, label="Fold AUC", zorder=3)
    ax.axhline(0.5, color="gray", linewidth=1, linestyle="--",
               alpha=0.6, label="Random baseline (0.5)")
    ax.axhline(r["mean_auc"], color=col, linewidth=1.5,
               linestyle=":", alpha=0.8,
               label=f"Mean = {r['mean_auc']:.4f}")

    mid = len(folds) // 2
    if mid > 0:
        ax.axvspan(0.5, mid + 0.5, alpha=0.06, color="green")
        ax.axvspan(mid + 0.5, len(folds) + 0.5, alpha=0.06, color="red")
        ax.text(0.5 + 0.1, 0.35, "Early folds", fontsize=7,
                color="green", alpha=0.8)
        ax.text(mid + 0.5 + 0.1, 0.35, "Late folds", fontsize=7,
                color="red", alpha=0.8)

    if key == "regime_switch":
        ax.axvline(mid + 0.5, color="red", linewidth=2,
                   linestyle="-.", alpha=0.9, label="Regime switch ≈ here")

    ax.set_title(data["name"], fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel("Walk-Forward Fold (chronological →)", fontsize=9)
    ax.set_ylabel("AUC-ROC", fontsize=9)
    ax.set_ylim([0.30, 1.05])
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=8)

fig.suptitle(
    "Baseline: Walk-Forward AUC — All 50 Features\n"
    "Key question: Does AUC drop in later folds for non-stationary data?",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()
plt.savefig("plots/step2_baseline_auc.png", dpi=150, bbox_inches="tight")
print("Saved: plots/step2_baseline_auc.png")
print()

save_data = {}
for key, data in all_results.items():
    save_data[key] = {
        "name":      data["name"],
        "mean_auc":  data["r"]["mean_auc"],
        "std_auc":   data["r"]["std_auc"],
        "fold_aucs": data["r"]["fold_aucs"],
    }
with open("results/step2_baseline.json", "w") as f:
    json.dump(save_data, f, indent=2)
print("Saved: results/step2_baseline.json")
print()
print("=" * 55)
print("  STEP 2 COMPLETE")
print("=" * 55)
print()
print("RECORD THESE NUMBERS:")
for key, data in all_results.items():
    r = data["r"]
    print(f"  {data['name']:<40}  mean={r['mean_auc']:.4f}  std={r['std_auc']:.4f}")
print()
print("Next: run step3_orpsoc_stationary.py")