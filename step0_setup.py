"""
STEP 0 — Environment Setup
============================
Run this FIRST before anything else.

It checks every package you need, installs what is missing,
and confirms your environment is ready.

Run with:
    python step0_setup.py
"""

# ── import guard ─────────────────────────────────────────────────────────────
# This file is a SCRIPT, not a module. It executes its whole pipeline at module
# level, so `import step0_setup` runs the entire thing as a side effect --
# for step7_ablation that is a 4-hour ablation triggered by an innocent-looking
# import. Fail loudly instead.
#
# `globals().get("__name__", ...)` rather than a bare `__name__`: helpers are
# reused by exec'ing the section above a marker into a fresh namespace (see
# experiments/apsoll_sweep.py), and that namespace has no __name__ at all.
if globals().get("__name__", "__main__") != "__main__":
    raise ImportError(
        "step0_setup.py is a script, not an importable module -- importing it would "
        "execute the full pipeline. To reuse a helper, exec the section above "
        "the main loop into a fresh namespace (see experiments/apsoll_sweep.py)."
    )
# ─────────────────────────────────────────────────────────────────────────────


import subprocess
import sys

# ── Every package this project needs ─────────────────────────────────────────
REQUIRED = {
    "numpy":        "numpy",
    "pandas":       "pandas",
    "sklearn":      "scikit-learn",
    "xgboost":      "xgboost",
    "statsmodels":  "statsmodels",
    "matplotlib":   "matplotlib",
}

print("=" * 55)
print("  OrPSOC Research — Environment Setup")
print("=" * 55)
print()

missing = []
for import_name, pip_name in REQUIRED.items():
    try:
        __import__(import_name)
        print(f"  OK   {import_name}")
    except ImportError:
        print(f"  MISSING  {import_name}  →  will install {pip_name}")
        missing.append(pip_name)

if missing:
    print()
    print(f"Installing {len(missing)} missing package(s)...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet", *missing
    ])
    print("Installation done.")
else:
    print()
    print("All packages present.")

# ── Final verification ────────────────────────────────────────────────────────
print()
print("Verifying imports...")
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from statsmodels.tsa.stattools import adfuller
import matplotlib
print("  All imports successful.")
print()
print("Python version :", sys.version.split()[0])
print("NumPy  version :", np.__version__)
print("Pandas version :", pd.__version__)
print()
print("=" * 55)
print("  Setup complete. Run step1_generate_data.py next.")
print("=" * 55)
