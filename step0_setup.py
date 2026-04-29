"""
STEP 0 — Environment Setup
============================
Run this FIRST before anything else.

It checks every package you need, installs what is missing,
and confirms your environment is ready.

Run with:
    python step0_setup.py
"""

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
