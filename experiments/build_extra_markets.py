"""
Build two additional real datasets: BONDS and COMMODITIES/GOLD.

Purpose: the "frozen filter beats a refitting filter on real data" result now
holds on two equity-derived datasets (sector ETFs, Fama-French factors). Both
are US equity risk. A genuine replication needs a DIFFERENT ASSET CLASS, where
volatility dynamics and feature correlation structure differ.

Methodology is deliberately identical to the sector-ETF build -- same 6
per-instrument features, same 4 cross-sectional features, same volatility-regime
target, same warm-up and trim. Only the instrument universe changes, so any
difference in result is attributable to the asset class rather than to the
pipeline.

Run:  python experiments/build_extra_markets.py
Writes data/bonds.pkl and data/commodities.pkl
"""
import os
import sys
import pickle

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Long-history, liquid, non-overlapping exposures within each class.
UNIVERSES = {
    # Treasuries across the curve + credit. TLT/IEF/SHY/LQD/AGG/TIP all list
    # 2002-2003, so the usable history is comparable to the sector ETFs.
    "bonds": (["TLT", "IEF", "SHY", "LQD", "AGG", "TIP"], "2003-01-01"),
    # Metals, energy, broad commodity, agriculture. GLD lists 2004, the rest
    # 2006-2007, so this one starts later and is the shorter series.
    "commodities": (["GLD", "SLV", "USO", "DBC", "DBA", "UNG"], "2007-01-01"),
}


def download(tickers, start, cache):
    if os.path.exists(cache):
        print(f"  [cache] {cache}")
        return pd.read_pickle(cache)
    import yfinance as yf
    print(f"  downloading {tickers} from {start} ...", flush=True)
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False,
                      threads=True)
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0)
        prices = raw["Close"].copy() if "Close" in lvl0 else raw.xs("Close", axis=1, level=1)
    else:
        prices = raw[["Close"]].copy()
    prices = prices[[t for t in tickers if t in prices.columns]]
    prices = prices.dropna(how="all").ffill().dropna()
    prices.to_pickle(cache)
    return prices


if __name__ == "__main__":
    # Reuse the EXACT sector-ETF builder so the methodology is identical.
    src = open("step9_real_data.py").read().split("#  ABLATION LOOP")[0]
    ns = {}
    exec(compile(src, "s9", "exec"), ns)
    build = ns["build_sector_etf_dataset"]

    for name, (tickers, start) in UNIVERSES.items():
        cache = f"data/raw_{name}_prices.pkl"
        try:
            prices = download(tickers, start, cache)
        except ModuleNotFoundError:
            # A missing package is a broken environment, not an unavailable
            # dataset. Swallowing it as a SKIP hid `No module named pyarrow`
            # behind a misleading "check network" message four stages later.
            raise
        except Exception as e:
            print(f"  SKIP {name}: {type(e).__name__}: {e}")
            continue
        print(f"  {name:<12} prices {prices.shape}  {prices.index.min().date()} "
              f"-> {prices.index.max().date()}  tickers={list(prices.columns)}")
        X, y, base, dates = build(prices)
        with open(f"data/{name}.pkl", "wb") as f:
            pickle.dump({"X": X, "y": y, "base": base}, f)
        print(f"  {name:<12} X={X.shape}  y-mean={y.mean():.3f}  -> data/{name}.pkl\n")
