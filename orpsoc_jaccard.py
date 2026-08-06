"""
orpsoc_jaccard.py — Jaccard stability measured against its null
================================================================
Inter-fold Jaccard similarity of selected subsets is reported as a stability
metric, but a raw Jaccard value is not interpretable on its own: it is
mechanically driven by subset SIZE. Two subsets drawn completely at random
already overlap, and the more features they contain the more they overlap.

    E[|A n B|] = k_a * k_b / N
    E[|A u B|] = k_a + k_b - k_a * k_b / N
    E[J]      ~ (k_a k_b / N) / (k_a + k_b - k_a k_b / N)

For equal sizes that reduces to  E[J] ~ k / (2N - k), so with N=50:

    k =  5  ->  0.053        k = 15  ->  0.176
    k = 10  ->  0.111        k = 50  ->  1.000

Two consequences for how the figure must be read:

  * The all-features baseline sits at Jaccard = 1.0 because k = N makes the
    formula an identity. That is arithmetic, not stability.
  * A selector reporting J ~ 0.10-0.20 while holding ~10-15 features is at
    roughly 1.0-1.2x its own null -- i.e. its fold-to-fold overlap is
    statistically indistinguishable from choosing at random each fold. That is
    a stronger statement than "unstable", and it is invisible unless the null
    is drawn alongside.

This module provides the null, the ratio and excess over it, a Monte-Carlo
z-score, and a per-seed significance test.

Run as a script to profile an ablation JSON:
    python orpsoc_jaccard.py [results/step7_ablation.json]
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache

import numpy as np

__all__ = ["null_jaccard_closed", "null_jaccard_mc", "jaccard_profile",
           "profile_condition", "EPS"]

EPS = 1e-12


def _nanmean(values) -> float:
    """np.nanmean without the all-NaN RuntimeWarning.

    An all-NaN column is normal here, not an error: the baseline holds every
    feature, so its null has zero spread and the z-score is genuinely
    undefined rather than missing.
    """
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


# ══════════════════════════════════════════════════════════════════════════════
#  THE NULL
# ══════════════════════════════════════════════════════════════════════════════

def null_jaccard_closed(k_a: int, k_b: int, n_features: int) -> float:
    """
    Expected Jaccard of two INDEPENDENT uniformly random subsets, sizes k_a and
    k_b drawn from n_features.

    This is a ratio of expectations, E[|A n B|] / E[|A u B|], not the
    expectation of the ratio. It is cheap, exact in the large-N limit, and
    within ~0.005 of the Monte-Carlo mean for the sizes this project uses
    (verified in the __main__ block). Use null_jaccard_mc() when you need the
    spread as well as the centre, or when k is very small.
    """
    if k_a <= 0 or k_b <= 0:
        return 0.0
    n = float(n_features)
    inter = k_a * k_b / n
    union = k_a + k_b - inter
    return float(inter / union) if union > EPS else 0.0


@lru_cache(maxsize=4096)
def _mc_cached(k_a: int, k_b: int, n_features: int, trials: int,
               seed: int) -> tuple:
    rng = np.random.RandomState(seed)
    j = np.empty(trials)
    for t in range(trials):
        a = rng.choice(n_features, k_a, replace=False)
        b = rng.choice(n_features, k_b, replace=False)
        inter = np.intersect1d(a, b, assume_unique=True).size
        j[t] = inter / (k_a + k_b - inter)
    return float(j.mean()), float(j.std())


def null_jaccard_mc(k_a: int, k_b: int, n_features: int, trials: int = 4000,
                    seed: int = 0) -> tuple:
    """
    Monte-Carlo null: returns (mean, std) of the Jaccard between two
    independent random subsets of the given sizes.

    Results are cached on (k_a, k_b, n_features, trials, seed), which matters
    because a full ablation asks for the same few size-pairs thousands of
    times. Argument order is normalised so (5, 10) and (10, 5) share a cache
    entry -- Jaccard is symmetric.
    """
    if k_a <= 0 or k_b <= 0:
        return 0.0, 0.0
    lo, hi = (k_a, k_b) if k_a <= k_b else (k_b, k_a)
    return _mc_cached(int(lo), int(hi), int(n_features), int(trials), int(seed))


# ══════════════════════════════════════════════════════════════════════════════
#  PER-RUN PROFILE
# ══════════════════════════════════════════════════════════════════════════════

def jaccard_profile(selected_sets, n_features: int, mc: bool = True,
                    trials: int = 4000, seed: int = 0) -> dict:
    """
    Profile one walk-forward sequence of selected subsets (one seed, one
    condition).

    selected_sets : list of per-fold selections, each a list/set of feature
                    names or indices. Consecutive pairs are compared.
    n_features    : size of the candidate pool the selector chose FROM. This is
                    the denominator of the null and must be the full feature
                    count, not the selected count.

    Returned per fold-pair:
      observed  raw Jaccard
      null      expected Jaccard for independent random subsets of the same
                two sizes -- the size-matched baseline
      ratio     observed / null. 1.0 = indistinguishable from random. Undefined
                (NaN) when null is ~0, which happens only for empty selections.
      excess    observed - null. Bounded and safe when null is small; prefer it
                to `ratio` whenever k is tiny.
      z         (observed - null_mean) / null_sd from the Monte-Carlo null.
                None when mc=False. This is the quantity to test, because it is
                already standardised for subset size.
    """
    out = {"observed": [], "null": [], "ratio": [], "excess": [], "z": [],
           "k_a": [], "k_b": []}
    for a_raw, b_raw in zip(selected_sets[:-1], selected_sets[1:]):
        A, B = set(a_raw), set(b_raw)
        union = A | B
        obs = len(A & B) / len(union) if union else float("nan")
        ka, kb = len(A), len(B)
        nul = null_jaccard_closed(ka, kb, n_features)
        out["observed"].append(obs)
        out["null"].append(nul)
        out["k_a"].append(ka)
        out["k_b"].append(kb)
        out["ratio"].append(obs / nul if nul > 1e-6 else float("nan"))
        out["excess"].append(obs - nul)
        if mc:
            mu, sd = null_jaccard_mc(ka, kb, n_features, trials, seed)
            out["z"].append((obs - mu) / sd if sd > 1e-9 else float("nan"))
        else:
            out["z"].append(float("nan"))
    return out


def profile_condition(seed_records, n_features: int, key: str = "fold_selected",
                      mc: bool = True, trials: int = 4000) -> dict:
    """
    Aggregate across seeds for one condition.

    seed_records : list of per-seed dicts, each holding `key` -> list of
                   per-fold selections (the layout step7 writes).

    Returns per-seed means plus a one-sided test of whether the excess over the
    null is greater than zero. The test is across SEEDS, which are independent;
    fold-pairs within a seed are not, because consecutive pairs share a fold.
    """
    per_seed = {"observed": [], "null": [], "ratio": [], "excess": [],
                "z": [], "k": []}
    for i, rec in enumerate(seed_records):
        sel = rec.get(key) or []
        if len(sel) < 2:
            continue
        p = jaccard_profile(sel, n_features, mc=mc, trials=trials, seed=i)
        for f in ("observed", "null", "ratio", "excess", "z"):
            per_seed[f].append(_nanmean(p[f]))
        per_seed["k"].append(float(np.mean([len(s) for s in sel])))

    res = {f: _nanmean(v) for f, v in per_seed.items()}
    res["n_seeds"] = len(per_seed["excess"])
    ex = np.asarray(per_seed["excess"], dtype=float)
    ex = ex[np.isfinite(ex)]
    res["excess_sd"] = float(ex.std()) if len(ex) else float("nan")
    res["per_seed_excess"] = per_seed["excess"]

    # One-sided: is the selector's overlap above its size-matched null?
    res["p_excess_gt_0"] = float("nan")
    vals = np.asarray(per_seed["excess"], dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) >= 3 and not np.allclose(vals, vals[0]):
        try:
            from scipy import stats
            res["p_excess_gt_0"] = float(stats.wilcoxon(
                vals, alternative="greater").pvalue)
        except Exception:
            pass
    elif len(vals) >= 1 and np.allclose(vals, 0.0, atol=1e-9):
        res["p_excess_gt_0"] = 1.0      # exactly at the null (e.g. baseline)
    return res


# ══════════════════════════════════════════════════════════════════════════════
#  SCRIPT
# ══════════════════════════════════════════════════════════════════════════════

def _report(path: str) -> None:
    with open(path) as f:
        doc = json.load(f)
    cfg = doc.get("config", {})
    print("=" * 88)
    print(f"  JACCARD vs NULL   —   {path}")
    print(f"  config: {cfg}")
    if cfg.get("fast_mode"):
        print("  ** FAST_MODE artefact: these numbers are superseded by the "
              "full run **")
    print("=" * 88)

    for level, ldata in doc.get("full_results", {}).items():
        conds = ldata.get("conditions", {})
        # Candidate-pool size: the baseline "selects" every feature, so its
        # subset length is the pool size. Falls back to 50 (the project default)
        # if the baseline condition is absent.
        n_feat = 50
        if "baseline" in conds and conds["baseline"]:
            sel = conds["baseline"][0].get("fold_selected") or []
            if sel:
                n_feat = len(sel[0])
        print(f"\n  {ldata.get('level_name', level)}   (N = {n_feat} features)")
        print(f"    {'condition':<20}{'mean k':>8}{'null J':>9}{'obs J':>8}"
              f"{'ratio':>8}{'excess':>9}{'z':>8}{'p':>9}")
        for cond, recs in conds.items():
            r = profile_condition(recs, n_feat)
            p = r["p_excess_gt_0"]
            pstr = "  n/a" if not np.isfinite(p) else f"{p:.4f}"
            zs = "     —" if not np.isfinite(r["z"]) else f"{r['z']:6.1f}"
            print(f"    {cond:<20}{r['k']:>8.1f}{r['null']:>9.3f}"
                  f"{r['observed']:>8.3f}{r['ratio']:>8.2f}"
                  f"{r['excess']:>+9.3f}{zs:>8}{pstr:>9}")
        print("      ratio 1.0 = indistinguishable from selecting at random each fold")


if __name__ == "__main__":
    # Self-check: the closed form should track the Monte-Carlo null.
    print("closed form vs Monte Carlo (N = 50)")
    print(f"  {'k':>4}{'closed':>10}{'MC mean':>10}{'MC sd':>8}{'diff':>8}")
    for k in (5, 8, 10, 12, 15, 20, 30, 50):
        c = null_jaccard_closed(k, k, 50)
        m, s = null_jaccard_mc(k, k, 50, trials=20000)
        print(f"  {k:>4}{c:>10.4f}{m:>10.4f}{s:>8.4f}{m - c:>+8.4f}")
    print("  (closed form runs slightly low for small k; use `excess`/`z` there)\n")

    _report(sys.argv[1] if len(sys.argv) > 1 else "results/step7_ablation.json")
