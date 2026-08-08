"""
TIER C — the real-data analyses that needed the full re-run (work order C2-C8).

C1 was satisfied by the 2026-08-07 pipeline run (step8 regenerated Table 2 and
Figure 1 from a single code version). C6 is a recorded decision, not an
analysis: real-market runs use 20 seeds, synthetic 30 (see step9_real_data.py).

This script covers:
  C2  five-condition table per real dataset, per-seed
  C3  importance-reinit delta + paired Wilcoxon on real data
  C4  detector trigger folds vs the DOCUMENTED break dates
  C5  per-seed variance (std AND range) -- the argument for paired tests
  C7/C8 per-fold recovery and Jaccard traces including the FIFTH condition

Per-seed values come from the step9 CHECKPOINTS. step9's JSON stored only
aggregates until the fix added seed_aucs; the checkpoints always held the
per-seed payloads, so no re-run was required to recover them.

Run:  python experiments/tier_c.py
Writes results/tier_c.json and plots/tier_c_*.png
"""
import os
import sys
import json
import glob
import pickle

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

COND = ["baseline", "standard_orpsoc", "apsoll", "full_hybrid",
        "full_hybrid_noimp"]
LABEL = {"baseline": "Baseline", "standard_orpsoc": "OrPSOC",
         "apsoll": "+APSOLL", "full_hybrid": "Full Hybrid",
         "full_hybrid_noimp": "FH no-imp"}
COLOR = {"baseline": "#78909C", "standard_orpsoc": "#42A5F5",
         "apsoll": "#66BB6A", "full_hybrid": "#EF5350",
         "full_hybrid_noimp": "#AB47BC"}
BREAKS = {"sector_etf":  ["2001-09-17", "2008-09-15", "2020-02-20"],
          "fama_french": ["2001-09-17", "2008-09-15", "2020-02-20"]}


def load_checkpoints():
    """dataset -> list of per-seed payloads, from the step9 checkpoint store."""
    files = sorted(glob.glob("results/checkpoints/step9/*/*.pkl"))
    if not files:
        files = sorted(glob.glob("results_ec2/*/results/checkpoints/step9/*/*.pkl"))
    out = {}
    for f in files:
        base = os.path.basename(f)
        if "_seed" not in base:
            continue
        ds = base.split("_seed")[0]
        try:
            out.setdefault(ds, []).append(pickle.load(open(f, "rb"))["payload"])
        except Exception as e:
            print(f"  [warn] unreadable checkpoint {base}: {e}")
    return out


def seed_means(seeds, c):
    return np.array([np.mean(s[c]["fold_aucs"]) for s in seeds
                     if c in s and s[c].get("fold_aucs")], float)


def fold_matrix(seeds, c):
    """(n_seeds, n_folds) of per-fold AUC, truncated to the shortest seed."""
    rows = [s[c]["fold_aucs"] for s in seeds if c in s and s[c].get("fold_aucs")]
    if not rows:
        return np.zeros((0, 0))
    m = min(len(r) for r in rows)
    return np.array([r[:m] for r in rows], float)


def c2_c5(ds, seeds, s9, out):
    """C2 five-condition table + C5 per-seed spread."""
    print(f"\n{'='*84}")
    print(f"  {ds}   (C2 five-condition table, C5 per-seed spread)   n_seeds={len(seeds)}")
    print(f"{'='*84}")
    base = seed_means(seeds, "baseline")
    print(f"  {'condition':<16}{'mean AUC':>10}{'std':>9}{'min':>9}{'max':>9}"
          f"{'range':>9}{'vs base':>10}{'p':>9}")
    print("  " + "-" * 81)
    rec = {}
    for c in COND:
        v = seed_means(seeds, c)
        if len(v) == 0:
            continue
        if c == "baseline":
            ptxt, dtxt = f"{'-':>9}", f"{'(base)':>10}"
            p = None
        else:
            try:
                p = float(stats.wilcoxon(v, base).pvalue)
                ptxt = f"{p:>9.4f}"
            except Exception:
                p, ptxt = None, f"{'n/a':>9}"
            dtxt = f"{v.mean()-base.mean():>+10.4f}"
        print(f"  {LABEL[c]:<16}{v.mean():>10.4f}{v.std():>9.4f}{v.min():>9.4f}"
              f"{v.max():>9.4f}{v.max()-v.min():>9.4f}{dtxt}{ptxt}")
        rec[c] = {"mean": float(v.mean()), "std": float(v.std()),
                  "min": float(v.min()), "max": float(v.max()),
                  "range": float(v.max() - v.min()),
                  "delta_vs_baseline": float(v.mean() - base.mean()),
                  "wilcoxon_p": p, "seed_aucs": [float(x) for x in v]}
    # C5 commentary
    spread = max(rec[c]["range"] for c in rec if c != "baseline")
    print(f"\n  C5: widest per-seed RANGE among selection arms = {spread:.4f} AUC;"
          f"  baseline range = {rec['baseline']['range']:.4f}")
    if rec["baseline"]["std"] < 1e-9:
        print("      Baseline is deterministic (no PSO), so seed-to-seed spread is")
        print("      entirely a property of the SELECTOR. Comparing means would hide")
        print("      it -- which is exactly why paired per-seed tests are the correct")
        print("      lens, not mean-vs-mean.")
    out.setdefault(ds, {})["C2_C5"] = rec


def c3(ds, seeds, out):
    """C3 importance-reinit delta, paired per seed."""
    a = seed_means(seeds, "full_hybrid")
    b = seed_means(seeds, "full_hybrid_noimp")
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if n == 0:
        return
    try:
        p_two = float(stats.wilcoxon(a, b).pvalue)
        p_gt = float(stats.wilcoxon(a, b, alternative="greater").pvalue)
    except Exception:
        p_two = p_gt = float("nan")
    print(f"\n  C3 IMPORTANCE-REINIT (supervisor suggestion #2) on {ds}")
    print(f"    full_hybrid        {a.mean():.4f}")
    print(f"    full_hybrid_noimp  {b.mean():.4f}")
    print(f"    delta              {a.mean()-b.mean():+.4f}   "
          f"p(two-tailed)={p_two:.4f}  p(greater)={p_gt:.4f}   n={n} seeds")
    wins = int(np.sum(a > b))
    print(f"    seeds where imp-reinit helps: {wins}/{n}")
    out.setdefault(ds, {})["C3"] = {
        "full_hybrid": float(a.mean()), "full_hybrid_noimp": float(b.mean()),
        "delta": float(a.mean() - b.mean()), "p_two_tailed": p_two,
        "p_greater": p_gt, "n_seeds": n, "seeds_helped": wins}


def c4(ds, s9, out):
    """C4 detector trigger folds vs documented break dates."""
    blk = s9["datasets"][ds]
    brk = blk.get("break_folds") or []
    fire = blk.get("trigger_fire_rate") or []
    raw = blk.get("trigger_raw_fire") or []
    ptr = blk.get("trigger_p_trans") or []
    wu = blk.get("trigger_is_warmup") or []
    n = len(fire)
    print(f"\n  C4 DETECTOR vs DOCUMENTED BREAKS on {ds}")
    print(f"    documented break dates : {BREAKS.get(ds)}")
    print(f"    break_folds (test window contains a break) : {brk}")
    print(f"    earliest CAUSALLY POSSIBLE response        : "
          f"{[b+1 for b in brk]}  (break must enter TRAINING first)")
    print(f"    {'fold':<10}" + "".join(f"f{i}".rjust(8) for i in range(n)))
    print(f"    {'p_trans':<10}" + "".join(f"{v:>8.3f}" for v in ptr))
    print(f"    {'raw fire':<10}" + "".join(f"{int(bool(v)):>8}" for v in raw))
    print(f"    {'warmup':<10}" + "".join(f"{int(bool(v)):>8}" for v in wu))
    print(f"    {'TRIGGERED':<10}" + "".join(f"{v:>8.2f}" for v in fire))
    fired = [i for i, v in enumerate(fire) if v > 0]
    expected = sorted({b + 1 for b in brk} | set(brk))
    hit = [f for f in fired if f in expected]
    print(f"    fired at folds {fired};  break-adjacent set {expected}")
    print(f"    -> {len(hit)}/{len(fired) if fired else 0} fires are break-adjacent"
          if fired else "    -> detector NEVER fired on this dataset")
    out.setdefault(ds, {})["C4"] = {
        "break_dates": BREAKS.get(ds), "break_folds": brk,
        "fired_folds": fired, "break_adjacent": expected,
        "fires_break_adjacent": hit, "p_trans": ptr,
        "raw_fire": [bool(v) for v in raw]}


def c7_c8(ds, seeds, s9, out):
    """C7 per-fold recovery + C8 Jaccard, BOTH including the fifth condition."""
    brk = s9["datasets"][ds].get("break_folds") or []
    fire = s9["datasets"][ds].get("trigger_fire_rate") or []
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))

    ax = axes[0]
    for c in COND:                                   # C7: all FIVE conditions
        M = fold_matrix(seeds, c)
        if M.size == 0:
            continue
        m, s = M.mean(0), M.std(0)
        x = np.arange(1, len(m) + 1)
        ax.plot(x, m, "-o", color=COLOR[c], lw=2, ms=5, label=LABEL[c])
        ax.fill_between(x, m - s, m + s, color=COLOR[c], alpha=0.10)
    for b in brk:
        ax.axvline(b + 1, color="red", ls="--", lw=1.6, alpha=0.8)
    for i, v in enumerate(fire):
        if v > 0:
            ax.axvline(i + 1, color="orange", ls=":", lw=2.0, alpha=0.9)
    ax.set_title(f"C7 — per-fold AUC, all five conditions ({ds})\n"
                 "red dashed = documented break fold, orange dotted = detector fired",
                 fontsize=9)
    ax.set_xlabel("fold"); ax.set_ylabel("test AUC")
    ax.legend(fontsize=7); ax.grid(alpha=0.25)

    ax = axes[1]
    for c in COND:                                   # C8: all FIVE conditions
        rows = [s[c]["jaccard"]["per_fold_jaccard"] for s in seeds
                if c in s and isinstance(s[c].get("jaccard"), dict)
                and s[c]["jaccard"].get("per_fold_jaccard")]
        if not rows:
            continue
        m0 = min(len(r) for r in rows)
        J = np.array([r[:m0] for r in rows], float)
        m, s_ = J.mean(0), J.std(0)
        x = np.arange(1, len(m) + 1)
        ax.plot(x, m, "-o", color=COLOR[c], lw=2, ms=5, label=LABEL[c])
        ax.fill_between(x, m - s_, m + s_, color=COLOR[c], alpha=0.10)
    for b in brk:
        ax.axvline(b + 0.5, color="red", ls="--", lw=1.6, alpha=0.8)
    ax.set_title(f"C8 — Jaccard between consecutive folds, all five conditions ({ds})",
                 fontsize=9)
    ax.set_xlabel("fold pair (i, i+1)"); ax.set_ylabel("Jaccard")
    ax.legend(fontsize=7); ax.grid(alpha=0.25)

    fig.tight_layout()
    p = f"plots/tier_c_{ds}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  C7/C8 saved {p}")
    out.setdefault(ds, {})["C7_C8_plot"] = p


def main():
    s9 = json.load(open("results/step9_real_data.json"))
    ck = load_checkpoints()
    os.makedirs("plots", exist_ok=True)
    print("#" * 84)
    print("#  TIER C — real-data analyses (work order C2-C8)")
    print(f"#  per-seed source: step9 checkpoints ({sum(len(v) for v in ck.values())} units)")
    print(f"#  config: {s9['config']['n_seeds']} seeds, {s9['config']['n_splits']} folds")
    print("#" * 84)
    out = {"config": s9["config"], "n_seeds": s9["config"]["n_seeds"]}
    for ds in s9["datasets"]:
        seeds = ck.get(ds, [])
        if not seeds:
            print(f"\n  [skip] {ds}: no checkpoints found")
            continue
        c2_c5(ds, seeds, s9, out)
        c3(ds, seeds, out)
        c4(ds, s9, out)
        c7_c8(ds, seeds, s9, out)
    with open("results/tier_c.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved: results/tier_c.json")


if __name__ == "__main__":
    main()
