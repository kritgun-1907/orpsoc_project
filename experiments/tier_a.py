"""
TIER A — analysis only, zero re-runs (work order §3, A1-A4, A6).

Reads results/step7_ablation_v2.json (synthetic, 30 seeds) and, for A6, the
step9 real-data checkpoints. Produces results/tier_a.json plus a printed report.

═══════════════════════════════════════════════════════════════════════════════
PRE-REGISTERED DEFINITIONS  —  fixed BEFORE any result was inspected
═══════════════════════════════════════════════════════════════════════════════
The work order requires these be written down first (A1.a, A6.a), because
choosing a metric after seeing the numbers is how a post-hoc story becomes an
apparent finding.

  A1.a  RECOVERY, two metrics, both reported:
        (i)  folds_to_recovery = number of post-switch folds before per-fold
             AUC first returns to within EPS = 0.02 of the pre-switch mean.
             If it never returns, the value is n_post_folds (right-censored)
             and the count of censored seeds is reported alongside.
        (ii) post_mean = mean per-fold AUC over post-switch folds.
             A blunt area-under-the-curve proxy; no threshold, so it cannot be
             gamed by the choice of EPS.

  A1.b  FOLD GROUPING: from `fold_phase`, NOT `fold_is_pre`.
        fold_is_pre pools the straddling fold into POST, which is wrong -- a
        straddle fold is neither, and orpsoc_utils itself flags this. Folds are
        grouped pre / straddle / post and the straddle fold is reported
        SEPARATELY and excluded from both groups.

  A6.a  BREAK-ADJACENT = break_folds ∪ (break_folds + 1).
        Rationale for the +1: a walk-forward learner cannot react to a break
        until that break has entered its TRAINING window, which happens one
        fold after the fold whose TEST window contains it.

  STATISTICS: paired per-seed Wilcoxon signed-rank, two-tailed. Every family of
  comparisons is reported with a Benjamini-Hochberg q alongside the raw p. The
  synthetic table has 30 seeds and the real table 20; with that many
  comparisons, uncorrected p-values are not interpretable on their own.

Run:  python experiments/tier_a.py
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

from scipy import stats

EPS = 0.02                     # A1.a(i) recovery tolerance -- pre-registered
ABLATION = "results/step7_ablation_v2.json"
COND = ["baseline", "standard_orpsoc", "apsoll", "full_hybrid",
        "full_hybrid_noimp"]
LABEL = {"baseline": "Baseline", "standard_orpsoc": "OrPSOC",
         "apsoll": "+APSOLL", "full_hybrid": "Full Hybrid",
         "full_hybrid_noimp": "FH no-imp"}


def bh(pvals):
    """Benjamini-Hochberg q-values, order-preserving."""
    p = np.asarray(pvals, float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        val = min(prev, p[i] * m / (m - rank + 1))
        q[i] = prev = val
    return q


def wilcoxon(a, b):
    """Paired two-tailed Wilcoxon; returns nan when undefined (all-tied)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 5 or np.allclose(a, b):
        return float("nan")
    try:
        return float(stats.wilcoxon(a, b).pvalue)
    except Exception:
        return float("nan")


# ══════════════════════════════════════════════════════════════════════════════
#  A1 — pre / post-switch recovery split
# ══════════════════════════════════════════════════════════════════════════════
def a1(full, out):
    lvl = "v2_regime_switch"
    conds = full[lvl]["conditions"]
    phases = conds["full_hybrid"][0]["fold_phase"]
    phase_of = [p["phase"] if isinstance(p, dict) else p for p in phases]
    pre_i = [i for i, p in enumerate(phase_of) if p == "pre"]
    strad = [i for i, p in enumerate(phase_of) if p == "straddle"]
    post_i = [i for i, p in enumerate(phase_of) if p == "post"]

    print("=" * 78)
    print("  A1 — PRE / POST-SWITCH RECOVERY  (Level 4, v2_regime_switch)")
    print("=" * 78)
    print(f"  fold phases      : {phase_of}")
    print(f"  pre folds        : {pre_i}")
    print(f"  straddle (EXCLUDED, reported separately) : {strad}")
    print(f"  post folds       : {post_i}")
    print(f"  recovery epsilon : {EPS}  (pre-registered)")
    print()

    rows = {}
    for c in COND:
        seeds = conds[c]
        pre_m, post_m, str_m, ftr, cens = [], [], [], [], 0
        for sr in seeds:
            fa = np.asarray(sr["fold_aucs"], float)
            if len(fa) <= max(post_i + pre_i):
                continue
            pm = fa[pre_i].mean()
            pre_m.append(pm)
            post_m.append(fa[post_i].mean())
            if strad:
                str_m.append(fa[strad].mean())
            # (i) folds to recovery, right-censored at n_post
            k = len(post_i)
            for j, fi in enumerate(post_i, start=1):
                if fa[fi] >= pm - EPS:
                    k = j
                    break
            else:
                cens += 1
            ftr.append(k)
        rows[c] = {"pre_mean": float(np.mean(pre_m)),
                   "post_mean": float(np.mean(post_m)),
                   "straddle_mean": float(np.mean(str_m)) if str_m else None,
                   "drop": float(np.mean(pre_m) - np.mean(post_m)),
                   "folds_to_recovery": float(np.mean(ftr)),
                   "censored_seeds": cens,
                   "_pre": pre_m, "_post": post_m, "_ftr": ftr}

    print(f"  {'condition':<14}{'pre':>9}{'straddle':>10}{'post':>9}"
          f"{'drop':>9}{'folds->rec':>12}{'censored':>10}")
    print("  " + "-" * 74)
    for c in COND:
        r = rows[c]
        sm = f"{r['straddle_mean']:>10.4f}" if r["straddle_mean"] is not None else f"{'-':>10}"
        print(f"  {LABEL[c]:<14}{r['pre_mean']:>9.4f}{sm}{r['post_mean']:>9.4f}"
              f"{r['drop']:>9.4f}{r['folds_to_recovery']:>12.2f}"
              f"{r['censored_seeds']:>8}/30")

    # A1.c — paired per-seed Wilcoxon vs baseline, within each group
    print()
    print("  A1.c  paired per-seed Wilcoxon vs BASELINE (two-tailed)")
    fam = []
    for grp in ("_pre", "_post"):
        for c in COND[1:]:
            p = wilcoxon(rows[c][grp], rows["baseline"][grp])
            fam.append((grp.strip("_"), c, np.mean(rows[c][grp]) - np.mean(rows["baseline"][grp]), p))
    qs = bh([f[3] for f in fam])
    print(f"    {'group':<8}{'condition':<14}{'delta vs base':>15}{'p':>9}{'BH q':>9}")
    for (g, c, d, p), q in zip(fam, qs):
        print(f"    {g:<8}{LABEL[c]:<14}{d:>+15.4f}{p:>9.4f}{q:>9.3f}")

    out["A1"] = {"phases": phase_of,
                 "rows": {c: {k: v for k, v in rows[c].items()
                              if not k.startswith("_")} for c in COND},
                 "wilcoxon_vs_baseline": [
                     {"group": g, "condition": c, "delta": d, "p": p, "q": float(q)}
                     for (g, c, d, p), q in zip(fam, qs)]}

    # A1.d — the claim, stated at supported strength
    b, fh = rows["baseline"], rows["full_hybrid"]
    print()
    print("  A1.d  CLAIM AT SUPPORTED STRENGTH")
    if fh["post_mean"] > b["post_mean"]:
        print("    Full Hybrid post-switch mean EXCEEDS baseline -- check q before claiming.")
    else:
        print(f"    Baseline post-switch mean ({b['post_mean']:.4f}) EXCEEDS Full Hybrid "
              f"({fh['post_mean']:.4f}).")
        print("    => 'faster recovery than baseline' is NOT supported. The defensible")
        print("       claim is about adaptively-correct SUBSETS at comparable AUC (see A2/A4).")
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  A2 — recall of post-switch signal features
# ══════════════════════════════════════════════════════════════════════════════
def a2(full, out):
    lvl = "v2_regime_switch"
    conds = full[lvl]["conditions"]
    phase_of = [p["phase"] if isinstance(p, dict) else p
                for p in conds["full_hybrid"][0]["fold_phase"]]
    pre_i = [i for i, p in enumerate(phase_of) if p == "pre"]
    post_i = [i for i, p in enumerate(phase_of) if p == "post"]

    print()
    print("=" * 78)
    print("  A2 — SIGNAL RECALL ACROSS THE SWITCH")
    print("     r2 = post-switch signals (should RISE after the break)")
    print("     r1 = pre-switch signals  (mirror image: should FALL)")
    print("=" * 78)
    print(f"  {'condition':<14}{'r2 pre':>9}{'r2 post':>9}{'r2 rise':>9}"
          f"{'p':>9}{'r1 pre':>9}{'r1 post':>9}{'r1 fall':>9}{'p':>9}")
    print("  " + "-" * 76)

    res, fam = {}, []
    for c in COND:
        seeds = conds[c]
        r2p, r2q, r1p, r1q = [], [], [], []
        for sr in seeds:
            r2 = np.asarray(sr["fold_r2_hits"], float)
            r1 = np.asarray(sr["fold_r1_hits"], float)
            if len(r2) <= max(post_i):
                continue
            r2p.append(r2[pre_i].mean()); r2q.append(r2[post_i].mean())
            r1p.append(r1[pre_i].mean()); r1q.append(r1[post_i].mean())
        p2 = wilcoxon(r2q, r2p)
        p1 = wilcoxon(r1q, r1p)
        fam += [p2, p1]
        res[c] = {"r2_pre": float(np.mean(r2p)), "r2_post": float(np.mean(r2q)),
                  "r2_rise": float(np.mean(r2q) - np.mean(r2p)), "r2_p": p2,
                  "r1_pre": float(np.mean(r1p)), "r1_post": float(np.mean(r1q)),
                  "r1_fall": float(np.mean(r1p) - np.mean(r1q)), "r1_p": p1}
        r = res[c]
        print(f"  {LABEL[c]:<14}{r['r2_pre']:>9.3f}{r['r2_post']:>9.3f}"
              f"{r['r2_rise']:>+9.3f}{p2:>9.4f}"
              f"{r['r1_pre']:>9.3f}{r['r1_post']:>9.3f}{r['r1_fall']:>+9.3f}{p1:>9.4f}")

    qs = bh(fam)
    for i, c in enumerate(COND):
        res[c]["r2_q"] = float(qs[2 * i])
        res[c]["r1_q"] = float(qs[2 * i + 1])

    print()
    print("  A2.b  THE SPECIFIC CLAIM: r2 recall rises after the switch for Full")
    print("        Hybrid but NOT for standard OrPSOC.")
    fh, so = res["full_hybrid"], res["standard_orpsoc"]
    print(f"        Full Hybrid rise {fh['r2_rise']:+.3f} (q={fh['r2_q']:.3f});  "
          f"OrPSOC rise {so['r2_rise']:+.3f} (q={so['r2_q']:.3f})")
    ok = fh["r2_rise"] > 0 and fh["r2_q"] < 0.05 and not (so["r2_rise"] > 0 and so["r2_q"] < 0.05)
    print(f"        => claim {'SUPPORTED' if ok else 'NOT supported as stated'}")
    out["A2"] = res


# ══════════════════════════════════════════════════════════════════════════════
#  A3 — Jaccard dip-and-recovery, tested rather than eyeballed
# ══════════════════════════════════════════════════════════════════════════════
def a3(full, out):
    lvl = "v2_regime_switch"
    conds = full[lvl]["conditions"]
    phase_of = [p["phase"] if isinstance(p, dict) else p
                for p in conds["full_hybrid"][0]["fold_phase"]]
    # per_fold_jaccard[i] compares fold i against fold i+1 (verified in
    # orpsoc_utils.compute_jaccard_metrics: `for i in range(n-1)` over
    # selected_sets[i] vs [i+1]).
    #
    # WHICH PAIR IS "THE SWITCH PAIR"? The break first appears in the TEST
    # window of the straddle fold, but a walk-forward learner cannot react to it
    # until it enters the TRAINING window -- one fold later. So the first pair
    # across which the subset CAN legitimately change is
    # (straddle, straddle+1) = index strad[0].
    #
    # An earlier version used strad[0]-1, i.e. the pre->straddle pair. That is
    # BEFORE adaptation is causally possible, so it measures ordinary churn and
    # reports a spurious "spike" rather than the adaptation dip.
    strad = [i for i, p in enumerate(phase_of) if p == "straddle"]
    sp = strad[0] if strad else len(phase_of) // 2

    print()
    print("=" * 78)
    print("  A3 — JACCARD DIP-AND-RECOVERY, TESTED")
    print(f"     switch fold-pair index = {sp}; compared against its neighbours")
    print("=" * 78)
    print(f"  {'condition':<14}{'J@switch':>10}{'J@adjacent':>12}{'dip':>9}{'p':>9}{'BH q':>9}")
    print("  " + "-" * 63)

    res, ps = {}, []
    for c in COND:
        at, adj = [], []
        for sr in conds[c]:
            j = sr.get("jaccard") or {}
            pf = j.get("per_fold_jaccard") or []
            if len(pf) <= sp + 1 or sp < 1:
                continue
            at.append(pf[sp])
            adj.append(np.mean([pf[sp - 1], pf[sp + 1]]))
        p = wilcoxon(at, adj) if at else float("nan")
        ps.append(p)
        res[c] = {"at_switch": float(np.mean(at)) if at else None,
                  "adjacent": float(np.mean(adj)) if adj else None,
                  "dip": float(np.mean(adj) - np.mean(at)) if at else None, "p": p}
    qs = bh(ps)
    for c, q in zip(COND, qs):
        res[c]["q"] = float(q)
        r = res[c]
        if r["at_switch"] is None:
            print(f"  {LABEL[c]:<14}{'n/a':>10}")
            continue
        print(f"  {LABEL[c]:<14}{r['at_switch']:>10.3f}{r['adjacent']:>12.3f}"
              f"{r['dip']:>+9.3f}{r['p']:>9.4f}{q:>9.3f}")

    # Full trace, so the reader can see the whole shape rather than trusting
    # one index. Pair i sits between fold i and fold i+1.
    print()
    print("  full per-fold-pair Jaccard trace (mean across seeds):")
    hdr = "".join(f"{i}-{i+1}".rjust(8) for i in range(len(phase_of) - 1))
    print(f"    {'pair':<14}{hdr}")
    for c in COND:
        tr = []
        for i in range(len(phase_of) - 1):
            vals = [sr["jaccard"]["per_fold_jaccard"][i]
                    for sr in conds[c]
                    if (sr.get("jaccard") or {}).get("per_fold_jaccard")
                    and len(sr["jaccard"]["per_fold_jaccard"]) > i]
            tr.append(np.mean(vals) if vals else float("nan"))
        mark = "".join(f"{v:>8.3f}" for v in tr)
        print(f"    {LABEL[c]:<14}{mark}")
        res[c]["trace"] = [float(v) for v in tr]
    print(f"    {'':<14}" + "".join(
        ("  <SWITCH" if i == sp else "").rjust(8)
        for i in range(len(phase_of) - 1)))
    print()
    print("  A dip is only evidence of adaptation if it is LARGER than the")
    print("  fold-to-fold churn the selector shows anyway -- that is what the")
    print("  adjacent-pair comparison controls for.")

    # EXPLORATORY, clearly separated from the pre-registered test above.
    fh_tr = res["full_hybrid"].get("trace") or []
    if fh_tr:
        lo = int(np.nanargmin(fh_tr))
        if lo != sp:
            print()
            print("  EXPLORATORY (NOT pre-registered -- do not report as a test):")
            print(f"    Full Hybrid's Jaccard MINIMUM is at pair {lo}-{lo+1} "
                  f"({fh_tr[lo]:.3f}), not at the pre-registered switch pair "
                  f"{sp}-{sp+1} ({fh_tr[sp]:.3f}).")
            print(f"    The trace does show a dip-and-recovery shape "
                  f"({fh_tr[sp]:.3f} -> {fh_tr[lo]:.3f} -> "
                  f"{fh_tr[min(lo+1, len(fh_tr)-1)]:.3f}), but one pair later than")
            print("    causal reasoning predicts. Moving the test index to match the")
            print("    observed minimum would be post-hoc fold selection -- exactly")
            print("    what A6.a warns against. Treat this as a HYPOTHESIS to be")
            print("    pre-registered and tested on a fresh benchmark draw.")
            res["exploratory_min_pair"] = {"index": lo, "value": float(fh_tr[lo]),
                                           "preregistered_index": sp}
    out["A3"] = res


# ══════════════════════════════════════════════════════════════════════════════
#  A4 — compactness, the trade-off statement, runtime
# ══════════════════════════════════════════════════════════════════════════════
def a4(full, summary, out):
    print()
    print("=" * 78)
    print("  A4 — COMPACTNESS / RUNTIME TRADE-OFF")
    print("=" * 78)
    res = {}
    for lvl in [k for k in full if not k.endswith("@logreg")]:
        conds = full[lvl]["conditions"]
        print(f"\n  {lvl}")
        print(f"    {'condition':<14}{'mean AUC':>10}{'mean k':>9}{'% of N':>9}"
              f"{'runtime s':>11}")
        print("    " + "-" * 53)
        res[lvl] = {}
        base_auc = np.mean(summary[lvl]["baseline"]["seed_aucs"])
        for c in COND:
            ks = [np.mean([len(s) for s in sr["fold_selected"]])
                  for sr in conds[c]]
            rt = [np.mean(sr["runtimes"]) for sr in conds[c]]
            auc = np.mean(summary[lvl][c]["seed_aucs"])
            N = 50
            res[lvl][c] = {"auc": float(auc), "mean_k": float(np.mean(ks)),
                           "pct_features": float(100 * np.mean(ks) / N),
                           "runtime_s": float(np.mean(rt)),
                           "pct_of_baseline_auc": float(100 * auc / base_auc)}
            r = res[lvl][c]
            print(f"    {LABEL[c]:<14}{auc:>10.4f}{r['mean_k']:>9.1f}"
                  f"{r['pct_features']:>8.0f}%{r['runtime_s']:>11.2f}")
        fh = res[lvl]["full_hybrid"]
        print(f"    => Full Hybrid attains {fh['pct_of_baseline_auc']:.1f}% of baseline AUC "
              f"using {fh['pct_features']:.0f}% of the features.")
    out["A4"] = res


# ══════════════════════════════════════════════════════════════════════════════
#  A6 — sector-ETF break-adjacent folds (real data, from step9 checkpoints)
# ══════════════════════════════════════════════════════════════════════════════
def a6(out):
    print()
    print("=" * 78)
    print("  A6 — SECTOR-ETF BREAK-ADJACENT FOLDS")
    print("     pre-registered: break-adjacent = break_folds U (break_folds + 1)")
    print("=" * 78)

    try:
        s9 = json.load(open("results/step9_real_data.json"))
    except FileNotFoundError:
        print("  results/step9_real_data.json absent -- skipping A6.")
        return

    ck = sorted(glob.glob("results/checkpoints/step9/*/*.pkl"))
    if not ck:
        ck = sorted(glob.glob(
            "results_ec2/*/results/checkpoints/step9/*/*.pkl"))
    if not ck:
        print("  no step9 checkpoints found -- A6 needs per-seed folds.")
        print("  (step9's JSON stores aggregates only; see step9_real_data.py fix.)")
        return

    res = {}
    for ds in s9.get("datasets", {}):
        brk = s9["datasets"][ds].get("break_folds") or []
        if not brk:
            continue
        adjacent = sorted({b for b in brk} | {b + 1 for b in brk})
        units = [f for f in ck if os.path.basename(f).startswith(ds + "_seed")]
        if not units:
            continue
        per = {c: [] for c in COND}
        ksz = {c: [] for c in COND}
        for f in units:
            pay = pickle.load(open(f, "rb"))["payload"]
            for c in COND:
                if c not in pay:
                    continue
                fa = np.asarray(pay[c]["fold_aucs"], float)
                idx = [i for i in adjacent if i < len(fa)]
                if not idx:
                    continue
                per[c].append(fa[idx].mean())
                sel = pay[c].get("fold_selected") or []
                if sel:
                    ksz[c].append(np.mean([len(sel[i]) for i in idx if i < len(sel)]))
        if not per["baseline"]:
            continue
        print(f"\n  {ds}   break_folds={brk}  break-adjacent={adjacent}  "
              f"n_seeds={len(per['baseline'])}")
        print(f"    {'condition':<14}{'AUC@adj':>10}{'vs base':>10}{'p':>9}"
              f"{'BH q':>9}{'mean k':>9}")
        print("    " + "-" * 61)
        ps = [wilcoxon(per[c], per["baseline"]) for c in COND[1:]]
        qs = bh(ps)
        res[ds] = {"break_folds": brk, "adjacent": adjacent,
                   "n_seeds": len(per["baseline"]), "conditions": {}}
        bm = np.mean(per["baseline"])
        res[ds]["conditions"]["baseline"] = {
            "auc_adjacent": float(bm),
            "mean_k": float(np.mean(ksz["baseline"])) if ksz["baseline"] else None}
        print(f"    {'Baseline':<14}{bm:>10.4f}{'-':>10}{'-':>9}{'-':>9}"
              f"{(np.mean(ksz['baseline']) if ksz['baseline'] else float('nan')):>9.1f}")
        for c, p, q in zip(COND[1:], ps, qs):
            if not per[c]:
                continue
            m = np.mean(per[c])
            k = np.mean(ksz[c]) if ksz[c] else float("nan")
            print(f"    {LABEL[c]:<14}{m:>10.4f}{m - bm:>+10.4f}{p:>9.4f}{q:>9.3f}{k:>9.1f}")
            res[ds]["conditions"][c] = {"auc_adjacent": float(m),
                                        "delta_vs_baseline": float(m - bm),
                                        "p": p, "q": float(q),
                                        "mean_k": float(k) if k == k else None}
    print()
    print("  A6.e  Why a break shows up one fold LATE: walk-forward causality.")
    print("        A learner cannot react to a break until it has entered the")
    print("        TRAINING window, which is one fold after the fold whose TEST")
    print("        window first contains it. A fire at break_fold+1 is therefore")
    print("        the earliest CAUSALLY POSSIBLE response, not a lag to explain away.")
    out["A6"] = res


def main():
    d = json.load(open(ABLATION))
    full, summary = d["full_results"], d["summary"]
    print()
    print("#" * 78)
    print("#  TIER A  —  analysis only, no re-runs")
    print(f"#  source: {ABLATION}")
    print(f"#  config: {d['config']['n_seeds']} seeds, "
          f"{d['config']['n_splits']} folds, theta={d['config']['theta']}, "
          f"provenance={d['provenance']['hash']}")
    print("#" * 78)

    out = {"source": ABLATION, "provenance": d["provenance"]["hash"],
           "eps": EPS}
    a1(full, out)
    a2(full, out)
    a3(full, out)
    a4(full, summary, out)
    a6(out)

    os.makedirs("results", exist_ok=True)
    with open("results/tier_a.json", "w") as f:
        json.dump(out, f, indent=2)
    print()
    print("Saved: results/tier_a.json")


if __name__ == "__main__":
    main()
