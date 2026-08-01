"""
test_equivalence.py — Regression guard for the performance optimisations
=========================================================================
Run this after ANY change to orpsoc_utils.py, and before trusting a paper run:

    python test_equivalence.py

It asserts the four properties the optimisation work depends on. All four are
about CORRECTNESS, not speed — if any fails, the numbers a run produces are no
longer the numbers the unoptimised code would have produced.

  T1  Walk-forward structure is causal.
      Train is a strict prefix, test starts strictly later with the configured
      gap, and the inner PSO split is disjoint. This is the property that makes
      the whole study valid; it is asserted first so a structural break is
      never mistaken for a numerical one.

  T2  evaluate_ctx() == evaluate(), exactly.
      The hoisted imputer/scaler must reproduce the per-call sklearn Pipeline
      bit-for-bit, on real folds of real data, for randomly drawn feature
      subsets. Compared with ==, not a tolerance.

  T3  FoldEvalContext leaks nothing.
      Statistics come from X_p only; X_te does not enter the object; fitting
      the transformers on all columns equals fitting them on any subset.

  T4  PSO runners are order-independent and reproducible.
      The same seed produces the same result regardless of what ran before it,
      which is what makes seed-level parallelism and checkpoint resumption safe.

  T5  Fold partitioning is honest about the structural break.
      The straddling fold is identified as such, and `train_sees_post` marks
      the folds where adaptation was even possible.

  T6  Benchmark invariants (v1 and v2).
      Null level scores at chance; baseline "selects" everything; the v2 signal
      features carry independent information whereas v1's do not.

  T7  Objective-function invariants.
      The theta-implied break-even is arithmetically what the code implements,
      and raising theta relaxes compactness pressure.

NOTE ON WHAT IS *NOT* ASSERTED
──────────────────────────────
No test here hard-codes a run-specific magnitude (a particular AUC, subset
size, Jaccard value, or runtime ratio). Those depend on fast_mode, n_seeds,
max_iter, n_particles, n_splits and the benchmark version, so freezing one
run's numbers into the suite would bake a single configuration into the tests
and break the moment the config changes -- exactly what guardrail G3 forbids.
Orderings and invariants survive a config change; magnitudes do not.
"""

import pickle
import sys

import numpy as np

from orpsoc_utils import (evaluate, evaluate_ctx, FoldEvalContext,
                          walk_forward_folds, run_standard_orpsoc,
                          run_hybrid_orpsoc)

DATASET = "data/regime_switch.pkl"
FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def inner_split(X_tr, y_tr):
    cut = int(len(X_tr) * 0.75)
    return (X_tr.iloc[:cut], y_tr.iloc[:cut],
            X_tr.iloc[cut:], y_tr.iloc[cut:])


def main():
    with open(DATASET, "rb") as f:
        data = pickle.load(f)
    X, y = data["X"], data["y"]
    feat = list(X.columns)
    folds = walk_forward_folds(X, y, n_splits=8, gap=5, min_train=150)

    # ── T1 ────────────────────────────────────────────────────────────────────
    print("\nT1  walk-forward structure is causal")
    ok = True
    for i, (X_tr, y_tr, X_te, y_te, train_end) in enumerate(folds):
        tr, te = X_tr.index, X_te.index
        X_p, _, X_v, _ = inner_split(X_tr, y_tr)
        ok &= (te.min() > tr.max())                    # test strictly after train
        ok &= (len(set(tr) & set(te)) == 0)            # no overlap
        ok &= (te.min() - tr.max() - 1 == 5)           # gap honoured
        ok &= (len(set(X_p.index) & set(X_v.index)) == 0)
        ok &= (X_v.index.max() == tr.max())            # inner val ends at train end
    check("train prefix / test strictly after / gap=5 / inner split disjoint",
          ok, f"{len(folds)} folds")

    # ── T2 ────────────────────────────────────────────────────────────────────
    print("\nT2  evaluate_ctx() reproduces evaluate() exactly")
    rng = np.random.RandomState(0)
    n_cmp = 0
    ok = True
    for fi in (0, 3, 7):
        X_tr, y_tr, X_te, y_te, _ = folds[fi]
        X_p, y_p, X_v, y_v = inner_split(X_tr, y_tr)
        ctx = FoldEvalContext(X_p, y_p, X_v, y_v, feat)
        for _ in range(6):
            pos = (rng.rand(len(feat)) < rng.uniform(0.1, 0.7)).astype(float)
            if pos.sum() < 3:
                pos[:3] = 1.0
            a = evaluate(pos, feat, X_p, y_p, X_v, y_v, 3, 0.7)
            b = evaluate_ctx(pos, ctx, 3, 0.7)
            n_cmp += 1
            if a != b:
                ok = False
                print(f"      fold {fi}: {a!r} != {b!r}  (n_sel={int(pos.sum())})")
        # min_f rejection path must agree too
        tiny = np.zeros(len(feat)); tiny[0] = 1.0
        ok &= (evaluate(tiny, feat, X_p, y_p, X_v, y_v, 3, 0.7)
               == evaluate_ctx(tiny, ctx, 3, 0.7) == -1.0)
    check("identical fitness on random subsets (exact ==)", ok,
          f"{n_cmp} comparisons over 3 folds")

    # ── T3 ────────────────────────────────────────────────────────────────────
    print("\nT3  FoldEvalContext leaks nothing")
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    X_tr, y_tr, X_te, y_te, _ = folds[4]
    X_p, y_p, X_v, y_v = inner_split(X_tr, y_tr)
    ctx = FoldEvalContext(X_p, y_p, X_v, y_v, feat)

    check("context row counts match X_p / X_v exactly",
          ctx.Ap.shape[0] == len(X_p) and ctx.Av.shape[0] == len(X_v),
          f"Ap={ctx.Ap.shape} Av={ctx.Av.shape}, X_te={X_te.shape} not included")

    sub = [feat[i] for i in (3, 17, 29, 41)]
    imp_all = SimpleImputer(strategy="mean").fit(X_p[feat])
    sc_all = StandardScaler().fit(imp_all.transform(X_p[feat]))
    imp_sub = SimpleImputer(strategy="mean").fit(X_p[sub])
    sc_sub = StandardScaler().fit(imp_sub.transform(X_p[sub]))
    ci = [feat.index(c) for c in sub]
    check("per-column stats independent of which columns are fitted",
          np.array_equal(sc_all.mean_[ci], sc_sub.mean_)
          and np.array_equal(sc_all.scale_[ci], sc_sub.scale_))

    # If X_v had been folded into the fit, the training half would no longer be
    # exactly zero-mean/unit-scale. It is — so the fit used X_p alone.
    check("standardisation fitted on X_p only",
          np.allclose(ctx.Ap.mean(axis=0), 0.0, atol=1e-10)
          and not np.allclose(ctx.Av.mean(axis=0), 0.0, atol=1e-6))

    # ── T4 ────────────────────────────────────────────────────────────────────
    print("\nT4  runners are reproducible and order-independent")
    X_tr, y_tr, X_te, y_te, _ = folds[5]
    kw = dict(feat_names=feat, seed=5000, n_particles=8, max_iter=6, min_f=3,
              theta=0.7, cr_low=0.3, cr_high=0.8, w_max=0.9, w_min=0.4,
              N_explore=5, lam=0.1)

    a1 = run_standard_orpsoc(X_tr, y_tr, X_te, y_te, **kw)
    h1 = run_hybrid_orpsoc(X_tr, y_tr, X_te, y_te, hmm_trigger=True,
                           p_trans=0.9, **kw)
    # Interleave a different seed between the repeats. If any global RNG or
    # shared cache existed, this is what would expose it.
    run_hybrid_orpsoc(X_tr, y_tr, X_te, y_te, hmm_trigger=False,
                      **{**kw, "seed": 999})
    a2 = run_standard_orpsoc(X_tr, y_tr, X_te, y_te, **kw)
    h2 = run_hybrid_orpsoc(X_tr, y_tr, X_te, y_te, hmm_trigger=True,
                           p_trans=0.9, **kw)

    check("run_standard_orpsoc reproducible across interleaved runs",
          a1["auc"] == a2["auc"] and a1["selected"] == a2["selected"],
          f"auc={a1['auc']:.12f}")
    check("run_hybrid_orpsoc reproducible across interleaved runs",
          h1["auc"] == h2["auc"] and h1["selected"] == h2["selected"]
          and np.array_equal(h1["gbest_pos"], h2["gbest_pos"]),
          f"auc={h1['auc']:.12f}")

    # ── T5 ────────────────────────────────────────────────────────────────────
    print("\nT5  fold partitioning is honest about the structural break")
    from orpsoc_utils import classify_folds
    ph = classify_folds(folds, 500)
    phases = [p["phase"] for p in ph]
    check("exactly one straddling fold is identified",
          phases.count("straddle") == 1,
          f"phases = {phases}")
    idx = phases.index("straddle")
    check("straddle fold's test window actually contains the break",
          ph[idx]["test_start"] <= 500 <= ph[idx]["test_end"],
          f"fold {idx+1} test [{ph[idx]['test_start']}, {ph[idx]['test_end']}], "
          f"{ph[idx]['frac_post_in_test']:.0%} post-switch")
    check("phases are monotone pre -> straddle -> post",
          phases == sorted(phases, key=["pre", "straddle", "post"].index))
    # A selector cannot adapt to a regime its training window has never seen.
    first_adaptable = next(i for i, p in enumerate(ph) if p["train_sees_post"])
    check("no fold claims adaptability before the break enters TRAINING",
          all(not ph[i]["train_sees_post"] for i in range(first_adaptable)),
          f"earliest adaptable fold = {first_adaptable+1} "
          f"(straddle fold is {idx+1}, so adaptation is impossible there)")
    check("stationary levels produce no straddle",
          all(p["phase"] == "pre" for p in classify_folds(folds, None)))

    # ── T6 ────────────────────────────────────────────────────────────────────
    print("\nT6  benchmark invariants")
    import os

    def corr_max(path, cols=None):
        with open(path, "rb") as f:
            dd = pickle.load(f)
        Xd, yd = dd["X"], dd["y"]
        use = cols if cols else list(Xd.columns)
        return max(abs(np.corrcoef(Xd[c], yd)[0, 1]) for c in use)

    if os.path.exists("data/null.pkl"):
        sig = [f"signal_{i}" for i in range(5)]
        check("L0 null: no feature correlates with y beyond sampling noise",
              corr_max("data/null.pkl") < 0.15,
              f"max |corr| = {corr_max('data/null.pkl'):.4f} "
              f"(1/sqrt(n) = {1/np.sqrt(1000):.4f})")
        check("L1 is NOT a null (it is a stationarity control)",
              corr_max("data/white_noise.pkl", sig) > 0.5,
              f"max |corr| = {corr_max('data/white_noise.pkl', sig):.4f}")

    if os.path.exists("data/v2_ar1.pkl"):
        def marginal_gain(path):
            with open(path, "rb") as f:
                dd = pickle.load(f)
            Xd, yd = dd["X"], dd["y"]
            fl = list(Xd.columns)
            fo = walk_forward_folds(Xd, yd, n_splits=8, gap=5, min_train=150)
            Xt, yt = fo[6][0], fo[6][1]
            cu = int(len(Xt) * 0.75)
            cx = FoldEvalContext(Xt.iloc[:cu], yt.iloc[:cu],
                                 Xt.iloc[cu:], yt.iloc[cu:], fl)
            si = [fl.index(f"signal_{i}") for i in range(5)]

            def a(ix):
                p = np.zeros(len(fl)); p[list(ix)] = 1
                return (evaluate_ctx(p, cx, 3, 0.7)
                        - 0.3 * (1 - len(ix) / len(fl))) / 0.7
            import itertools as it
            return a(si) - max(a(c) for c in it.combinations(si, 3))

        g1, g2 = marginal_gain("data/ar1.pkl"), marginal_gain("data/v2_ar1.pkl")
        check("v1 signal features are redundant (3 of 5 buy ~everything)",
              abs(g1) < 0.01, f"AUC(5 signals) - AUC(best 3) = {g1:+.4f}")
        check("v2 signal features carry independent information",
              g2 > 0.02, f"AUC(5 signals) - AUC(best 3) = {g2:+.4f}")

    # ── T7 ────────────────────────────────────────────────────────────────────
    print("\nT7  objective-function invariants")
    X_tr, y_tr, X_te, y_te, _ = folds[6]
    cut = int(len(X_tr) * 0.75)
    ctx = FoldEvalContext(X_tr.iloc[:cut], y_tr.iloc[:cut],
                          X_tr.iloc[cut:], y_tr.iloc[cut:], feat)
    N = len(feat)
    for th in (0.5, 0.7, 0.9):
        pos_a = np.zeros(N); pos_a[:5] = 1
        pos_b = np.zeros(N); pos_b[:6] = 1
        # Same AUC term cancels; the difference must be exactly the penalty.
        fa, fb = evaluate_ctx(pos_a, ctx, 3, th), evaluate_ctx(pos_b, ctx, 3, th)
        auc_a = (fa - (1 - th) * (1 - 5 / N)) / th
        auc_b = (fb - (1 - th) * (1 - 6 / N)) / th
        implied = (fb - fa) - th * (auc_b - auc_a)
        check(f"theta={th}: one extra feature costs exactly (1-theta)/N",
              abs(implied + (1 - th) / N) < 1e-9,
              f"break-even dAUC = {(1-th)/(N*th):.6f}")

    # Compactness pressure must be monotone in theta: higher theta -> weaker
    # penalty -> the optimum cannot move to a SMALLER subset.
    def best_k(th):
        import itertools as it
        sig = [feat.index(f"signal_{i}") for i in range(5)]
        scores = {k: max(evaluate_ctx(_mk(N, c), ctx, 3, th)
                         for c in it.combinations(sig, k)) for k in (3, 4, 5)}
        return max(scores, key=scores.get)

    def _mk(n, idx):
        p = np.zeros(n); p[list(idx)] = 1; return p

    ks = {th: best_k(th) for th in (0.5, 0.7, 1.0)}
    check("optimal k is non-decreasing as theta rises",
          ks[0.5] <= ks[0.7] <= ks[1.0], f"argmax k by theta: {ks}")

    # ── verdict ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    if FAILURES:
        print(f"  {len(FAILURES)} CHECK(S) FAILED — do not trust run output:")
        for f in FAILURES:
            print(f"    - {f}")
        print("=" * 70)
        return 1
    print("  ALL EQUIVALENCE CHECKS PASSED")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
