"""
orpsoc_criteria.py — candidate fitness criteria for the selection bake-off
==========================================================================
The measured problem (see reports/): the quantity OrPSOC maximises barely
predicts the quantity we report. Correlation between inner-validation AUC and
test AUC is 0.14-0.48 on real data and -0.03 at the first post-switch fold on
synthetic -- i.e. at exactly the fold the study is about, the compass reads
noise. Standard trailing-window validation optimises for the OLD regime.

Every criterion here is computed CAUSALLY, using only rows inside the training
window of the current walk-forward fold. X_te is never touched.

  current      AUC on the last 25% of X_tr, single fit.  <- what the code does
  mean_k       mean AUC over K contiguous inner walk-forward blocks
  median_k     median over those blocks
  min_k        worst block (distributionally robust / worst-case)
  mean_sd      mean - std over blocks (variance-penalised)
  mb_stability MEAN Meinshausen-Buhlmann selection frequency (size-blind; see
               the note at its scoring branch -- kept only as a documented trap)
  mb_thresh    MB selection frequency in classical threshold form: sum of
               (freq - pi) over members, so the optimum is 'everything above pi'
  mb_perf      mean AUC across moving-block bootstrap resamples
  pooled       regime-pooled validation: HMM-identified states are weighted
               EQUALLY rather than by row count, so a subset that only works
               in the dominant regime is penalised

Design note on the two new ones
-------------------------------
`mb_stability` is the classical Meinshausen-Buhlmann idea adapted to a fitness
function. Classical MB is a *selector*: resample, select, keep features chosen
often. Used as a *score* for a candidate subset, the natural analogue is the
mean selection frequency of that subset's members. The per-feature frequencies
are computed ONCE per fold, so scoring any subset afterwards is a lookup --
which makes this by far the cheapest criterion here despite the bootstrap.

Moving-block (not i.i.d.) bootstrap is used throughout, because i.i.d.
resampling of a time series destroys the autocorrelation that makes these
features what they are.
"""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier


def _fast_model(seed=42):
    return LGBMClassifier(n_estimators=40, num_leaves=15, learning_rate=0.1,
                          verbosity=-1, random_state=seed, n_jobs=1)


def _auc(Atr, ytr, Ava, yva, idx, seed=42):
    """AUC of a column subset on pre-transformed arrays. NaN if degenerate."""
    if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2 or len(idx) == 0:
        return np.nan
    try:
        m = _fast_model(seed)
        m.fit(Atr[:, idx], ytr)
        return roc_auc_score(yva, m.predict_proba(Ava[:, idx])[:, 1])
    except Exception:
        return np.nan


def _moving_block_indices(n, block, rng):
    """Moving-block bootstrap index vector of length n (preserves local order)."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.randint(0, max(1, n - block + 1), size=n_blocks)
    return np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]


class CriterionBank:
    """
    Pre-computes everything a fold needs, once, so scoring a candidate subset is
    cheap. Built from the TRAINING window only.

    Parameters
    ----------
    X_tr, y_tr   training window of one walk-forward fold
    feat_names   column order
    k_blocks     inner walk-forward blocks for mean_k / median_k / min_k / mean_sd
    n_boot       moving-block bootstrap resamples for the mb_* criteria
    block_frac   block length as a fraction of the training window
    regime_state per-row regime label (e.g. HMM argmax) for `pooled`; None
                 disables that criterion
    """

    def __init__(self, X_tr, y_tr, feat_names, k_blocks=4, n_boot=12,
                 block_frac=0.10, regime_state=None, seed=0, pi_thresh=0.6,
                 criteria=None):
        """
        criteria : optional list of the criterion names that will actually be
            requested. The moving-block bootstrap and the Meinshausen-Buhlmann
            selection frequencies cost n_boot model fits to build and are used
            only by mb_perf / mb_stability / mb_thresh. When a caller asks for,
            say, median_k alone, building them is pure waste -- and inside a PSO
            run the bank is rebuilt once per condition per fold, so the waste
            multiplies. Passing `criteria` skips whatever is not needed; leaving
            it None builds everything (backward compatible).
        """
        _want = set(criteria) if criteria else None
        _need_boot = (_want is None
                      or bool(_want & {"mb_perf", "mb_stability", "mb_thresh"}))
        cols = list(feat_names)
        self.n_feat = len(cols)
        self.pi_thresh = pi_thresh
        rng = np.random.RandomState(seed)

        imp = SimpleImputer(strategy="mean").fit(X_tr[cols])
        Z = imp.transform(X_tr[cols])
        sc = StandardScaler().fit(Z)
        A = np.ascontiguousarray(sc.transform(Z))
        y = np.asarray(y_tr)
        n = len(A)

        # ── current: single trailing 25% split ──────────────────────────────
        cut = int(n * 0.75)
        self._cur = (A[:cut], y[:cut], A[cut:], y[cut:])

        # ── contiguous inner walk-forward blocks ────────────────────────────
        edges = [int(n * (i + 1) / (k_blocks + 1)) for i in range(k_blocks + 1)]
        self._blocks = [(A[:edges[i]], y[:edges[i]],
                         A[edges[i]:edges[i + 1]], y[edges[i]:edges[i + 1]])
                        for i in range(k_blocks)]

        # ── moving-block bootstrap resamples (mb_perf) ──────────────────────
        blk = max(5, int(n * block_frac))
        self._boot = []
        for _ in range(n_boot if _need_boot else 0):
            ix = _moving_block_indices(n, blk, rng)
            c = int(len(ix) * 0.75)
            self._boot.append((A[ix[:c]], y[ix[:c]], A[ix[c:]], y[ix[c:]]))

        # ── MB selection frequencies (mb_stability) ─────────────────────────
        # Computed ONCE per fold: fit a fast model on each resample, take the
        # top-m features by importance, count how often each feature appears.
        top_m = max(3, self.n_feat // 5)
        counts = np.zeros(self.n_feat)
        used = 0
        _freq_src = self._boot if (_want is None or
                                   bool(_want & {"mb_stability", "mb_thresh"})) else []
        for (Ab, yb, _, _) in _freq_src:
            if len(np.unique(yb)) < 2:
                continue
            try:
                m = _fast_model(); m.fit(Ab, yb)
                imp_v = np.asarray(m.feature_importances_, dtype=float)
                counts[np.argsort(imp_v)[::-1][:top_m]] += 1
                used += 1
            except Exception:
                continue
        self._freq = counts / used if used else np.full(self.n_feat, np.nan)

        # ── regime-pooled validation split ──────────────────────────────────
        # Train on the first half; validate on each regime's rows in the second
        # half SEPARATELY, then average across regimes with equal weight. Equal
        # weighting is the whole point: it stops the dominant regime from
        # deciding the score on its own.
        self._pools = None
        if regime_state is not None:
            st = np.asarray(regime_state)[:n]
            half = n // 2
            Atr_p, ytr_p = A[:half], y[:half]
            pools = []
            for s in np.unique(st[half:]):
                sel = np.where(st[half:] == s)[0] + half
                if len(sel) >= 20 and len(np.unique(y[sel])) >= 2:
                    pools.append((Atr_p, ytr_p, A[sel], y[sel]))
            self._pools = pools or None

    # ── scoring ─────────────────────────────────────────────────────────────
    def score(self, idx, which):
        """Score one column-index subset under the named criterion."""
        idx = np.asarray(idx, dtype=int)
        if len(idx) == 0:
            return np.nan

        if which == "current":
            return _auc(*self._cur, idx)

        if which in ("mean_k", "median_k", "min_k", "mean_sd"):
            v = np.array([_auc(*b, idx) for b in self._blocks], float)
            v = v[~np.isnan(v)]
            if len(v) == 0:
                return np.nan
            if which == "mean_k":
                return float(v.mean())
            if which == "median_k":
                return float(np.median(v))
            if which == "min_k":
                return float(v.min())
            return float(v.mean() - v.std()) if len(v) > 1 else float(v.mean())

        if which == "mb_perf":
            v = np.array([_auc(*b, idx) for b in self._boot], float)
            v = v[~np.isnan(v)]
            return float(v.mean()) if len(v) else np.nan

        if which == "mb_stability":
            # MEAN selection frequency. Size-blind by construction: a subset of
            # 3 top-frequency features outscores 20 good ones, because the mean
            # is dragged down by every additional member. Measured: this
            # criterion's argmax picks k~5.5 where every other criterion picks
            # ~16-21 from the same uniform k=3..29 pool. Kept for the record --
            # it is the naive reading of "score by selection frequency" and it
            # is a trap. Use mb_thresh.
            f = self._freq[idx]
            return float(np.nanmean(f)) if np.isfinite(f).any() else np.nan

        if which == "mb_thresh":
            # Classical Meinshausen-Buhlmann semantics: keep every feature whose
            # selection frequency exceeds a threshold pi. As a subset score that
            # is the SUM of (freq - pi) over members -- adding a stable feature
            # helps, adding an unstable one hurts, and the optimum is exactly
            # "all features above pi" rather than "the three best".
            f = self._freq[idx]
            if not np.isfinite(f).any():
                return np.nan
            return float(np.nansum(f - self.pi_thresh))

        if which == "pooled":
            if self._pools is None:
                return np.nan
            v = np.array([_auc(*p, idx) for p in self._pools], float)
            v = v[~np.isnan(v)]
            return float(v.mean()) if len(v) else np.nan

        raise ValueError(f"unknown criterion: {which}")


ALL_CRITERIA = ["current", "mean_k", "median_k", "min_k", "mean_sd",
                "mb_perf", "mb_stability", "mb_thresh", "pooled"]
