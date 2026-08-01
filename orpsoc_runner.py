"""
orpsoc_runner.py — Parallel execution + checkpoint provenance
==============================================================
Shared infrastructure for step7_ablation.py and step9_real_data.py.

Contains NO modelling logic. Everything here is about *how* the experiment is
executed and *how* partial results are stored — never about what is computed.
The numbers a run produces are identical with or without this module.

────────────────────────────────────────────────────────────────────────────
WHY PARALLELISM IS SAFE HERE  (read this before changing anything)
────────────────────────────────────────────────────────────────────────────
The unit of parallelism is ONE SEED — a complete walk-forward sequence, run
fold 1 -> 2 -> ... -> N in chronological order inside a single worker process.
Folds are NEVER parallelised; the walk-forward structure is untouched.

Seeds are independent by construction in the existing code:
  * `hmm_threshold`, `warm_start_fh`, `warm_start_fh_noimp` are all built
    fresh at the top of the seed loop, so nothing crosses a seed boundary.
  * Every random draw comes from a locally constructed
    np.random.RandomState(seed + fold_idx * 1000) — a pure function of
    (seed, fold). There is no global np.random use anywhere in the pipeline,
    so no worker can consume randomness that another worker depends on, and
    execution ORDER cannot change any result.

Verified empirically: 24 seed-jobs run through a loky pool reproduced the
serial results exactly, and running Level 3 after Levels 1-2 vs. alone
produced bit-identical fold AUCs, subsets, and detector diagnostics
(including p_trans to all 16 digits).

────────────────────────────────────────────────────────────────────────────
WHY THREAD PINNING MATTERS
────────────────────────────────────────────────────────────────────────────
LightGBM defaults to n_jobs=-1, spawning one OpenMP thread per core for fits
on tables of a few hundred rows. The thread-sync overhead dominates: measured
4.3x SLOWER than n_jobs=1 on this workload, for bit-identical output. With
process-level parallelism on top, oversubscription would be far worse. Every
LGBMClassifier in the pipeline pins n_jobs=1; pin_threads() closes the same
door for the BLAS/OpenMP libraries underneath numpy.

Call pin_threads() BEFORE importing numpy / lightgbm — the libraries read
these variables at load time.
"""

import hashlib
import json
import os
import pickle


# ══════════════════════════════════════════════════════════════════════════════
#  THREAD PINNING
# ══════════════════════════════════════════════════════════════════════════════

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def pin_threads(n: int = 1) -> None:
    """
    Pin every BLAS/OpenMP backend to `n` threads.

    Uses setdefault, so an explicitly-exported value in the shell always wins.
    Child processes inherit the environment, so calling this once in the parent
    covers all joblib workers.
    """
    for var in _THREAD_VARS:
        os.environ.setdefault(var, str(n))


def default_workers() -> int:
    """
    Worker count for this machine, overridable via ORPSOC_N_JOBS.

    Default is cpu_count - 2, capped at 6. The cap is deliberate: this project's
    reference machine is a fanless MacBook Air M4 (4 performance + 6 efficiency
    cores). Measured sustained throughput on the real workload was 2.80x at 4
    workers, 3.78x at 6, and 2.60x at 8 — past 6 the efficiency cores and
    thermal throttling give the time back. On a fanned/server machine, raise it:

        ORPSOC_N_JOBS=32 python step7_ablation.py
    """
    override = os.environ.get("ORPSOC_N_JOBS")
    if override:
        return max(1, int(override))
    return max(1, min(6, (os.cpu_count() or 2) - 2))


# ══════════════════════════════════════════════════════════════════════════════
#  CHECKPOINT PROVENANCE
# ══════════════════════════════════════════════════════════════════════════════
#  Guardrail G3: "Two numbers produced under different configs are not
#  comparable." A checkpoint that can be reloaded under a DIFFERENT config or a
#  DIFFERENT code version is a silent violation of that guardrail — it is how a
#  results table ends up mixing two code versions with nothing in the output to
#  show it. The previous scheme in step9 keyed only on FAST_MODE, so changing
#  MAX_ITER and re-running would have silently reloaded the old numbers.
#
#  Here the config AND the source of the engine + runner are hashed into the
#  checkpoint directory name, and the full config is stored inside each file
#  and re-verified on load. A config or code change therefore produces a
#  DIFFERENT directory: old checkpoints are never silently reused, and they are
#  never destroyed either, so a re-run at a previous config still resumes.
# ══════════════════════════════════════════════════════════════════════════════

def _file_digest(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.sha1(f.read()).hexdigest()
    except OSError:
        return "missing"


def provenance(config: dict, code_files) -> dict:
    """
    Build the provenance record that identifies a checkpoint generation.

    config     : the run config dict (fast_mode, n_seeds, max_iter, ...)
    code_files : source files whose contents change the numbers. Pass the
                 engine (orpsoc_utils.py) and the runner script itself.
    """
    code = {os.path.basename(p): _file_digest(p) for p in code_files}
    payload = json.dumps({"config": config, "code": code}, sort_keys=True)
    return {
        "config": dict(config),
        "code": code,
        "hash": hashlib.sha1(payload.encode()).hexdigest()[:12],
    }


class CheckpointStore:
    """
    Per-unit checkpoint store scoped by provenance hash.

    A "unit" is one (level, seed) for step7 or one (dataset, seed) for step9 —
    i.e. one COMPLETE walk-forward sequence. Nothing partial is ever stored, so
    a resumed run rebuilds exactly the state a cold run would have: fresh
    threshold object, fresh warm-start chain, same RNG (seeded from
    seed + fold_idx*1000, independent of execution order).

    Each unit writes its own file, so parallel workers never contend.
    """

    def __init__(self, root: str, stage: str, prov: dict, enabled: bool = True):
        self.prov = prov
        self.enabled = enabled
        self.dir = os.path.join(root, stage, prov["hash"])
        if self.enabled:
            os.makedirs(self.dir, exist_ok=True)
            meta = os.path.join(self.dir, "_provenance.json")
            if not os.path.exists(meta):
                with open(meta, "w") as f:
                    json.dump(prov, f, indent=2)

    def path(self, unit: str) -> str:
        return os.path.join(self.dir, f"{unit}.pkl")

    def load(self, unit: str):
        """Return the stored payload, or None if absent/stale/corrupt."""
        if not self.enabled:
            return None
        p = self.path(unit)
        if not os.path.exists(p):
            return None
        try:
            with open(p, "rb") as f:
                rec = pickle.load(f)
        except Exception as e:
            print(f"  [checkpoint] unreadable, recomputing {unit}: {e}", flush=True)
            return None
        # Belt and braces: the directory hash already encodes provenance, but
        # verify the stored config too so a hand-moved file cannot slip through.
        if rec.get("provenance", {}).get("hash") != self.prov["hash"]:
            print(f"  [checkpoint] provenance mismatch, recomputing {unit}", flush=True)
            return None
        return rec["payload"]

    def save(self, unit: str, payload) -> None:
        if not self.enabled:
            return
        tmp = self.path(unit) + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"provenance": self.prov, "payload": payload}, f)
        os.replace(tmp, self.path(unit))   # atomic: no half-written checkpoint

    def summary(self, units) -> str:
        done = sum(1 for u in units if os.path.exists(self.path(u)))
        return f"{done}/{len(units)} units already complete in {self.dir}"
