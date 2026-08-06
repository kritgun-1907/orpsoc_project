#!/usr/bin/env bash
# ==============================================================================
# remote_pipeline.sh — runs ON the EC2 instance. Not meant to be run locally.
#
# Executes the full OrPSOC regeneration end to end, in dependency order, with a
# data-integrity gate in front of the expensive parts.
#
# RESUMABILITY: step7 and step9 checkpoint per (level, seed) into
# results/checkpoints/<stage>/<provenance-hash>/. Re-running this script picks
# up where it stopped rather than starting over, which is what makes a SPOT
# instance viable -- an interruption costs one seed, not the run.
#
# The provenance hash covers the config AND the source of orpsoc_utils.py plus
# the runner, so if you change either, the old checkpoints are correctly ignored
# instead of silently reloaded.
# ==============================================================================
set -euo pipefail

REPO="${REPO:-$HOME/orpsoc_research}"
VENV="${VENV:-$HOME/orpsoc-venv}"
PY="$VENV/bin/python"
LOG_DIR="$REPO/logs"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

# Parallelism. step7 runs 30 seeds per level and step9 runs 20, and seeds are
# the unit of parallelism -- so more workers than seeds buys nothing. Leave two
# cores for the OS.
NPROC="$(nproc)"
: "${ORPSOC_N_JOBS:=$(( NPROC > 32 ? 30 : (NPROC > 2 ? NPROC - 2 : 1) ))}"
export ORPSOC_N_JOBS

mkdir -p "$LOG_DIR"
cd "$REPO"

log()  { printf '\n\033[1m[%s] %s\033[0m\n' "$(date -u +%H:%M:%S)" "$*"; }
fail() { printf '\n\033[31m[%s] FAILED: %s\033[0m\n' "$(date -u +%H:%M:%S)" "$*"; exit 1; }

# Run a stage, tee to its own log, and stop the pipeline if it fails.
stage() {
  local name="$1"; shift
  log "$name"
  if ! "$@" 2>&1 | tee "$LOG_DIR/${STAMP}_${name}.log"; then
    fail "$name (see $LOG_DIR/${STAMP}_${name}.log)"
  fi
  # tee masks the exit status of the left-hand command; PIPESTATUS recovers it.
  [ "${PIPESTATUS[0]}" -eq 0 ] || fail "$name"
}

log "environment"
"$PY" -V 2>/dev/null || python3 -V
echo "  vCPUs=$NPROC   ORPSOC_N_JOBS=$ORPSOC_N_JOBS"
echo "  free memory: $(free -g 2>/dev/null | awk '/^Mem:/{print $7"Gi"}' || echo n/a)"
echo "  disk: $(df -h "$REPO" | awk 'NR==2{print $4" free of "$2}')"

# ── 1. dependencies ───────────────────────────────────────────────────────────
# Ubuntu 24.04 specifics, all three of which will hard-fail a naive setup:
#   * pip is not installed at all
#   * /usr/lib/python3.12/EXTERNALLY-MANAGED (PEP 668) blocks `pip install`
#     into the system interpreter, so a venv is required rather than optional
#   * libgomp is absent, and LightGBM will not even import without it
# Amazon Linux needs none of this, hence the branch on the package manager.
# Key the marker on the CONTENT of requirements.txt. A plain .deps_installed
# flag meant that fixing a missing dependency and re-running would skip the
# install entirely and fail exactly the same way again.
REQ_HASH="$(sha1sum requirements.txt 2>/dev/null | cut -c1-12 || echo nohash)"
DEPS_MARKER="$REPO/.deps_installed_$REQ_HASH"
if [ ! -f "$DEPS_MARKER" ]; then
  log "01_system_packages"
  if command -v apt-get >/dev/null; then
    sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
      python3-venv python3-dev build-essential libgomp1 \
      >"$LOG_DIR/${STAMP}_01_system.log" 2>&1 || fail "apt-get install"
  elif command -v dnf >/dev/null; then
    sudo dnf install -y -q python3-pip python3-devel gcc libgomp \
      >"$LOG_DIR/${STAMP}_01_system.log" 2>&1 || fail "dnf install"
  fi
  ldconfig -p | grep -q libgomp || fail "libgomp still missing -- LightGBM cannot import"

  log "02_venv"
  [ -x "$PY" ] || python3 -m venv "$VENV" || fail "could not create venv at $VENV"
  stage 03_pip  "$PY" -m pip install --quiet --upgrade pip setuptools wheel
  stage 04_reqs "$PY" -m pip install --quiet -r requirements.txt
  "$PY" -m pip install --quiet joblib scipy pypdf || true

  # Fail here rather than 40 minutes into step7.
  stage 05_import_check "$PY" -c "import numpy, pandas, sklearn, lightgbm, joblib, scipy; \
import lightgbm as l; print('  lightgbm', l.__version__, 'imports cleanly')"
  rm -f "$REPO"/.deps_installed_* 2>/dev/null || true
  touch "$DEPS_MARKER"
else
  log "dependencies already installed for this requirements.txt (hash $REQ_HASH)"
fi
[ -x "$PY" ] || fail "venv python missing at $PY"

# ── 2. regenerate every derived dataset from the raw caches ───────────────────
# The raw_*.pkl files are shipped from the laptop rather than re-downloaded:
# yfinance returns a different series on a different day (new bars, occasional
# restatements), so re-downloading here would silently change the datasets and
# break comparability with anything already reported.
for f in raw_sector_prices raw_fama_french raw_bonds_prices raw_commodities_prices; do
  [ -f "data/$f.pkl" ] || fail "missing data/$f.pkl -- the sync did not include the raw caches"
done

stage 03_synth_v1  "$PY" step1_generate_data.py
stage 04_null      "$PY" make_level0_null.py
stage 05_synth_v2  "$PY" make_benchmark_v2.py
stage 06_markets   "$PY" experiments/build_extra_markets.py
# step9 in PREP_ONLY mode rebuilds sector_etf + fama_french from the raw caches.
stage 07_realdata  env ORPSOC_PREP_ONLY=1 "$PY" - <<'PYEOF'
import re
src = open("step9_real_data.py").read()
src = re.sub(r'^PREP_ONLY\s*=.*$', 'PREP_ONLY   = True', src, count=1, flags=re.M)
exec(compile(src, "step9_prep", "exec"), {"__name__": "__main__"})
PYEOF

# ── 3. INTEGRITY GATE — before anything expensive ─────────────────────────────
# A stale dataset silently invalidates everything downstream. This has already
# happened once on this project (fama_french held labels from a superseded
# target definition and agreed with its own base only 47% of the time).
stage 08_verify_data "$PY" experiments/verify_data.py

# ── 4. engine correctness gate ────────────────────────────────────────────────
stage 09_equivalence "$PY" test_equivalence.py

# ── 5. the expensive runs ─────────────────────────────────────────────────────
stage 10_step7_ablation "$PY" step7_ablation.py
stage 11_step8_results  "$PY" step8_results.py
stage 12_step9_realdata "$PY" step9_real_data.py

# ── 6. analyses ───────────────────────────────────────────────────────────────
# Point the analysis at whichever ablation the run actually produced -- step7
# writes results/step7_ablation_v2.json when BENCHMARK_VERSION="v2".
ABLATION_JSON="$(ls -1t results/step7_ablation*.json 2>/dev/null | head -1)"
[ -n "$ABLATION_JSON" ] || fail "no step7 ablation output found"
log "analysing $ABLATION_JSON"
stage 13_jaccard_null   "$PY" orpsoc_jaccard.py "$ABLATION_JSON"
stage 14_adaptation     "$PY" experiments/ff_adaptation.py || true
stage 15_compass        "$PY" experiments/compass_ceiling.py || true

# ── 7. package ────────────────────────────────────────────────────────────────
log "packaging"
OUT="orpsoc_results_${STAMP}.tar.gz"
tar -czf "$HOME/$OUT" \
  -C "$REPO" results plots logs reports 2>/dev/null || \
  tar -czf "$HOME/$OUT" -C "$REPO" results logs
echo "  wrote $HOME/$OUT ($(du -h "$HOME/$OUT" | cut -f1))"

log "PIPELINE COMPLETE"
echo "  retrieve with:  scp <instance>:$HOME/$OUT ."
