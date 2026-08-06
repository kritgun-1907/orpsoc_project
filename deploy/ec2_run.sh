#!/usr/bin/env bash
# ==============================================================================
# ec2_run.sh — drive the full OrPSOC regeneration on an EC2 instance.
#
# Runs FROM YOUR LAPTOP. Syncs the repo up, launches the pipeline under a
# detached session so it survives SSH drops, streams the log, pulls results back.
#
#   ./deploy/ec2_run.sh --host ec2-1-2-3-4.compute-1.amazonaws.com --key ~/.ssh/k.pem
#   ./deploy/ec2_run.sh --instance-id i-0abc123 --key ~/.ssh/k.pem
#   ./deploy/ec2_run.sh --host <dns> --key <pem> --sync-only     # push, don't run
#   ./deploy/ec2_run.sh --host <dns> --key <pem> --fetch-only    # pull results
#   ./deploy/ec2_run.sh --host <dns> --key <pem> --attach        # re-attach to a run
#
# DELIBERATELY DOES NOT LAUNCH OR TERMINATE INSTANCES. Provisioning and
# destroying cloud resources costs money and is hard to undo; this script only
# talks to a box you already control. Create and stop the instance yourself, or
# via the helper commands printed by --help.
# ==============================================================================
set -euo pipefail

HOST=""; KEY=""; INSTANCE_ID=""; USER_NAME="${EC2_USER:-ec2-user}"
REMOTE_DIR="orpsoc_research"
MODE="full"
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"

usage() {
  sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Options
  --host <dns|ip>      public DNS or IP of a running instance
  --instance-id <id>   resolve the DNS from an instance id (needs aws cli)
  --key <path.pem>     SSH private key
  --user <name>        SSH user (default ec2-user; use ubuntu on Ubuntu AMIs)
  --sync-only          push code+data, then stop
  --fetch-only         pull results, then stop
  --attach             tail an already-running pipeline
  --region <region>    AWS region for --instance-id lookup

Sizing note
  step7 runs 30 seeds per level and step9 runs 20; seeds are the unit of
  parallelism, so more than ~30 vCPUs buys nothing. A 32-vCPU compute-optimised
  instance (c7i.8xlarge / c6i.8xlarge) is the sweet spot. Memory is not the
  constraint -- roughly 400 MB per worker, so 64 GB is ample. Disk needs ~10 GB;
  the repo plus all results is well under 2 GB.

Spot instances are a good fit: step7/step9 checkpoint per (level, seed), so an
interruption costs one seed and re-running resumes.

Useful AWS commands (run them yourself, they cost money):
  aws ec2 describe-instances --instance-ids <id> \
      --query 'Reservations[].Instances[].PublicDnsName' --output text
  aws ec2 stop-instances      --instance-ids <id>
  aws ec2 terminate-instances --instance-ids <id>
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --host)        HOST="$2"; shift 2 ;;
    --instance-id) INSTANCE_ID="$2"; shift 2 ;;
    --key)         KEY="$2"; shift 2 ;;
    --user)        USER_NAME="$2"; shift 2 ;;
    --region)      REGION="$2"; shift 2 ;;
    --sync-only)   MODE="sync";  shift ;;
    --fetch-only)  MODE="fetch"; shift ;;
    --attach)      MODE="attach"; shift ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

# ── preflight ─────────────────────────────────────────────────────────────────
command -v rsync >/dev/null || { echo "rsync not found" >&2; exit 1; }
command -v ssh   >/dev/null || { echo "ssh not found" >&2; exit 1; }

if [ -n "$INSTANCE_ID" ] && [ -z "$HOST" ]; then
  command -v aws >/dev/null || { echo "aws cli needed for --instance-id" >&2; exit 1; }
  HOST="$(aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" \
          --query 'Reservations[].Instances[].PublicDnsName' --output text)"
  [ -n "$HOST" ] && [ "$HOST" != "None" ] || { echo "no public DNS for $INSTANCE_ID" >&2; exit 1; }
  echo "resolved $INSTANCE_ID -> $HOST"
fi

[ -n "$HOST" ] || { echo "need --host or --instance-id" >&2; usage; exit 1; }
[ -n "$KEY"  ] || { echo "need --key" >&2; usage; exit 1; }
[ -f "$KEY"  ] || { echo "key not found: $KEY" >&2; exit 1; }
chmod 600 "$KEY" 2>/dev/null || true

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)
TARGET="${USER_NAME}@${HOST}"

echo "instance : $TARGET"
echo "repo     : $REPO_ROOT"
"${SSH[@]}" "$TARGET" 'echo "  reachable: $(uname -srm), $(nproc) vCPUs"' \
  || { echo "cannot reach $TARGET -- check security group allows SSH from your IP" >&2; exit 1; }

# ── sync ──────────────────────────────────────────────────────────────────────
# data/*.pkl is gitignored, so this cannot be a git clone. The raw_*.pkl caches
# MUST travel: re-downloading on the instance would fetch a different series
# (yfinance adds bars daily) and silently change every dataset.
sync_up() {
  echo
  echo "syncing repo -> instance"
  rsync -az --info=stats1 -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new" \
    --exclude '.git/' --exclude '__pycache__/' --exclude '*.pyc' \
    --exclude 'logs/' --exclude 'results/checkpoints/' \
    --exclude 'data/checkpoint_*' --exclude '*STALE*' \
    "$REPO_ROOT/" "$TARGET:$REMOTE_DIR/"
  "${SSH[@]}" "$TARGET" "chmod +x $REMOTE_DIR/deploy/*.sh"
  echo "raw data caches present on instance:"
  "${SSH[@]}" "$TARGET" "ls -la $REMOTE_DIR/data/raw_*.pkl | awk '{print \"  \"\$9, \$5}'"
}

fetch_down() {
  echo
  echo "fetching results <- instance"
  mkdir -p "$REPO_ROOT/ec2_results"
  # Checkpoints stay on the instance: they are large, regenerable, and only
  # useful there -- they exist so an interrupted run can resume in place.
  rsync -az --info=stats1 --exclude 'checkpoints/' \
    -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new" \
    "$TARGET:$REMOTE_DIR/results/"  "$REPO_ROOT/ec2_results/results/" || true
  rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new" \
    "$TARGET:$REMOTE_DIR/plots/"    "$REPO_ROOT/ec2_results/plots/"   || true
  rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new" \
    "$TARGET:$REMOTE_DIR/logs/"     "$REPO_ROOT/ec2_results/logs/"    || true
  echo "  -> $REPO_ROOT/ec2_results/"
  echo "  NOTE: kept separate from results/ so an EC2 run never silently"
  echo "        overwrites local numbers. Diff before promoting."
}

case "$MODE" in
  sync)  sync_up; exit 0 ;;
  fetch) fetch_down; exit 0 ;;
  attach)
    echo; echo "attaching (Ctrl-C detaches, the run keeps going)"
    "${SSH[@]}" "$TARGET" "tail -f $REMOTE_DIR/pipeline.out"
    exit 0 ;;
esac

# ── full run ──────────────────────────────────────────────────────────────────
sync_up

echo
echo "starting pipeline under setsid (survives SSH disconnect)"
"${SSH[@]}" "$TARGET" bash -lc "
  cd $REMOTE_DIR &&
  if pgrep -f remote_pipeline.sh >/dev/null; then
    echo 'a pipeline is already running on this instance -- use --attach'; exit 1
  fi &&
  setsid nohup ./deploy/remote_pipeline.sh > pipeline.out 2>&1 < /dev/null &
  sleep 2; echo 'started'
"

echo
echo "streaming log -- Ctrl-C detaches WITHOUT stopping the run"
echo "re-attach later:  $0 --host $HOST --key $KEY --attach"
echo
trap 'echo; echo "detached. run continues on the instance."; exit 0' INT
"${SSH[@]}" "$TARGET" "tail -f $REMOTE_DIR/pipeline.out" || true

if "${SSH[@]}" "$TARGET" "grep -q 'PIPELINE COMPLETE' $REMOTE_DIR/pipeline.out"; then
  fetch_down
  echo
  echo "done. The instance is STILL RUNNING and still costing money."
  echo "stop it:      aws ec2 stop-instances      --instance-ids <id>"
  echo "terminate it: aws ec2 terminate-instances --instance-ids <id>"
else
  echo "pipeline did not report completion -- inspect with --attach before fetching."
fi
