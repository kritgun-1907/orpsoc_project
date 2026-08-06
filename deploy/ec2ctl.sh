#!/usr/bin/env bash
# ==============================================================================
# ec2ctl.sh — control the OrPSOC compute instance from the terminal.
#
#   ./deploy/ec2ctl.sh status      what state is it in, and what does it cost
#   ./deploy/ec2ctl.sh start       start it and wait until SSH is reachable
#   ./deploy/ec2ctl.sh dns         print the CURRENT public DNS
#   ./deploy/ec2ctl.sh ssh         open a shell on it
#   ./deploy/ec2ctl.sh top         live CPU/memory while a run is going
#   ./deploy/ec2ctl.sh tail        follow the pipeline log
#   ./deploy/ec2ctl.sh stop        stop it (keeps the disk, stops the hourly charge)
#   ./deploy/ec2ctl.sh terminate   destroy it and its disk  [asks first]
#
# The instance has NO Elastic IP, so its public DNS changes on every
# stop/start. Never cache the hostname -- every subcommand re-resolves it.
# ==============================================================================
set -euo pipefail

INSTANCE_ID="${ORPSOC_INSTANCE_ID:-i-02415f8ffa4e0da4d}"
REGION="${AWS_REGION:-us-east-1}"
KEY="${ORPSOC_SSH_KEY:-$HOME/.ssh/orpsoc.pem}"
USER_NAME="${EC2_USER:-}"
REMOTE_DIR="orpsoc_research"

command -v aws >/dev/null || { echo "aws cli not installed" >&2; exit 1; }

q() { aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" \
        --query "$1" --output text 2>/dev/null; }

state() { q 'Reservations[].Instances[].State.Name'; }
dns()   { q 'Reservations[].Instances[].PublicDnsName'; }

need_running() {
  local s; s="$(state)"
  [ "$s" = "running" ] || { echo "instance is '$s' -- run: $0 start" >&2; exit 1; }
}


# The correct SSH user depends on the AMI: Ubuntu images use `ubuntu`, Amazon
# Linux uses `ec2-user`, Debian `admin`. Guessing wrong yields a bare
# "Permission denied (publickey)" that looks identical to a bad key, so probe
# instead of assuming. Override with EC2_USER to skip the probe.
detect_user() {
  local host="$1" u
  if [ -n "${USER_NAME:-}" ]; then echo "$USER_NAME"; return; fi
  for u in ubuntu ec2-user admin fedora centos root; do
    if ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o BatchMode=yes \
           -o ConnectTimeout=8 "$u@$host" 'exit 0' 2>/dev/null; then
      echo "$u"; return
    fi
  done
  echo "could not authenticate as any standard AMI user -- check the key" >&2
  exit 1
}

ssh_to() {
  need_running
  local host; host="$(dns)"
  [ -n "$host" ] && [ "$host" != "None" ] || { echo "no public DNS yet" >&2; exit 1; }
  [ -f "$KEY" ] || { echo "ssh key not found: $KEY  (set ORPSOC_SSH_KEY)" >&2; exit 1; }
  local u; u="$(detect_user "$host")"
  ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 \
      "${u}@${host}" "$@"
}

case "${1:-status}" in

  status)
    s="$(state)"
    echo "instance : $INSTANCE_ID  ($REGION)"
    echo "type     : $(q 'Reservations[].Instances[].InstanceType')"
    echo "state    : $s"
    if [ "$s" = "running" ]; then
      echo "public   : $(dns)"
      echo "since    : $(q 'Reservations[].Instances[].LaunchTime')"
      echo
      echo "BILLING: a running instance is charged by the second. Stop it when idle:"
      echo "  $0 stop"
    else
      echo
      echo "stopped instances cost only EBS storage (cents/month), not compute."
    fi
    ;;

  start)
    s="$(state)"
    if [ "$s" = "running" ]; then echo "already running: $(dns)"; exit 0; fi
    echo "starting $INSTANCE_ID ..."
    aws ec2 start-instances --region "$REGION" --instance-ids "$INSTANCE_ID" >/dev/null
    aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"
    echo "running. waiting for status checks (SSH is not up until these pass) ..."
    aws ec2 wait instance-status-ok --region "$REGION" --instance-ids "$INSTANCE_ID"
    echo "ready: $(dns)"
    echo
    echo "NOTE: the public DNS above is NEW -- this instance has no Elastic IP."
    ;;

  stop)
    aws ec2 stop-instances --region "$REGION" --instance-ids "$INSTANCE_ID" >/dev/null
    echo "stopping. compute charges end once it reaches 'stopped'."
    aws ec2 wait instance-stopped --region "$REGION" --instance-ids "$INSTANCE_ID" \
      && echo "stopped."
    ;;

  terminate)
    echo "TERMINATE destroys $INSTANCE_ID and its root volume. This cannot be undone."
    echo "If you only want to stop paying for compute, use:  $0 stop"
    printf 'type the instance id to confirm: '
    read -r ans
    [ "$ans" = "$INSTANCE_ID" ] || { echo "aborted."; exit 1; }
    aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID" >/dev/null
    echo "terminating."
    ;;

  dns)  dns ;;
  ssh)  shift; ssh_to "$@" ;;

  top)
    ssh_to "top -b -n1 | head -20; echo; echo 'orpsoc python processes:';
            pgrep -fl 'step7|step9|python3' | head"
    ;;

  tail)
    echo "following pipeline log (Ctrl-C detaches; the run keeps going)"
    ssh_to "tail -f $REMOTE_DIR/pipeline.out"
    ;;

  progress)
    ssh_to "cd $REMOTE_DIR && \
      echo 'completed checkpoints per stage:' && \
      find results/checkpoints -name '*.pkl' 2>/dev/null | \
        sed 's|.*/checkpoints/||; s|/[^/]*\$||' | sort | uniq -c && \
      echo && tail -5 pipeline.out"
    ;;

  *)
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
    exit 1 ;;
esac
