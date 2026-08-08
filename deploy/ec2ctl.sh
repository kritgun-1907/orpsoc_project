#!/usr/bin/env bash
# ==============================================================================
# ec2ctl.sh — control the OrPSOC compute instance from the terminal.
#
#   ./deploy/ec2ctl.sh status      what state is it in, and what does it cost
#   ./deploy/ec2ctl.sh start       start it and wait until SSH is reachable
#   ./deploy/ec2ctl.sh dns         print the CURRENT public DNS
#   ./deploy/ec2ctl.sh ssh         open a shell on it
#   ./deploy/ec2ctl.sh allowip     re-open port 22 after your ISP rotates your IP
#   ./deploy/ec2ctl.sh progress    checkpoints completed + last log lines
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
#
# Three failures look alike from the outside and must NOT be conflated:
#   TCP timeout        -> the security group does not allow your current IP
#   Permission denied  -> wrong key, or the right key for a different AMI user
#   Connection refused -> sshd is not up yet (instance still booting)
# This used to report "check the key" for all three, which sent us hunting for a
# key problem three separate times when the real cause was an ISP IP rotation.
# So keep ssh's own stderr and classify on it.
detect_user() {
  local host="$1" u out rc last=""
  if [ -n "${USER_NAME:-}" ]; then echo "$USER_NAME"; return; fi
  for u in ubuntu ec2-user admin fedora centos root; do
    out="$(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o BatchMode=yes \
             -o ConnectTimeout=10 "$u@$host" 'exit 0' 2>&1)"; rc=$?
    [ $rc -eq 0 ] && { echo "$u"; return; }
    last="$out"
    # A timeout is a property of the network path, not of the username -- once
    # one user times out the rest will too, so stop rather than burn 6x10s.
    case "$last" in *"timed out"*|*"Operation timed out"*) break ;; esac
  done

  case "$last" in
    *"timed out"*)
      echo "cannot reach $host:22 -- TCP timeout, NOT an authentication failure." >&2
      echo >&2
      echo "A security group drops non-matching packets silently, so a blocked" >&2
      echo "source IP looks exactly like a hung host. Your ISP most likely" >&2
      echo "rotated your address. Fix with:" >&2
      echo "    $0 allowip" >&2
      ;;
    *"Permission denied"*)
      echo "reached $host:22 but every standard AMI user was rejected." >&2
      echo "This one really is credentials: check \$ORPSOC_SSH_KEY (now: $KEY)." >&2
      ;;
    *"Connection refused"*)
      # A RST can come from the far end (sshd down) or from a middlebox on YOUR
      # network. Many campus/corporate networks refuse all outbound port 22.
      # Probe a third party that certainly runs sshd to tell the two apart --
      # otherwise this reports "sshd is not up yet" about a perfectly healthy
      # instance, which is what it did on the NKN academic network.
      if ! (ssh -o BatchMode=yes -o StrictHostKeyChecking=no \
                -o ConnectTimeout=8 git@github.com 'exit' 2>&1 \
            | grep -qv "Connection refused"); then
        echo "port 22 is blocked by YOUR network, not by the instance." >&2
        echo >&2
        echo "github.com:22 is refused too, and GitHub certainly runs sshd --" >&2
        echo "so the RST comes from a middlebox on this network. Campus and" >&2
        echo "corporate networks routinely block outbound SSH." >&2
        echo >&2
        echo "Options:  switch to a mobile hotspot, then '$0 allowip'" >&2
        echo "          or monitor without SSH -- these use HTTPS and still work:" >&2
        echo "            $0 status" >&2
      else
        echo "$host refused port 22 -- sshd is not up yet. Wait for the status" >&2
        echo "checks to pass:  $0 start" >&2
      fi
      ;;
    *)
      echo "could not connect to $host:22" >&2
      echo "  ssh said: $last" >&2
      ;;
  esac
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

  allowip)
    # The SSH rule is pinned to a single /32. Consumer ISPs rotate the customer
    # address (mobile broadband especially), and when yours changes the symptom
    # is a CONNECTION TIMEOUT, not "permission denied" -- a security group drops
    # non-matching SYNs silently instead of sending a RST. That looks exactly
    # like a hung or dead instance, so check this before debugging the run.
    SG="$(q 'Reservations[].Instances[].SecurityGroups[].GroupId' | awk '{print $1}')"
    [ -n "$SG" ] && [ "$SG" != "None" ] || { echo "could not resolve security group" >&2; exit 1; }
    MYIP="$(curl -s --max-time 10 https://checkip.amazonaws.com | tr -d '[:space:]')"
    case "$MYIP" in
      *.*.*.*) ;;
      *) echo "could not determine public IP (got '$MYIP')" >&2; exit 1 ;;
    esac
    echo "security group : $SG"
    echo "your public IP : $MYIP"

    # NOTE the [0] rather than []. `SecurityGroups[].IpPermissions[?...]` opens a
    # projection, and the subsequent .IpRanges[] flatten inside it silently
    # yields NOTHING -- no error, just an empty result. That made this command
    # report no existing rules at all on its first outing, so the stale-CIDR
    # prune below never fired. We always query exactly one group, so index it.
    SSH_CIDRS_Q='SecurityGroups[0].IpPermissions[?FromPort==`22`].IpRanges[].CidrIp'

    if aws ec2 describe-security-groups --region "$REGION" --group-ids "$SG" \
         --query "$SSH_CIDRS_Q" \
         --output text | tr '\t' '\n' | grep -qx "$MYIP/32"; then
      echo "already allowed -- nothing to do."
    else
      aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG" \
        --protocol tcp --port 22 --cidr "$MYIP/32" >/dev/null
      echo "added $MYIP/32 -> port 22"
    fi

    # Old /32s are dead weight: each is a standing grant to an address that has
    # since been recycled to some other ISP customer. Prune them.
    stale="$(aws ec2 describe-security-groups --region "$REGION" --group-ids "$SG" \
      --query "$SSH_CIDRS_Q" \
      --output text | tr '\t' '\n' | grep -v "^$MYIP/32\$" | grep '/32$' || true)"
    if [ -n "$stale" ]; then
      echo "stale single-host rules still present:"
      echo "$stale" | sed 's/^/  /'
      printf 'revoke them? [y/N] '
      read -r ans
      if [ "$ans" = "y" ] || [ "$ans" = "Y" ]; then
        echo "$stale" | while read -r c; do
          [ -n "$c" ] || continue
          aws ec2 revoke-security-group-ingress --region "$REGION" --group-id "$SG" \
            --protocol tcp --port 22 --cidr "$c" >/dev/null && echo "  revoked $c"
        done
      fi
    fi
    ;;

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
