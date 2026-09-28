#!/usr/bin/env bash
# Benchmark client VM in the cluster's region (AWS eu-west-1).
#   ./vm.sh up       import SSH key, create SG (22 from this IP only), launch c7i.large AL2023
#   ./vm.sh push     copy the harness + .env, install pinned deps, record TCP floor
#   ./vm.sh run      phase1 -> phase2 -> phase3 -> report in the background (log: ~/ta/run.log);
#                    RUN_CMD='...' runs a custom command instead
#   ./vm.sh status   tail the remote log
#   ./vm.sh wait     block until the run writes RUN_EXIT=<code>
#   ./vm.sh ssh      interactive shell
#   ./vm.sh fetch    copy results/ back and VERIFY the phase-3 files arrived
#   ./vm.sh down     terminate + delete SG and key pair (refuses unless fetch verified; FORCE=1 overrides)
# Extra args for the run: PHASE2_ARGS="--rows 10000" PHASE3_ARGS="--duration 60" ./vm.sh run
set -euo pipefail

REGION="${REGION:-eu-west-1}"
AZ="${AZ:-eu-west-1a}"
TYPE="${TYPE:-c7i.large}"
NAME="${NAME:-ta-bench}"
KEY="${KEY:-$HOME/.ssh/id_ed25519}"
DIR="$(cd "$(dirname "$0")" && pwd)"
STATE="$DIR/.vm_state"
export AWS_REGION="$REGION" AWS_PAGER=""

load() { if [ -f "$STATE" ]; then . "$STATE"; fi; }
ssh_() { ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$DIR/.vm_known_hosts" "ec2-user@$IP" "$@"; }
scp_() { scp -i "$KEY" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$DIR/.vm_known_hosts" "$@"; }

case "${1:-}" in
up)
  load; if [ -n "${INSTANCE_ID:-}" ]; then echo "already up: $INSTANCE_ID $IP"; exit 0; fi
  MYIP=$(curl -s https://checkip.amazonaws.com)
  KP="$NAME-$(date +%Y%m%d%H%M)"
  aws ec2 import-key-pair --key-name "$KP" --public-key-material "fileb://$KEY.pub" >/dev/null
  VPC=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
  SUBNET=$(aws ec2 describe-subnets --filters Name=vpc-id,Values="$VPC" Name=availability-zone,Values="$AZ" \
           --query 'Subnets[0].SubnetId' --output text)
  SG=$(aws ec2 create-security-group --group-name "$KP" --description "typeahead bench SSH" --vpc-id "$VPC" \
       --query GroupId --output text)
  aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 --cidr "$MYIP/32" >/dev/null
  AMI=$(aws ssm get-parameter --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
        --query Parameter.Value --output text)
  INSTANCE_ID=$(aws ec2 run-instances --image-id "$AMI" --instance-type "$TYPE" --key-name "$KP" \
       --security-group-ids "$SG" --subnet-id "$SUBNET" \
       --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=20,VolumeType=gp3}' \
       --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$KP}]" \
       --query 'Instances[0].InstanceId' --output text)
  printf 'INSTANCE_ID=%s\nSG=%s\nKP=%s\n' "$INSTANCE_ID" "$SG" "$KP" > "$STATE"
  aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
  IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
       --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
  echo "IP=$IP" >> "$STATE"
  echo "waiting for SSH on $IP ..."
  until ssh_ -o ConnectTimeout=5 true 2>/dev/null; do sleep 5; done
  echo "up: $INSTANCE_ID $TYPE $AZ $IP (SSH from $MYIP only)"
  ;;
push)
  load
  COPYFILE_DISABLE=1 tar -C "$DIR" --exclude .venv --exclude results --exclude .env --exclude '.vm_*' \
    --exclude __pycache__ -czf /tmp/ta_bench.tgz .
  scp_ /tmp/ta_bench.tgz "ec2-user@$IP:~/ta_bench.tgz"
  ENVF="$DIR/.env"; [ -f "$ENVF" ] || ENVF="$DIR/../.env"
  scp_ "$ENVF" "ec2-user@$IP:~/ta.env"
  ssh_ 'set -euo pipefail
    mkdir -p ~/ta && tar -xzf ~/ta_bench.tgz -C ~/ta && mv ~/ta.env ~/ta/.env && chmod 600 ~/ta/.env
    sudo dnf install -y -q tmux >/dev/null 2>&1 || true
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
    export PATH="$HOME/.local/bin:$PATH"
    cd ~/ta && uv venv -q --python 3.12 .venv && uv pip install -q --python .venv/bin/python -r requirements.txt
    .venv/bin/python -c "import config; print(\"TCP connect median ms:\", round(config.tcp_connect_ms(20), 2))"'
  ;;
run)
  load
  # RUN_CMD overrides the default pipeline. RUN_EXIT=<code> is always the last log line, so a
  # watcher can wait on it without guessing at process names.
  CMD="${RUN_CMD:-.venv/bin/python phase1_capability.py && .venv/bin/python phase2_load.py ${PHASE2_ARGS:-} && .venv/bin/python phase3_bench.py ${PHASE3_ARGS:-} && .venv/bin/python report.py}"
  printf 'export PYTHONUNBUFFERED=1 CLUSTER_DB_VERSION=%s\n%s\necho RUN_EXIT=$?\n' "${CLUSTER_DB_VERSION:-}" "$CMD" > /tmp/ta_run.sh
  scp_ /tmp/ta_run.sh "ec2-user@$IP:~/ta/run.sh"
  ssh_ "cd ~/ta && nohup bash run.sh > run.log 2>&1 < /dev/null & disown" < /dev/null
  echo "started; ./vm.sh status to follow, ./vm.sh wait to block until RUN_EXIT"
  ;;
wait)
  load
  until code=$(ssh_ 'grep -o "RUN_EXIT=[0-9]*" ~/ta/run.log' 2>/dev/null); do sleep "${POLL:-30}"; done
  echo "$code"; ssh_ 'grep -v ev_poll ~/ta/run.log | tail -n 30'
  ;;
status) load; ssh_ 'grep -v ev_poll ~/ta/run.log | tail -n ${N:-25}' ;;
ssh) load; ssh_ ;;
fetch)
  load
  mkdir -p "$DIR/results"
  ssh_ 'cd ~/ta && tar -czf /tmp/results.tgz results run.log'
  scp_ "ec2-user@$IP:/tmp/results.tgz" /tmp/ta_results.tgz
  tar -xzf /tmp/ta_results.tgz -C "$DIR/results" --strip-components 1 results
  tar -xzf /tmp/ta_results.tgz -O run.log > "$DIR/results/vm_run.log"
  # Verify every remote phase-3 run landed locally with its summary + requests before allowing teardown.
  missing=0
  for d in $(ssh_ 'ls -d ~/ta/results/*/ 2>/dev/null | xargs -n1 basename'); do
    [ -d "$DIR/results/$d" ] || { echo "MISSING $d"; missing=1; }
  done
  for f in $(ssh_ 'cd ~/ta/results && ls */phase3_summary.csv */phase3_requests.csv */report.md 2>/dev/null'); do
    [ -s "$DIR/results/$f" ] || { echo "MISSING $f"; missing=1; }
  done
  if [ "$missing" = 0 ]; then date -u > "$DIR/.vm_fetched"; echo "fetch verified -> results/"; else exit 1; fi
  ;;
down)
  load
  if [ ! -f "$DIR/.vm_fetched" ] && [ "${FORCE:-0}" != 1 ]; then
    echo "refusing: results not fetched+verified (./vm.sh fetch), or FORCE=1"; exit 1
  fi
  aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" >/dev/null
  aws ec2 wait instance-terminated --instance-ids "$INSTANCE_ID"
  aws ec2 delete-security-group --group-id "$SG"
  aws ec2 delete-key-pair --key-name "$KP"
  rm -f "$STATE" "$DIR/.vm_fetched" "$DIR/.vm_known_hosts"
  echo "terminated $INSTANCE_ID, deleted $SG and $KP"
  ;;
*) sed -n 2,11p "$0"; exit 1 ;;
esac
