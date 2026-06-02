#!/usr/bin/env bash
# Deploy + run the factor-investment data pipeline on the Synology NAS.
# Mirrors the court-reserve deploy convention (laptop-side, sudo docker).
#
# Prerequisites (one-time NAS setup):
#   1. SSH public key:        ssh-copy-id hsheng@192.168.68.70
#   2. GitHub access:         NAS can auth to git@github.com (key present)
#   3. Passwordless docker:   /etc/sudoers.d/docker-hsheng  (already configured)
#   4. Data shared folder:    /volume1/factor-data  (created in DSM)
#   5. Initial clone:         handled automatically on first run below
#
# Usage (from the laptop):
#   ./deploy/deploy.sh                 # full backfill (~45-90 min, detached)
#   ./deploy/deploy.sh incremental     # cheap scheduled-style update
#   ./deploy/deploy.sh status          # last 40 log lines
#   ./deploy/deploy.sh tail            # follow the log live
set -euo pipefail

NAS_HOST="${NAS_HOST:-hsheng@192.168.68.70}"
NAS_DIR="${NAS_DIR:-/volume1/docker/factor-investment}"
DATA_DIR="${DATA_DIR:-/volume1/factor-data}"     # compose default; see docker-compose.yml
REPO_URL="${REPO_URL:-git@github.com:hs220/factor-investment.git}"
BRANCH="${BRANCH:-master}"
DOCKER="sudo /usr/local/bin/docker"
LOG="$DATA_DIR/backfill.log"
MODE="${1:-backfill}"

case "$MODE" in
  tail)   exec ssh "$NAS_HOST" "tail -f '$LOG'" ;;
  status) exec ssh "$NAS_HOST" "tail -n 40 '$LOG'" ;;
esac

SVC="build-dataset"
[ "$MODE" = "incremental" ] && SVC="incremental"

echo "==> Sync code on NAS ($NAS_DIR @ $BRANCH)..."
ssh "$NAS_HOST" "
  set -e
  if [ -d '$NAS_DIR/.git' ]; then
    git -C '$NAS_DIR' fetch origin '$BRANCH'
    git -C '$NAS_DIR' checkout '$BRANCH'
    git -C '$NAS_DIR' pull --ff-only
  else
    git clone --branch '$BRANCH' '$REPO_URL' '$NAS_DIR'
  fi
  mkdir -p '$DATA_DIR/raw' '$DATA_DIR/processed'
"

echo "==> Build image (service: $SVC)..."
ssh "$NAS_HOST" "cd '$NAS_DIR' && $DOCKER compose build $SVC"

echo "==> Launch '$SVC' detached (log: $LOG)..."
ssh "$NAS_HOST" "cd '$NAS_DIR' && \
  nohup $DOCKER compose run --rm $SVC > '$LOG' 2>&1 </dev/null & \
  echo 'launched, host PID '\$!"

echo "==> Started. Follow with:  ./deploy/deploy.sh tail"
