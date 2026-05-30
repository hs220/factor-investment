#!/usr/bin/env bash
# Stand up the TimescaleDB warehouse on the Synology NAS (persistent service).
# Mirrors the court-reserve deploy convention (laptop-side, sudo docker).
#
# Usage (from the laptop):
#   ./deploy/deploy_timescale.sh up       # create .env (if missing) + start DB
#   ./deploy/deploy_timescale.sh status   # container + health
#   ./deploy/deploy_timescale.sh logs
#   ./deploy/deploy_timescale.sh down
#
# Config via env:
#   NAS_HOST   ssh target  (default hsheng@192.168.68.70)
#   NAS_DIR    code dir     (default /volume1/docker/factor-investment)
#   BRANCH     git branch   (default strategy-b-pipeline)
set -euo pipefail

NAS_HOST="${NAS_HOST:-hsheng@192.168.68.70}"
NAS_DIR="${NAS_DIR:-/volume1/docker/factor-investment}"
BRANCH="${BRANCH:-strategy-b-pipeline}"
REPO_URL="${REPO_URL:-git@github.com:hs220/factor-investment.git}"
DOCKER="sudo /usr/local/bin/docker"
COMPOSE_DIR="$NAS_DIR/deploy/timescale"
MODE="${1:-up}"

remote() { ssh "$NAS_HOST" "$1"; }

case "$MODE" in
  status) exec ssh "$NAS_HOST" "cd '$COMPOSE_DIR' && $DOCKER compose ps" ;;
  logs)   exec ssh "$NAS_HOST" "cd '$COMPOSE_DIR' && $DOCKER compose logs --tail=50" ;;
  down)   exec ssh "$NAS_HOST" "cd '$COMPOSE_DIR' && $DOCKER compose down" ;;
esac

echo "==> Sync code on NAS ($NAS_DIR @ $BRANCH)..."
remote "
  set -e
  if [ -d '$NAS_DIR/.git' ]; then
    git -C '$NAS_DIR' fetch origin '$BRANCH' && git -C '$NAS_DIR' checkout '$BRANCH' && git -C '$NAS_DIR' pull --ff-only
  else
    git clone --branch '$BRANCH' '$REPO_URL' '$NAS_DIR'
  fi
"

echo "==> Ensure .env with a generated password (if missing)..."
remote "
  set -e
  cd '$COMPOSE_DIR'
  if [ ! -f .env ]; then
    pw=\$(tr -dc 'A-Za-z0-9' </dev/urandom | head -c 32)
    cp .env.example .env
    sed -i \"s/^POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=\$pw/\" .env
    echo 'generated new .env (password set)'
  else
    echo '.env already present; leaving as-is'
  fi
"

echo "==> Start TimescaleDB..."
remote "cd '$COMPOSE_DIR' && $DOCKER compose up -d"

echo "==> Health (wait a few seconds for init)..."
remote "cd '$COMPOSE_DIR' && $DOCKER compose ps"

cat <<EOF

==> TimescaleDB is starting on $NAS_HOST:5432 (db 'factor', user 'factor').
    Retrieve the password for laptop access:
      ssh $NAS_HOST "grep POSTGRES_PASSWORD $COMPOSE_DIR/.env"
    Then on the laptop:
      export POSTGRES_PASSWORD=<that value>
      python -c "from src.data import db; print('db reachable:', db.ping())"
EOF
