#!/usr/bin/env bash
# Deploy Dagster (webserver + daemon) on the Synology NAS.
# Mirrors the court-reserve deploy convention (laptop-side, sudo docker).
#
# Usage (from the laptop):
#   ./deploy/deploy_dagster.sh up        # create dagster DB, build, start
#   ./deploy/deploy_dagster.sh status
#   ./deploy/deploy_dagster.sh logs
#   ./deploy/deploy_dagster.sh down
#
# UI: http://192.168.68.70:3030
set -euo pipefail

NAS_HOST="${NAS_HOST:-hsheng@192.168.68.70}"
NAS_DIR="${NAS_DIR:-/volume1/docker/factor-investment}"
BRANCH="${BRANCH:-master}"
REPO_URL="${REPO_URL:-git@github.com:hs220/factor-investment.git}"
DOCKER="sudo /usr/local/bin/docker"
TS_DIR="$NAS_DIR/deploy/timescale"
DG_DIR="$NAS_DIR/deploy/dagster"
MODE="${1:-up}"

remote() { ssh "$NAS_HOST" "$1"; }

case "$MODE" in
  status) exec ssh "$NAS_HOST" "cd '$DG_DIR' && $DOCKER compose ps" ;;
  logs)   exec ssh "$NAS_HOST" "cd '$DG_DIR' && $DOCKER compose logs --tail=60" ;;
  down)   exec ssh "$NAS_HOST" "cd '$DG_DIR' && $DOCKER compose down" ;;
  materialize)  # headless materialize of $2 (default: fundamentals + sectors)
    sel="${2:-fundamental_facts,sectors}"
    exec ssh "$NAS_HOST" "$DOCKER exec factor-dagster-daemon \
      dagster asset materialize --select '$sel' -m orchestration.definitions"
    ;;
  materialize-fundamentals)
    exec ssh "$NAS_HOST" "$DOCKER exec factor-dagster-daemon \
      dagster asset materialize --select 'fundamental_facts,sectors' -m orchestration.definitions"
    ;;
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

echo "==> Build dagster .env from the Timescale password..."
remote "
  set -e
  pw=\$(grep '^POSTGRES_PASSWORD=' '$TS_DIR/.env' | cut -d= -f2)
  cat > '$DG_DIR/.env' <<EOF
POSTGRES_PASSWORD=\$pw
DAGSTER_PG_HOST=192.168.68.70
FACTOR_DB_HOST=192.168.68.70
FACTOR_DB_PORT=5433
DAGSTER_PORT=3030
EOF
  echo 'wrote $DG_DIR/.env'
"

echo "==> Ensure 'dagster' database exists (run-history storage)..."
remote "
  $DOCKER exec factor-timescale psql -U factor -d factor -tc \"SELECT 1 FROM pg_database WHERE datname='dagster'\" | grep -q 1 \
    || $DOCKER exec factor-timescale psql -U factor -d factor -c 'CREATE DATABASE dagster'
  echo 'dagster DB ready'
"

echo "==> Build image and start webserver + daemon..."
remote "cd '$DG_DIR' && $DOCKER compose build && $DOCKER compose up -d"

echo "==> Status..."
remote "cd '$DG_DIR' && $DOCKER compose ps"

cat <<EOF

==> Dagster UI:  http://192.168.68.70:3030
    Materialize fundamentals (the restatement-aware re-fetch) from the UI,
    or headless:
      ./deploy/deploy_dagster.sh materialize-fundamentals
EOF
