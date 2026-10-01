#!/usr/bin/env bash
# Deploy the Streamlit dashboard on the Synology NAS (same convention as
# deploy_dagster.sh). The NAS pulls from origin — push before deploying.
#
# Usage (from the laptop):
#   ./deploy/deploy_streamlit.sh up       # sync code, build, start
#   ./deploy/deploy_streamlit.sh status | logs | down
#
# UI: http://192.168.68.70:8501
set -euo pipefail

NAS_HOST="${NAS_HOST:-hsheng@192.168.68.70}"
NAS_DIR="${NAS_DIR:-/volume1/docker/factor-investment}"
BRANCH="${BRANCH:-master}"
DOCKER="sudo /usr/local/bin/docker"
TS_DIR="$NAS_DIR/deploy/timescale"
ST_DIR="$NAS_DIR/deploy/streamlit"
MODE="${1:-up}"

remote() { ssh "$NAS_HOST" "$1"; }

case "$MODE" in
  status) exec ssh "$NAS_HOST" "cd '$ST_DIR' && $DOCKER compose ps" ;;
  logs)   exec ssh "$NAS_HOST" "cd '$ST_DIR' && $DOCKER compose logs --tail=60" ;;
  down)   exec ssh "$NAS_HOST" "cd '$ST_DIR' && $DOCKER compose down" ;;
esac

echo "==> Sync code on NAS ($NAS_DIR @ $BRANCH)..."
remote "git -C '$NAS_DIR' fetch origin '$BRANCH' && git -C '$NAS_DIR' checkout '$BRANCH' && git -C '$NAS_DIR' pull --ff-only"

echo "==> Write streamlit .env from the Timescale password..."
remote "
  set -e
  pw=\$(grep '^POSTGRES_PASSWORD=' '$TS_DIR/.env' | cut -d= -f2)
  umask 077
  printf 'POSTGRES_PASSWORD=%s\nFACTOR_DB_HOST=192.168.68.70\nFACTOR_DB_PORT=5433\nDASHBOARD_PORT=8501\n' \"\$pw\" > '$ST_DIR/.env'
"

echo "==> Build image and start the dashboard..."
remote "cd '$ST_DIR' && $DOCKER compose build && $DOCKER compose up -d && $DOCKER compose ps"

echo
echo "==> Dashboard: http://192.168.68.70:8501"
