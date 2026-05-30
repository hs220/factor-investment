#!/usr/bin/env bash
#
# NAS deployment for the factor-investment data backfill (steps 2-4).
# Run this ON the Synology after SSHing in and creating the shared folder.
#
#   2. clone/update the repo on the NAS
#   3. point the container's /data volume at the NAS shared folder
#   4. build the image and run the backfill
#
# Bootstrap (first time): scp just this file up, then run it —
#   scp deploy/nas_backfill.sh admin@<nas-ip>:/volume1/factor-data/
#   ssh admin@<nas-ip> 'REPO_URL=<git-url> bash /volume1/factor-data/nas_backfill.sh'
#
# Config via env vars (all optional except REPO_URL on first run):
#   REPO_URL          git URL to clone (required if repo not already at DEST_DIR)
#   DEST_DIR          where the repo lives on the NAS   (default /volume1/factor-data/repo)
#   FACTOR_HOST_DATA  NAS shared folder for data        (default /volume1/factor-data)
#   SERVICE           compose service to run            (default build-dataset; use 'incremental' for updates)
#   INCREMENTAL=1     shorthand for SERVICE=incremental
#
set -euo pipefail

REPO_URL="${REPO_URL:-}"
DEST_DIR="${DEST_DIR:-/volume1/factor-data/repo}"
export FACTOR_HOST_DATA="${FACTOR_HOST_DATA:-/volume1/factor-data}"
SERVICE="${SERVICE:-build-dataset}"
[ "${INCREMENTAL:-0}" = "1" ] && SERVICE="incremental"

log() { printf '\n=== %s ===\n' "$*"; }

# --- resolve docker / docker compose (Synology often needs sudo) -------------
DOCKER="docker"
if ! docker info >/dev/null 2>&1; then
  if sudo docker info >/dev/null 2>&1; then
    DOCKER="sudo docker"
  else
    echo "ERROR: cannot run docker. Is Container Manager installed and running?" >&2
    exit 1
  fi
fi
if $DOCKER compose version >/dev/null 2>&1; then
  COMPOSE="$DOCKER compose"
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE="${DOCKER%docker}docker-compose"   # preserve sudo prefix if present
else
  echo "ERROR: docker compose not found." >&2
  exit 1
fi
echo "Using: $COMPOSE   (data volume: $FACTOR_HOST_DATA)"

# --- step 2: clone or update the repo ----------------------------------------
log "Step 2: sync repo at $DEST_DIR"
if [ -d "$DEST_DIR/.git" ]; then
  echo "Repo exists; pulling latest..."
  git -C "$DEST_DIR" pull --ff-only
else
  if [ -z "$REPO_URL" ]; then
    echo "ERROR: $DEST_DIR is not a git repo and REPO_URL is unset." >&2
    echo "Set REPO_URL=<git-url> on first run." >&2
    exit 1
  fi
  echo "Cloning $REPO_URL ..."
  git clone "$REPO_URL" "$DEST_DIR"
fi

# --- step 3: ensure the data shared folder exists ----------------------------
log "Step 3: ensure data dir $FACTOR_HOST_DATA"
mkdir -p "$FACTOR_HOST_DATA/raw" "$FACTOR_HOST_DATA/processed"
echo "OK"

# --- step 4: build image and run the job -------------------------------------
cd "$DEST_DIR"
log "Step 4: build image"
$COMPOSE build "$SERVICE"

log "Step 4: run '$SERVICE' (this is the long backfill; ~45-90 min)"
$COMPOSE run --rm "$SERVICE"

log "Done. Artifacts in $FACTOR_HOST_DATA/processed/"
