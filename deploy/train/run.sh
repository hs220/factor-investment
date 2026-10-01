#!/usr/bin/env bash
# Run one training job on the training box (manual / Phase-1 path).
#
#   ./deploy/train/run.sh                  # tuned LightGBM, 1m horizon
#   ./deploy/train/run.sh --no-tune        # args pass through to the entrypoint
#
# DB creds come from ~/.factor.env on the box (POSTGRES_PASSWORD, FACTOR_DB_HOST,
# FACTOR_DB_PORT; mode 600) — passed as a path relative to ~, where ssh lands.
# --network host keeps the container on the host's firewall: the box may reach
# the NAS only on Postgres :5433 (ufw D12 exception), whereas bridge-networked
# traffic would bypass those host OUTPUT rules.
set -euo pipefail

TRAIN_HOST="${TRAIN_HOST:-hsheng@192.168.68.76}"
IMAGE="${IMAGE:-factor-train:latest}"

exec ssh "$TRAIN_HOST" docker run --rm --network host \
  --env-file .factor.env --name factor-train "$IMAGE" "$@"
