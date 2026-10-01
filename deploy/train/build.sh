#!/usr/bin/env bash
# Build the factor-train image ON the training box from the committed tree.
#
#   ./deploy/train/build.sh                # build factor-train:latest from HEAD
#
# The build context is `git archive HEAD` streamed over ssh, so only committed
# code ships (commit first) and the box needs no GitHub access. PipesDockerClient
# runs an existing image; it doesn't build — rerun this after code changes.
#
# Config via env:
#   TRAIN_HOST  ssh target  (default hsheng@192.168.68.76)
#   IMAGE       image tag   (default factor-train:latest)
set -euo pipefail

TRAIN_HOST="${TRAIN_HOST:-hsheng@192.168.68.76}"
IMAGE="${IMAGE:-factor-train:latest}"
SHA="$(git rev-parse --short HEAD)"

if [ -n "$(git status --porcelain -- src orchestration config deploy/train requirements-train.txt constraints-ml.txt)" ]; then
  echo "warning: uncommitted changes are NOT in the image (context = HEAD $SHA)" >&2
fi

echo "==> Building $IMAGE on $TRAIN_HOST from HEAD $SHA..."
git archive --format=tar HEAD \
  | ssh "$TRAIN_HOST" "docker build -t '$IMAGE' -f deploy/train/Dockerfile --build-arg CODE_SHA=$SHA -"
ssh "$TRAIN_HOST" "docker image inspect '$IMAGE' --format '{{.Id}} {{.Created}}'"
