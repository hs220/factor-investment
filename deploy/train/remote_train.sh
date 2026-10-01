#!/usr/bin/env bash
# SSH forced command for the NAS's training key (installed on the training box by
# deploy/train/setup_nas_key.sh as ~/factor-train/remote_train.sh).
#
# authorized_keys pins the NAS key to this script (`restrict,command=...,from=NAS`),
# so whatever the NAS asks for arrives only as $SSH_ORIGINAL_COMMAND and is parsed
# against a strict allowlist — the key can start a training run and nothing else
# (no shell, no docker API, no forwarding). A compromised NAS can at most trigger
# training; it gains no foothold on the box (keeps the spirit of ufw rule D12).
#
# Accepted:  [--model lightgbm|elasticnet|xgboost] [--horizon <N>m] [--no-tune]
set -euo pipefail

IMAGE="factor-train:latest"
args=()
read -r -a req <<< "${SSH_ORIGINAL_COMMAND:-}"

i=0
while [ "$i" -lt "${#req[@]}" ]; do
  a="${req[$i]}"
  case "$a" in
    --no-tune)
      args+=(--no-tune) ;;
    --model)
      v="${req[$((i + 1))]:-}"
      [[ "$v" =~ ^(lightgbm|elasticnet|xgboost)$ ]] || { echo "rejected --model '$v'" >&2; exit 2; }
      args+=(--model "$v"); i=$((i + 1)) ;;
    --horizon)
      v="${req[$((i + 1))]:-}"
      [[ "$v" =~ ^[0-9]{1,2}m$ ]] || { echo "rejected --horizon '$v'" >&2; exit 2; }
      args+=(--horizon "$v"); i=$((i + 1)) ;;
    *)
      echo "rejected argument '$a'" >&2; exit 2 ;;
  esac
  i=$((i + 1))
done

# One run at a time: the fixed container name makes a concurrent request fail fast.
# --network host keeps the container behind the host's ufw (Postgres :5433 only).
exec docker run --rm --network host --env-file "$HOME/.factor.env" \
  --name factor-train "$IMAGE" ${args[@]+"${args[@]}"}
