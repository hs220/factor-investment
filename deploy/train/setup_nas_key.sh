#!/usr/bin/env bash
# One-time (idempotent) setup so NAS Dagster can launch training on the box.
#
#   ./deploy/train/setup_nas_key.sh
#
# 1. NAS: create a dedicated ed25519 key + a pinned known_hosts for the box under
#    deploy/dagster/train_ssh/ (gitignored; mounted read-only into the Dagster
#    containers at /run/train_ssh).
# 2. Box: install the forced-command script ~/factor-train/remote_train.sh.
# 3. Box: authorize the NAS key ONLY for that script, ONLY from the NAS IP.
#
# Run from the laptop (it can ssh to both; the box itself may not dial the NAS).
set -euo pipefail

NAS_HOST="${NAS_HOST:-hsheng@192.168.68.70}"
NAS_IP="${NAS_IP:-192.168.68.70}"
NAS_DIR="${NAS_DIR:-/volume1/docker/factor-investment}"
TRAIN_HOST="${TRAIN_HOST:-hsheng@192.168.68.76}"
TRAIN_IP="${TRAIN_HOST#*@}"
KEY_DIR="$NAS_DIR/deploy/dagster/train_ssh"
TAG="factor-dagster-train"

echo "==> NAS: ensure training key in $KEY_DIR..."
ssh "$NAS_HOST" "
  set -e
  mkdir -p '$KEY_DIR' && chmod 700 '$KEY_DIR'
  [ -f '$KEY_DIR/id_ed25519' ] || ssh-keygen -q -t ed25519 -N '' -C '$TAG' -f '$KEY_DIR/id_ed25519'
  chmod 600 '$KEY_DIR/id_ed25519'
"

# Pin the box's host key on the NAS. Synology has no ssh-keyscan, so read the
# box's own public host key and require it to match the key this laptop already
# trusts for the box (so a spoofed box can't get pinned).
echo "==> Pin the box host key on the NAS (verified against this laptop's known_hosts)..."
hostkey="$(ssh "$TRAIN_HOST" 'cut -d" " -f1,2 /etc/ssh/ssh_host_ed25519_key.pub')"
ssh-keygen -F "$TRAIN_IP" | grep -qF "$hostkey" \
  || { echo "box host key does not match ~/.ssh/known_hosts for $TRAIN_IP — aborting" >&2; exit 1; }
printf '%s %s\n' "$TRAIN_IP" "$hostkey" | ssh "$NAS_HOST" "cat > '$KEY_DIR/known_hosts' && chmod 644 '$KEY_DIR/known_hosts'"

echo "==> Box: install forced-command script..."
ssh "$TRAIN_HOST" "mkdir -p ~/factor-train && cat > ~/factor-train/remote_train.sh && chmod 755 ~/factor-train/remote_train.sh" \
  < "$(dirname "$0")/remote_train.sh"

echo "==> Box: authorize the NAS key for that command only..."
pub="$(ssh "$NAS_HOST" "cat '$KEY_DIR/id_ed25519.pub'")"
box_home="$(ssh "$TRAIN_HOST" 'echo $HOME')"     # command= takes no env expansion
printf 'restrict,command="%s/factor-train/remote_train.sh",from="%s" %s\n' \
    "$box_home" "$NAS_IP" "$pub" \
  | ssh "$TRAIN_HOST" "
      set -e
      mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys
      { grep -v ' $TAG\$' ~/.ssh/authorized_keys || true; cat; } > ~/.ssh/authorized_keys.new
      chmod 600 ~/.ssh/authorized_keys.new && mv ~/.ssh/authorized_keys.new ~/.ssh/authorized_keys
      grep ' $TAG\$' ~/.ssh/authorized_keys | cut -c1-110
    "
echo "==> Done. Test from the NAS:  ssh -i $KEY_DIR/id_ed25519 -o UserKnownHostsFile=$KEY_DIR/known_hosts $TRAIN_HOST --bogus   (expect: rejected argument)"
