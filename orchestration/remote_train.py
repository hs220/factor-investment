"""Launch model training on the training box from a Dagster step.

The NAS can't run the walk-forward itself, so ``model_predictions`` dials the
training box over ssh with a key that ``authorized_keys`` pins to a single forced
command (``deploy/train/remote_train.sh`` -> ``docker run factor-train``). The
container reads the warehouse, trains, and writes ``predictions`` + the
``model_registry`` artifact itself; this side only streams its log into the run
and learns the new ``model_version`` from the ``registered ...`` line.

Why not ``PipesDockerClient``: that needs the full Docker API over ssh — i.e. a
root-equivalent key on the box. The forced command limits a compromised NAS to
"start a training run".
"""
from __future__ import annotations

import os
import re
import subprocess
import threading
from collections.abc import Iterable

KEY_DIR = "/run/train_ssh"
# Entrypoint prints: "registered <version> in model_registry; <n> prediction rows"
_REGISTERED = re.compile(r"registered (\S+) in model_registry")


def ssh_command(args: list[str], *, host: str | None = None, key_dir: str = KEY_DIR) -> list[str]:
    """ssh argv for the forced command; ``args`` arrive as $SSH_ORIGINAL_COMMAND."""
    host = host or os.environ.get("TRAIN_SSH_HOST", "hsheng@192.168.68.76")
    return [
        "ssh", "-T", "-i", f"{key_dir}/id_ed25519",   # -T: no pty (forced command)
        "-o", "BatchMode=yes",
        "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=yes",
        "-o", f"UserKnownHostsFile={key_dir}/known_hosts",
        "-o", "ConnectTimeout=15",
        # Detect a dead box/LAN instead of hanging for hours on a silent socket.
        "-o", "ServerAliveInterval=60", "-o", "ServerAliveCountMax=10",
        # glibc getopt permutes argv, so without "--" ssh would parse the remote
        # args (e.g. --no-tune) as its own options.
        "--", host, *args,
    ]


def registered_version(lines: Iterable[str]) -> str | None:
    """The model_version the container reported registering (last one wins)."""
    version = None
    for line in lines:
        m = _REGISTERED.search(line)
        if m:
            version = m.group(1)
    return version


def run_remote_training(
    log,
    args: list[str] | None = None,
    *,
    timeout_s: int = 4 * 3600,
    host: str | None = None,
) -> str:
    """Run one training job on the box; stream its output to ``log.info``.

    Returns the registered ``model_version``. Raises on non-zero exit, timeout,
    or if the container never reported a registration.
    """
    cmd = ssh_command(list(args or []), host=host)
    log.info(f"launching training: {' '.join(cmd[-1 - len(args or []):])}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    timer = threading.Timer(timeout_s, proc.kill)
    timer.start()
    tail: list[str] = []
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip()
            if line:
                log.info(line)
                tail = (tail + [line])[-200:]
        rc = proc.wait()
    finally:
        timed_out = not timer.is_alive() and proc.returncode not in (0, None)
        timer.cancel()
    if timed_out:
        raise RuntimeError(f"remote training killed after {timeout_s}s timeout")
    if rc != 0:
        raise RuntimeError(f"remote training failed (exit {rc}); last output:\n"
                           + "\n".join(tail[-20:]))
    version = registered_version(tail)
    if version is None:
        raise RuntimeError("remote training exited 0 but reported no model_registry entry")
    return version
