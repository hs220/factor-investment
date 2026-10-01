"""Tests for the NAS -> training-box launcher (no ssh / box needed)."""
from __future__ import annotations

import subprocess

import pytest

from orchestration import remote_train


class _Log:
    def __init__(self):
        self.lines: list[str] = []

    def info(self, msg):
        self.lines.append(msg)


def test_ssh_command_pins_key_and_host_key():
    cmd = remote_train.ssh_command(["--no-tune"], host="u@box", key_dir="/k")
    assert cmd[:3] == ["ssh", "-i", "/k/id_ed25519"]
    assert "StrictHostKeyChecking=yes" in cmd and "UserKnownHostsFile=/k/known_hosts" in cmd
    assert cmd[-2:] == ["u@box", "--no-tune"]


def test_registered_version_last_wins():
    lines = ["panel_monthly: 1 rows", "registered a-1m-1 in model_registry; 5 prediction rows",
             "registered a-1m-2 in model_registry; 5 prediction rows"]
    assert remote_train.registered_version(lines) == "a-1m-2"
    assert remote_train.registered_version(["nothing"]) is None


def _fake(monkeypatch, script: str):
    """Make the 'ssh' call run a local shell snippet instead."""
    monkeypatch.setattr(remote_train, "ssh_command",
                        lambda args, host=None: ["bash", "-c", script])


def test_run_streams_log_and_returns_version(monkeypatch):
    _fake(monkeypatch, "echo training; echo 'registered m-1m-9 in model_registry; 3 prediction rows'")
    log = _Log()
    assert remote_train.run_remote_training(log) == "m-1m-9"
    assert "training" in log.lines


def test_run_raises_on_failure(monkeypatch):
    _fake(monkeypatch, "echo boom; exit 3")
    with pytest.raises(RuntimeError, match="exit 3"):
        remote_train.run_remote_training(_Log())


def test_run_raises_without_registration(monkeypatch):
    _fake(monkeypatch, "echo done")
    with pytest.raises(RuntimeError, match="no model_registry entry"):
        remote_train.run_remote_training(_Log())


def test_run_timeout(monkeypatch):
    _fake(monkeypatch, "sleep 5")
    with pytest.raises(RuntimeError, match="timeout"):
        remote_train.run_remote_training(_Log(), timeout_s=1)


@pytest.mark.parametrize("req, ok", [
    ("", True), ("--no-tune", True), ("--model elasticnet --horizon 3m", True),
    ("--model 'lightgbm;id'", False), ("bash -i", False), ("--horizon ../x", False),
    ("--model", False), ("$(id)", False),
])
def test_forced_command_allowlist(tmp_path, req, ok):
    """remote_train.sh only ever execs docker with allowlisted args."""
    stub = tmp_path / "docker"
    stub.write_text('#!/bin/sh\necho "DOCKER $*"\n')
    stub.chmod(0o755)
    env = {"PATH": f"{tmp_path}:/usr/bin:/bin", "HOME": str(tmp_path),
           "SSH_ORIGINAL_COMMAND": req}
    r = subprocess.run(["bash", "deploy/train/remote_train.sh"], env=env,
                       capture_output=True, text=True)
    assert (r.returncode == 0) == ok, r.stderr
    if ok:
        assert r.stdout.startswith("DOCKER run --rm --network host")
        assert "factor-train:latest" in r.stdout
