"""Configuration loading and project paths.

All config lives in ``config/*.yaml``. Modules call :func:`load_config`
rather than hard-coding parameters, so behavior is tunable without code edits.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Project root = parent of src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"

# Data root is configurable so storage can live on a NAS/network mount without
# code changes: `export FACTOR_DATA_ROOT=/mnt/nas/factor-data`. Defaults to the
# local ./data directory.
DATA_ROOT = Path(os.environ.get("FACTOR_DATA_ROOT", PROJECT_ROOT / "data"))
DATA_RAW = DATA_ROOT / "raw"
DATA_PROCESSED = DATA_ROOT / "processed"


@lru_cache(maxsize=None)
def load_config(name: str) -> dict[str, Any]:
    """Load a YAML config by stem name (e.g. ``"data"`` -> config/data.yaml)."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    """Create the data cache directories if they do not exist."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
