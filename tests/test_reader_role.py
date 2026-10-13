"""home-agent's reader role and the serving writes, on a real throwaway
TimescaleDB (the image deploy/timescale runs). Opt-in, since it needs Docker
and pulls the image:

    FACTOR_DOCKER_TESTS=1 python -m pytest -q tests/test_reader_role.py

scripts/create_reader.sh runs as on the NAS, except that `docker compose exec
timescaledb` is pointed at the test container. Nothing here touches a real
database; the container is removed afterwards.
"""
from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("FACTOR_DOCKER_TESTS") != "1" or shutil.which("docker") is None,
    reason="set FACTOR_DOCKER_TESTS=1 (needs Docker)",
)

REPO = Path(__file__).resolve().parents[1]
IMAGE = "timescale/timescaledb:2.17.2-pg16"
ROLE = "home_agent_reader"
DAGSTER_DDL = """
CREATE TABLE runs (id bigserial PRIMARY KEY, run_id varchar(255) UNIQUE, snapshot_id text,
    pipeline_name text, mode text, status varchar(63), run_body text, partition text,
    partition_set text, create_timestamp timestamp DEFAULT now(),
    update_timestamp timestamp DEFAULT now(), start_time float, end_time float,
    backfill_id varchar(255));
CREATE TABLE run_tags (id bigserial PRIMARY KEY, run_id text, key text, value text);
CREATE TABLE asset_check_executions (id bigserial PRIMARY KEY, asset_key text,
    check_name text, partition text, run_id varchar(255), execution_status varchar(255),
    evaluation_event text, evaluation_event_timestamp timestamp,
    evaluation_event_storage_id bigint, materialization_event_storage_id bigint,
    create_timestamp timestamp DEFAULT now());
INSERT INTO runs (run_id, pipeline_name, status, run_body) VALUES ('r1', 'daily_ingest', 'SUCCESS', '{}');
"""


def _run(*args, check=True, **kw):
    return subprocess.run(args, capture_output=True, text=True, check=check, **kw)


class Pg:
    def __init__(self, name: str, password: str, port: int, tmp: Path):
        self.name, self.password, self.port, self.tmp = name, password, port, tmp

    def admin(self, sql: str, db: str = "factor") -> str:
        return _run("docker", "exec", "-i", self.name, "psql", "-U", "factor", "-d", db,
                    "-v", "ON_ERROR_STOP=1", "-qtA", input=sql).stdout.strip()

    def reader(self, sql: str, password: str, db: str = "factor"):
        return _run("docker", "exec", "-i", "-e", f"PGPASSWORD={password}", self.name,
                    "psql", "-h", "127.0.0.1", "-U", ROLE, "-d", db, "-v", "ON_ERROR_STOP=1",
                    "-qtA", input=sql, check=False)

    def create_reader(self):
        """The script, run from a copy laid out as the repo, against this container."""
        return _run("bash", str(self.tmp / "scripts" / "create_reader.sh"), check=False,
                    env={**os.environ, "DOCKER": f"sh {self.tmp / 'docker.sh'}",
                         "CONTAINER": self.name})

    def can_read_artifact(self) -> bool:
        return self.admin(
            f"SELECT has_column_privilege('{ROLE}', 'public.model_registry', 'artifact', 'SELECT')"
        ) == "t"


@pytest.fixture(scope="module")
def pg(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("reader")
    name = f"factor-reader-test-{secrets.token_hex(4)}"
    password = secrets.token_hex(16)
    _run("docker", "run", "-d", "--rm", "--name", name,
         "-e", "POSTGRES_USER=factor", "-e", "POSTGRES_DB=factor",
         "-e", f"POSTGRES_PASSWORD={password}", "-p", "127.0.0.1::5432",
         "-v", f"{REPO / 'deploy/timescale/schema.sql'}:/docker-entrypoint-initdb.d/01-schema.sql:ro",
         IMAGE)
    try:
        # Ready once it answers over TCP: the init's temporary server listens
        # on the socket only.
        for _ in range(90):
            if _run("docker", "exec", name, "psql", "-h", "127.0.0.1", "-U", "factor",
                    "-d", "factor", "-c", "SELECT 1 FROM universe LIMIT 1", check=False,
                    env={**os.environ}).returncode == 0:
                break
            time.sleep(1)
        else:
            raise RuntimeError("test database never came up")
        port = int(_run("docker", "port", name, "5432/tcp").stdout.strip().rsplit(":", 1)[1])

        (tmp / "scripts").mkdir()
        shutil.copy(REPO / "scripts" / "create_reader.sh", tmp / "scripts")
        (tmp / "deploy" / "timescale").mkdir(parents=True)
        (tmp / "deploy" / "timescale" / ".env").write_text(
            f"POSTGRES_USER=factor\nPOSTGRES_DB=factor\nPOSTGRES_PASSWORD={password}\n")
        # create_reader.sh runs `$DOCKER compose exec -T timescaledb <cmd>`.
        (tmp / "docker.sh").write_text('shift 4\nexec docker exec -i "$CONTAINER" "$@"\n')

        db_ = Pg(name, password, port, tmp)
        db_.admin("CREATE DATABASE dagster;")
        db_.admin(DAGSTER_DDL, db="dagster")
        db_.admin("INSERT INTO prices VALUES ('AAA', '2026-09-30', 10, 100);")
        yield db_
    finally:
        _run("docker", "rm", "-f", name, check=False)


@pytest.fixture
def warehouse_env(pg, monkeypatch):
    """src.data.db pointed at the test container."""
    from src.data import db

    monkeypatch.setenv("FACTOR_DB_HOST", "127.0.0.1")
    monkeypatch.setenv("FACTOR_DB_PORT", str(pg.port))
    monkeypatch.setenv("POSTGRES_PASSWORD", pg.password)
    db.get_engine.cache_clear()
    yield db
    db.get_engine.cache_clear()


def _password(out: str) -> str:
    return out.strip().splitlines()[-1].strip()


def test_the_reader_role_end_to_end(pg, warehouse_env):
    db = warehouse_env

    # No model_registry yet: the script refuses rather than hand out a role
    # that the first registered model would open the artifact to.
    out = pg.create_reader()
    assert out.returncode != 0 and "model_registry does not exist yet" in out.stderr

    db.ensure_model_registry_table()
    db.save_model_registry({"model_version": "v1", "horizon": "1m", "model_name": "m"},
                           b"\x00model-bytes")
    out = pg.create_reader()
    assert out.returncode == 0, out.stderr
    pw = _password(out.stdout)
    assert pg.reader("SELECT count(manifest) FROM model_registry;", pw).returncode == 0
    assert pg.reader("SELECT artifact FROM model_registry;", pw).returncode != 0
    assert not pg.can_read_artifact()

    # Recreated by code that knows nothing of the reader (an older training
    # image): the event trigger withholds the artifact again.
    pg.admin("DROP TABLE model_registry;")
    pg.admin("CREATE TABLE model_registry (model_version text PRIMARY KEY, horizon text, "
             "model_name text, manifest jsonb, artifact bytea, created_at timestamptz);"
             "INSERT INTO model_registry VALUES ('v2', '1m', 'm', '{}', '\\x00', now());")
    assert not pg.can_read_artifact()
    assert pg.reader("SELECT artifact FROM model_registry;", pw).returncode != 0
    assert pg.reader("SELECT model_version FROM model_registry;", pw).stdout.strip() == "v2"

    # And without the trigger, through the app's own path: ensure re-applies it.
    pg.admin("DROP EVENT TRIGGER home_agent_reader_registry_guard; DROP TABLE model_registry;")
    db.ensure_model_registry_table()
    assert not pg.can_read_artifact()

    # Dagster: only the status columns, and a hand-made grant is undone on re-run.
    assert pg.reader("SELECT count(status) FROM runs;", pw, db="dagster").returncode == 0
    for sql in ("SELECT run_body FROM runs;", "SELECT value FROM run_tags;",
                "SELECT evaluation_event FROM asset_check_executions;"):
        assert pg.reader(sql, pw, db="dagster").returncode != 0, sql
    pg.admin(f"GRANT INSERT ON runs TO {ROLE};", db="dagster")
    out = pg.create_reader()
    assert out.returncode == 0, out.stderr
    assert pg.admin(f"SELECT has_table_privilege('{ROLE}', 'runs', 'INSERT')",
                    db="dagster") == "f"

    # The sweep fails the script on a write privilege no reset removes.
    pg.admin(f"GRANT CREATE ON SCHEMA public TO {ROLE};")
    out = pg.create_reader()
    assert out.returncode != 0 and "schema public" in out.stderr
    pg.admin(f"REVOKE CREATE ON SCHEMA public FROM {ROLE};")
    assert pg.create_reader().returncode == 0


def test_a_rebuilt_ranking_leaves_no_stale_rows(pg, warehouse_env):
    db = warehouse_env
    asof = pd.Timestamp("2026-10-31")
    first = pd.DataFrame({"ticker": ["A", "B", "C"], "gics_sector": "X", "pred": [3.0, 2, 1]})
    db.load_recommendations(first, "v1", asof)
    again = pd.DataFrame({"ticker": ["C", "D"], "gics_sector": "X", "pred": [2.0, 1]})
    db.load_recommendations(again, "v1", asof)
    rows = db.read_sql("SELECT rank, ticker FROM recommendations "
                       "WHERE as_of = '2026-10-31' AND model_version = 'v1' ORDER BY rank")
    assert rows.values.tolist() == [[1, "C"], [2, "D"]]


def test_a_failed_backtest_write_changes_nothing(pg, warehouse_env, monkeypatch):
    from src.backtest.run import run_strategy_backtest

    db = warehouse_env
    rng = np.random.default_rng(0)
    preds = pd.concat([
        pd.DataFrame({"date": d, "ticker": [f"T{i}" for i in range(40)],
                      "pred": rng.normal(size=40), "forward_return": rng.normal(0, .05, 40)})
        for d in pd.date_range("2024-01-31", periods=12, freq="ME")], ignore_index=True)
    res = run_strategy_backtest(preds, n_holdings=10, buffer_rank=10)
    params = {"n_holdings": 10, "buffer_rank": 10, "cost_bps": 6.0, "min_price": 5.0,
              "max_position_weight": 0.08}
    assert db.load_strategy_backtest("bt", res, params) == len(res.returns)

    def boom(*a, **k):
        raise RuntimeError("monthly rows failed")

    monkeypatch.setattr(db, "_upsert_rows", boom)
    with pytest.raises(RuntimeError):
        db.load_strategy_backtest("bt", res, {**params, "n_holdings": 99})
    row = db.read_sql("SELECT n_holdings, max_position_weight, (SELECT count(*) FROM "
                      "strategy_returns WHERE model_version = 'bt') AS months "
                      "FROM strategy_backtest WHERE model_version = 'bt'").iloc[0]
    assert int(row["n_holdings"]) == 10 and row["max_position_weight"] == 0.08
    assert int(row["months"]) == len(res.returns)
