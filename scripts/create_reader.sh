#!/usr/bin/env bash
# Create (or re-key) the read-only role home-agent's `factor` connector uses.
#
# The connector reads the warehouse to answer questions about the strategy: the
# model's current ranking, the saved Roth IRA holdings, the backtest, and whether
# the pipeline is fresh. It gets its own role rather than the `factor` superuser,
# and the role can do one thing: SELECT. That is enforced three ways, so no one
# of them is load-bearing:
#   - privileges: CONNECT and SELECT, nothing else, no ownership of anything;
#   - default_transaction_read_only, so even a granted write would be refused;
#   - statement_timeout, so a bad query cannot pin the NAS's two cores.
# The probes at the end check the first one on its own: they open a READ WRITE
# transaction (which the role is free to do) and expect the write to be refused
# for lack of privilege.
#
# What it can read:
#   - every table in the `factor` database, now and future (ALTER DEFAULT
#     PRIVILEGES: the pipeline creates tables lazily -- holdings, the serving
#     snapshots -- so a plain GRANT would miss the ones not yet created), except
#     model_registry's `artifact` column: the pickled model is no reader's business;
#   - in the `dagster` database, only these columns:
#       runs: run_id, pipeline_name, status, start_time, end_time, create_timestamp
#       asset_check_executions: asset_key, check_name, run_id, execution_status,
#         evaluation_event_timestamp, create_timestamp
#     Nothing that can carry run config, tags or payloads: not runs.run_body,
#     not run_tags, event_logs or asset_check_executions.evaluation_event.
#
# Prints a newly generated password ONCE. Paste it into the home-agent app:
# Settings > Connections > Factor Investment. It is not saved anywhere here, and
# re-running this script replaces it (the old one stops working).
#
# Usage (on the NAS, from the repo directory, with the timescale stack up):
#   scripts/create_reader.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROLE="home_agent_reader"

# DSM's docker needs sudo and is not on the default PATH; a dev Mac's is neither.
if [ -n "${DOCKER:-}" ]; then
  :
elif [ -x /usr/local/bin/docker ] && [ -d /volume1 ]; then
  DOCKER="sudo /usr/local/bin/docker"
else
  DOCKER="docker"
fi

cd "$REPO_DIR/deploy/timescale"

# shellcheck disable=SC1091
set -a; . ./.env; set +a
PGUSER_="${POSTGRES_USER:-factor}"
PGDB_="${POSTGRES_DB:-factor}"
DAGSTER_DB="dagster"

psql_admin() {  # $1 = database; SQL on stdin
  $DOCKER compose exec -T timescaledb psql -U "$PGUSER_" -d "$1" -v ON_ERROR_STOP=1 --quiet -tA
}

# Alphanumeric only, so it survives being typed on a phone and needs no quoting.
# A bounded read, not `tr < /dev/urandom | head`: that kills tr with SIGPIPE,
# which pipefail turns into a silent exit before anything is printed.
PASSWORD="$(head -c 512 /dev/urandom | LC_ALL=C tr -dc 'A-Za-z0-9' | cut -c1-32)"
[ "${#PASSWORD}" -eq 32 ] || { echo "FAILED: could not generate a password" >&2; exit 1; }

# The SQL goes over stdin, not argv: a password in a command line is visible to
# anyone running `ps` on the NAS. :'pw' is psql's quoted-literal interpolation.
psql_admin "$PGDB_" <<SQL
\\set pw '$PASSWORD'
SELECT 'CREATE ROLE $ROLE LOGIN'
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '$ROLE') \\gexec
SELECT format('ALTER ROLE $ROLE WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION PASSWORD %L', :'pw') \\gexec
ALTER ROLE $ROLE SET default_transaction_read_only = on;
ALTER ROLE $ROLE SET statement_timeout = '10s';
ALTER ROLE $ROLE CONNECTION LIMIT 4;

GRANT CONNECT ON DATABASE $PGDB_ TO $ROLE;
GRANT USAGE ON SCHEMA public TO $ROLE;
-- From scratch on every run, so a privilege granted by hand since is undone.
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM $ROLE;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO $ROLE;
ALTER DEFAULT PRIVILEGES FOR ROLE $PGUSER_ IN SCHEMA public GRANT SELECT ON TABLES TO $ROLE;

-- model_registry: everything but the pickled model. A table-wide SELECT would
-- cover every column, so it is replaced by a column list.
DO \$\$
BEGIN
  IF to_regclass('public.model_registry') IS NOT NULL THEN
    REVOKE SELECT ON model_registry FROM $ROLE;
    GRANT SELECT (model_version, horizon, model_name, manifest, created_at)
      ON model_registry TO $ROLE;
  END IF;
END \$\$;
SQL

# Dagster's run storage is a separate database on the same server. Status
# columns only: run_body and evaluation_event carry run config and payloads.
if [ "$(echo "SELECT 1 FROM pg_database WHERE datname = '$DAGSTER_DB'" | psql_admin "$PGDB_")" = "1" ]; then
  psql_admin "$DAGSTER_DB" <<SQL
GRANT CONNECT ON DATABASE $DAGSTER_DB TO $ROLE;
GRANT USAGE ON SCHEMA public TO $ROLE;
DO \$\$
BEGIN
  IF to_regclass('public.runs') IS NOT NULL THEN
    GRANT SELECT (run_id, pipeline_name, status, start_time, end_time, create_timestamp)
      ON runs TO $ROLE;
  END IF;
  IF to_regclass('public.asset_check_executions') IS NOT NULL THEN
    GRANT SELECT (asset_key, check_name, run_id, execution_status,
                  evaluation_event_timestamp, create_timestamp)
      ON asset_check_executions TO $ROLE;
  END IF;
END \$\$;
SQL
  HAVE_DAGSTER=1
else
  echo "Note: no '$DAGSTER_DB' database here; the connector will not see pipeline runs."
  HAVE_DAGSTER=0
fi

# Prove it as the role itself, logged in over TCP with the new password -- that
# is the only path on which the ALTER ROLE ... SET lines apply, and it checks the
# password too. The password rides in on stdin's first line, never argv.
as_reader() {  # $1 = database, $2 = SQL
  printf '%s\n%s\n' "$PASSWORD" "$2" | $DOCKER compose exec -T timescaledb sh -c \
    'read -r PGPASSWORD; export PGPASSWORD; exec psql -h 127.0.0.1 -U "$0" -d "$1" -v ON_ERROR_STOP=1 --quiet -tA' \
    "$ROLE" "$1"
}
refused() {  # $1 = database, $2 = what, $3 = SQL that must fail
  if as_reader "$1" "$3" >/dev/null 2>&1; then
    echo "FAILED: $ROLE could $2; it is not read-only" >&2
    exit 1
  fi
}

as_reader "$PGDB_" "SELECT max(date) FROM prices;" >/dev/null \
  || { echo "FAILED: $ROLE cannot read prices" >&2; exit 1; }
# READ WRITE on purpose: these must be refused by privileges alone.
refused "$PGDB_" "create a table" \
  "BEGIN READ WRITE; CREATE TABLE _reader_probe (x int); ROLLBACK;"
refused "$PGDB_" "write to universe" \
  "BEGIN READ WRITE; INSERT INTO universe (ticker) VALUES ('_reader_probe'); ROLLBACK;"
refused "$PGDB_" "read the model artifact" \
  "SELECT artifact FROM model_registry LIMIT 1;"
if [ "$HAVE_DAGSTER" = 1 ]; then
  as_reader "$DAGSTER_DB" "SELECT count(status) FROM runs;" >/dev/null \
    || { echo "FAILED: $ROLE cannot read Dagster run status" >&2; exit 1; }
  refused "$DAGSTER_DB" "read run configs" "SELECT run_body FROM runs LIMIT 1;"
fi

echo "Role $ROLE is ready: reads succeed, writes are refused."
echo
echo "Password (shown once; set it in the home-agent app under"
echo "Settings > Connections > Factor Investment):"
echo
echo "  $PASSWORD"
