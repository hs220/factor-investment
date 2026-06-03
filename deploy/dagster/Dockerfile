# Dagster orchestration image — runs the ingestion assets on the NAS.
# Webserver (UI) and daemon (schedules/run-queue) share this image; run history
# and schedule state live in the 'dagster' Postgres DB (stateless container).
FROM python:3.12-slim

WORKDIR /app

# libgomp1: OpenMP runtime required by LightGBM (model training asset).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-orchestration.txt .
RUN pip install --no-cache-dir -r requirements-orchestration.txt

# Library + orchestration code + config (data/ is the warehouse, not copied).
COPY src/ ./src/
COPY orchestration/ ./orchestration/
COPY config/ ./config/

# Dagster instance config (Postgres-backed storage) + code location.
COPY deploy/dagster/dagster.yaml /opt/dagster/home/dagster.yaml
COPY deploy/dagster/workspace.yaml ./workspace.yaml
ENV DAGSTER_HOME=/opt/dagster/home

EXPOSE 3000
