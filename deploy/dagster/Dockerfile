# Dagster orchestration image — runs the ingestion assets on the NAS.
# Webserver (UI) and daemon (schedules/run-queue) share this image; run history
# and schedule state live in the 'dagster' Postgres DB (stateless container).
FROM python:3.12-slim

WORKDIR /app

# libgomp1: OpenMP runtime for LightGBM (NAS-side inference / ad-hoc runs).
# openssh-client: model_predictions launches training on the training box over ssh.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 openssh-client \
    && rm -rf /var/lib/apt/lists/*

# constraints-ml.txt pins the ML stack to the training image's versions so a
# model_registry artifact trained on the training box loads here.
COPY requirements-orchestration.txt constraints-ml.txt ./
RUN pip install --no-cache-dir -r requirements-orchestration.txt -c constraints-ml.txt

# Library + orchestration code + config (data/ is the warehouse, not copied).
COPY src/ ./src/
COPY orchestration/ ./orchestration/
COPY config/ ./config/

# Dagster instance config (Postgres-backed storage) + code location.
COPY deploy/dagster/dagster.yaml /opt/dagster/home/dagster.yaml
COPY deploy/dagster/workspace.yaml ./workspace.yaml
ENV DAGSTER_HOME=/opt/dagster/home

EXPOSE 3000
