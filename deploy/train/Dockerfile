# Training image — runs the heavy walk-forward + deployment fit on the training box
# (Linux mini PC, 192.168.68.76), launched by hand or by Dagster PipesDockerClient
# over remote Docker. Reads panel_monthly from the NAS warehouse and writes
# predictions + the model_registry artifact back to it; needs no volumes.
#
# python:3.12 + constraints-ml.txt match the NAS dagster image exactly, so an
# artifact trained here unpickles there. (CPU-only: LightGBM doesn't use a GPU and
# the box has none; the torch phase will need a CUDA base on a GPU host.)
FROM python:3.12-slim

WORKDIR /app

# libgomp1: OpenMP runtime required by LightGBM.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-train.txt constraints-ml.txt ./
RUN pip install --no-cache-dir -r requirements-train.txt -c constraints-ml.txt

COPY src/ ./src/
COPY orchestration/__init__.py orchestration/train_entrypoint.py ./orchestration/
COPY config/ ./config/

# Stamped by deploy/train/build.sh (no .git in the image) -> manifest.code_sha.
ARG CODE_SHA=unknown
ENV FACTOR_CODE_SHA=${CODE_SHA}
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "orchestration.train_entrypoint"]
