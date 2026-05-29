# Build/data pipeline image — runs the I/O-intensive jobs on the NAS.
# Network-bound work (yfinance + SEC EDGAR fetches) and parquet writes stay
# local to the NAS disk; the workstation only reads processed parquet.
FROM python:3.12-slim

# libgomp1 is required by lightgbm/xgboost at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the library + entry points (data/ is NOT copied — it is a mounted volume).
COPY src/ ./src/
COPY pipelines/ ./pipelines/
COPY config/ ./config/

# Data lives on the mounted NAS volume.
ENV FACTOR_DATA_ROOT=/data
VOLUME ["/data"]

# Default job; override in compose/Task Scheduler (e.g. train, backtest).
ENTRYPOINT ["python", "-m"]
CMD ["pipelines.build_dataset"]
