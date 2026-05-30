# Data pipeline image — runs the I/O-intensive backfill/incremental jobs on the
# NAS. Network-bound work (yfinance + SEC EDGAR fetches) and parquet writes stay
# local to the NAS disk; the workstation reads the processed parquet over NFS.
#
# Lean by design: only data-layer deps (no sklearn/lightgbm/cvxpy/jupyter) so
# the image builds fast and reliably on a NAS CPU. Heavy modeling runs on the
# workstation via the full requirements.txt.
FROM python:3.12-slim

WORKDIR /app

# Install data-only deps first for layer caching.
COPY requirements-data.txt .
RUN pip install --no-cache-dir -r requirements-data.txt

# Copy the library + entry points (data/ is NOT copied — it is a mounted volume).
COPY src/ ./src/
COPY pipelines/ ./pipelines/
COPY config/ ./config/

# Data lives on the mounted NAS volume.
ENV FACTOR_DATA_ROOT=/data
VOLUME ["/data"]

# Default job is the full backfill; override for incremental in scheduler:
#   docker compose run --rm build-dataset python -m pipelines.build_dataset --incremental
ENTRYPOINT ["python", "-m"]
CMD ["pipelines.build_dataset"]
