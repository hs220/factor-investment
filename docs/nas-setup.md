# Running the data pipeline on a Synology / QNAP NAS

Goal: run the I/O-intensive jobs (`build_dataset`, optionally `train`/`backtest`)
**on the NAS**, with data stored on the NAS, while the workstation only runs
notebooks/training that read the processed parquet over a mount.

The network-bound work (yfinance + SEC EDGAR fetches) and parquet writes stay
local to the NAS disk. NAS CPUs are modest, so heavy modeling can still be run
on the workstation reading the same `data/processed/` over NFS.

## 1. Create the shared folder

In DSM: **Control Panel → Shared Folder → Create** → e.g. `factor-data`
(on `/volume1/factor-data`). This holds `raw/` and `processed/`.

## 2. (Workstation access) Enable NFS and mount

- DSM: **Control Panel → File Services → NFS** → enable.
- On the shared folder: **Edit → NFS Permissions** → add your workstation's IP,
  `Read/Write`, squash = "no mapping".
- On the workstation (macOS/Linux):

  ```bash
  sudo mount -t nfs <nas-ip>:/volume1/factor-data /mnt/factor-data
  export FACTOR_DATA_ROOT=/mnt/factor-data   # code now reads/writes here
  ```

  Prefer NFS over SMB — better parquet throughput and POSIX rename semantics
  (the cache uses atomic temp-file + `os.replace`, which NFS honors).

## 3. Run the pipeline on the NAS (Container Manager)

1. Copy this repo to the NAS (e.g. `/volume1/factor-data/repo` or any share).
2. Edit `docker-compose.yml` → set the volume host path to your shared folder
   (`/volume1/factor-data:/data`).
3. DSM: **Container Manager → Project → Create** → point at the repo folder and
   its `docker-compose.yml`. Container Manager builds the image from the
   `Dockerfile`.
4. Run a one-off job: in the project, run the `build-dataset` service
   (equivalent to `docker compose run --rm build-dataset`).

The container sets `FACTOR_DATA_ROOT=/data`, mapped to the NAS folder, so all
artifacts land in `/volume1/factor-data/processed/`.

## 4. Schedule periodic rebuilds

DSM: **Control Panel → Task Scheduler → Create → Scheduled Task → User-defined
script**, run e.g. monthly after month-end:

```bash
cd /volume1/factor-data/repo && /usr/local/bin/docker compose run --rm build-dataset
```

(Path to `docker`/`docker compose` may differ by DSM version; check
`which docker`.)

## 5. Workstation workflow

With `FACTOR_DATA_ROOT` pointed at the mount, run notebooks/training normally —
they read the NAS-built parquet:

```bash
export FACTOR_DATA_ROOT=/mnt/factor-data
python -m pipelines.train --model lightgbm
jupyter lab    # notebooks read processed parquet over the mount
```

### Retrieving data: parquet + DuckDB (no DB server)

The container is a *producer* — it writes parquet; it does **not** host a
database. To query from the laptop, read the parquet in place with DuckDB
(`src/data/query.py`). DuckDB pushes filters down to the parquet so it only
reads the columns/row-groups you ask for — fast even over NFS, nothing to run:

```python
from src.data import query
query.load_panel(start="2024-01-01")                       # filtered panel
query.sql("SELECT date, ticker, pred FROM {predictions} "
          "WHERE date >= '2024-01-01' ORDER BY pred DESC")  # ad-hoc SQL
```

A Postgres/MySQL container would be the wrong shape here: the data is
columnar/analytical and written once per batch, so an embedded engine over
parquet beats ETL-ing into a row store. Revisit only if many machines must query
concurrently without mounting.

## 6. Backfill vs. incremental (scheduling)

Differentiate the two run modes — they have very different cost:

| | Backfill | Incremental (`--incremental`) |
|---|---|---|
| When | once / rare full rebuild | scheduled weekly or monthly |
| Prices | all history, all names | only months since the last cached month-end |
| Fundamentals | full EDGAR companyfacts pull | checks each company's latest filing date first; **skips** the heavy pull when nothing changed |
| Cost | ~45-90 min, high query volume | minutes, tiny query volume |

```bash
# One-time (or rare) backfill — run on the NAS, off-hours:
docker compose run --rm build-dataset

# Scheduled update — cheap; add --incremental in Task Scheduler:
docker compose run --rm build-dataset python -m pipelines.build_dataset --incremental
```

Schedule the incremental run after month-end (fundamentals post on filing
cadence; prices settle daily). The incremental price fetch re-pulls only a
small trailing window and merges it into the cached monthly series (atomic
write), and fundamentals re-pull only tickers with a new SEC filing.

## Notes & tradeoffs

- **CPU**: EDGAR fetch is network-bound (fine on NAS); pandas parsing is CPU-ish
  and may be slower than the workstation. Fetch on NAS, heavy modeling on the
  workstation is the usual split.
- **Raw caching (future)**: cache raw EDGAR JSON under `data/raw/edgar/` on the
  NAS so rebuilds run offline without re-hitting SEC. Not yet implemented.
- **Scale path**: if the panel grows to many GB, query parquet in place with
  DuckDB over the mount (filter pushdown, no server) and partition by year.
- **Backups**: RAID is not a backup — enable Snapshot Replication on the
  `factor-data` shared folder.
