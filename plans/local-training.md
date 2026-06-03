# Plan: Local Burst Training (NAS Dagster → desktop GPU box via remote Docker)

## Why
The NAS can't run the LightGBM walk-forward (heartbeat starvation / RAM). The home
**Windows desktop (NVIDIA 3070)** is far stronger and — crucially — **on the same
LAN as the warehouse**, so it reaches Postgres (`192.168.68.70:5433`) directly. That
removes Azure's whole networking problem (no DB migration, no Blob round-trip), costs
nothing, and the GPU unlocks the future torch/deep-learning roadmap. Azure stays the
cloud fallback (`plans/azure-training.md`).

## Decision (locked with user)
Keep **Dagster + Postgres on the NAS** as orchestrator + warehouse. The heavy
training step runs in a **Docker container on the desktop**, launched by Dagster via
**`PipesDockerClient` over a remote Docker host** (`DOCKER_HOST=ssh://you@desktop`).
The container reads the NAS Postgres directly and writes results back to it.

```
NAS (Dagster daemon)                         Desktop (Docker, WSL2, 3070)
────────────────────                         ────────────────────────────
model_predictions asset
  PipesDockerClient ──run container (ssh://)─▶ factor-train:latest
     │  (DOCKER_HOST=ssh://you@desktop)         · reads NAS Postgres (LAN)
     │                                          · src.models.training.train_and_deploy
     │  ◀── Pipes msgs via container stdout ──  · writes predictions -> predictions table
     ▼                                          · writes model -> model_registry (DB)
  asset materialized (OOS metrics)              · (GPU available for future torch)
```

Everything crosses machines over the **LAN Postgres** — no shared filesystem, no cloud.

## Artifact transport: a DB-backed model registry
The current artifact lives in a `models/` **filesystem** volume — that doesn't cross
machines (the desktop container can't write the NAS volume cleanly). So add a
**`model_registry` table** as the cross-machine transport:

```
model_registry(model_version PK, horizon, model_name, manifest jsonb,
               artifact bytea, created_at)   -- artifact = the joblib bytes
```
- `src/models/artifact.py` gains a **DB store** alongside the filesystem one:
  `save_artifact(..., store="db")` serializes the pipeline to bytes → upsert;
  `load_artifact(horizon, store="db")` reads + `joblib.loads`. Filesystem store stays
  for local dev. Inference (`predict_with_artifact`, the dashboard) reads from the DB
  registry — so the desktop trains, the NAS serves, both via the LAN DB.
- The `predictions.model_version` key already references this lineage.

## Reuse — training code is unchanged
The container entrypoint is a thin wrapper around the *same*
`src.models.training.train_and_deploy`; only the edges differ (it writes the artifact
to the registry + predictions to the DB). No model logic duplicated. The NAS
`pipelines/train.py` still works in-process for local/off-box runs.

## Components to build
1. **`model_registry`** — `db.ensure_model_registry_table` + DB store in
   `artifact.py`; point `predict_with_artifact` / dashboard at it.
2. **Training image** — `deploy/train/Dockerfile`: CUDA-capable base (GPU-ready for
   torch; LightGBM runs CPU fine), ML deps, `src/`, `dagster-pipes`, and an entrypoint
   `orchestration/train_entrypoint.py` that `open_dagster_pipes()` → `train_and_deploy`
   → `db.load_predictions` + register artifact → `report_asset_materialization`.
3. **Pipes asset** — rewrite `model_predictions` to use
   `PipesDockerClient().run(image="factor-train:latest", command=[...], env={...},
   context=...)`; pass DB creds via env; `container_kwargs` with a GPU
   `DeviceRequest` (no-op for LightGBM, ready for torch). Default message reader
   streams Pipes messages back over container stdout — works remotely, no shared FS.
4. **NAS daemon image** — add `docker` (docker-py) + `docker[ssh]`/paramiko; mount/
   provide an SSH key to the desktop; set `DOCKER_HOST=ssh://you@desktop` (or a Docker
   context) in the daemon env. Training deps (lightgbm/sklearn) can stay only for
   NAS-side **inference**.
5. **Image build/distribution** — a `deploy/train/build.sh` that builds
   `factor-train:latest` on the desktop (ssh + `docker build`, or a small NAS-hosted
   registry the desktop pulls from). PipesDockerClient runs an existing image; it
   doesn't build.

## Desktop one-time setup
- **WSL2 + Docker** (Docker Desktop WSL2 backend, or Docker Engine in WSL2) — gives a
  Linux runtime so the image is byte-identical to dev/NAS.
- **NVIDIA Container Toolkit** in WSL2 (CUDA-on-WSL2 with a recent driver) for
  `--gpus all`. Not needed for LightGBM; needed for the torch phase.
- **OpenSSH Server** on Windows (key-based auth from the NAS); the desktop's Docker
  reachable via that SSH for `DOCKER_HOST=ssh://`.
- **DHCP reservation** (static IP) for the desktop.

## Phasing
| Phase | Work | Proves |
|---|---|---|
| **0** | On the desktop (WSL2), run `FACTOR_DB_HOST=192.168.68.70 python -m pipelines.train --model lightgbm` against the NAS DB. Time it. | Data locality + the desktop's speed, before any Docker/Pipes. |
| **1** | `model_registry` + `deploy/train` image + entrypoint; run the container on the desktop manually; verify `predictions` + `model_registry` populated and the dashboard can load the model. | The containerized training path end-to-end. |
| **2** | NAS Dagster → `PipesDockerClient` (`DOCKER_HOST=ssh://desktop`) as the `model_predictions` asset; re-enable the monthly `model_train` schedule (now dispatches to the desktop); WoL optional. | Hands-off monthly retrain orchestrated from the NAS. |

## Wake-on-LAN (optional, Phase 2+)
The desktop isn't always-on. For a monthly retrain: power it on manually, or have the
asset send a magic packet first (BIOS WoL + Windows NIC "wake on magic packet" +
disable Fast Startup; send from the **NAS host**, not the bridge-networked container).
A Windows Task Scheduler self-wake is the no-Dagster alternative. Not required for v1.

## GPU / future
Same pattern serves the torch roadmap: swap the image's base to CUDA + torch, keep the
`DeviceRequest`, train LSTM/Transformer on the 3070. LightGBM today doesn't use it.

## Risks / watch-items
- **Desktop availability** — off ⇒ dispatch fails; mitigate with WoL or manual power-on
  (monthly cadence makes this minor). If "often asleep" becomes painful, `dagster-celery`
  with the desktop as a queued worker is the escalation (broker + always-on worker).
- **Remote Docker auth** — SSH key NAS→desktop; least-privilege user; `docker[ssh]`
  needs paramiko in the daemon image.
- **Image/version parity** — the training image and the NAS inference env must pin the
  same joblib/sklearn/lightgbm so a registry artifact loads without skew.
- **Registry blob size** — joblib of a Pipeline+LightGBM is a few MB; fine as `bytea`.
  Keep N versions, prune old ones.

## Out of scope (v1)
Always-on auto-wake, multi-worker pools (celery/dask), GPU/torch models, real-time
inference. This is the single-box burst-training lane via remote Docker.
