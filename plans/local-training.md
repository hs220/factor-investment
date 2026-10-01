# Plan: Local Burst Training (NAS Dagster → desktop GPU box via remote Docker)

## Why
The NAS can't run the LightGBM walk-forward (heartbeat starvation / RAM). The home
**Windows desktop (NVIDIA 3070)** is far stronger and — crucially — **on the same
LAN as the warehouse**, so it reaches Postgres (`192.168.68.70:5433`) directly. That
removes Azure's whole networking problem (no DB migration, no Blob round-trip), costs
nothing, and the GPU unlocks the future torch/deep-learning roadmap. Azure stays the
cloud fallback (`plans/azure-training.md`).

## Update 2026-09-30 — training host is now a Linux box, Phase 0 passed
The burst-training host is a dedicated **Linux mini PC** (`192.168.68.76`, HP ProDesk
600 G4 DM, i5-8500T 6c, 16 GB, Debian 13, Docker 29) instead of the Windows desktop —
so the WSL2 / Windows OpenSSH / Fast-Startup steps below no longer apply (native Docker,
SSH key auth already works). It has **no NVIDIA GPU**: the torch phase still needs the
3070 or Azure. The box also hosts another project whose firewall blocks all traffic to
the NAS; a single ufw exception allows `tcp 192.168.68.70:5433` (Postgres) only.

**Phase 0 result** (`python -m pipelines.train --model lightgbm`, nested tuning on,
against the NAS warehouse): **42m56s wall**, 14,351 s CPU (~5.6 of 6 cores busy),
**peak RSS 1.86 GB**, CPU ≤ 76 °C at ~30 W. 279 OOS months, mean IC 0.0286, IC IR 0.412,
t-stat 6.89, hit rate 70.6%, re-tuned 24×. Deployment artifact ~0.9 MB (joblib) — fine
as a `model_registry` `bytea`. The NAS never finished this job (killed after ~73 min).
Caveat: the first attempt hard-powered-off the box ~1 min in with nothing logged; CPU and
AVX2 stress tests plus the 43-min rerun were then clean — watch for recurrence.

**Phase 1 result (2026-09-30, commit `7067941`) — done.** `model_registry` table +
`store="db"` in `src/models/artifact.py` (`$FACTOR_MODEL_STORE`, default fs);
`orchestration/train_entrypoint.py` (Pipes-aware, plain stdout when run by hand);
`deploy/train/{Dockerfile,build.sh,run.sh}`; `constraints-ml.txt` pins the ML stack to
the NAS image's versions (py3.12, sklearn 1.9.0, lightgbm 4.6.0, joblib 1.5.3, numpy
2.4.6, pandas 3.0.3) and both images install with it. Containerized run on the box:
**43m48s**, exit 0, registered `lightgbm-1m-20261001032715` (0.94 MB) and 1,130,640
`predictions` rows (2003-03..2026-06); metrics identical to the Phase-0 venv run. The
**NAS `factor-dagster` container unpickled the registry artifact and scored the
2026-06-30 cross-section** (4,038 names), and `latest_recommendations(store="db")`
returns the same scores — so serving off the registry works. The container runs with
`--network host` so the ufw D12 rules still apply (bridge traffic would bypass them).
Not yet done (Phase 2): the in-process `model_predictions` asset still trains on the NAS
with the fs store; the dashboard/NAS must set `FACTOR_MODEL_STORE=db` when deployed.

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
- **Wake-on-LAN** (see Phase 0 below): BIOS WoL + Windows NIC "wake on magic packet" +
  disable Fast Startup. Done now so the box can be woken on demand later.

## Phasing
| Phase | Work | Proves |
|---|---|---|
| **0** | (a) On the desktop (WSL2), run `FACTOR_DB_HOST=192.168.68.70 python -m pipelines.train --model lightgbm` against the NAS DB; time it. (b) Configure **Wake-on-LAN** and verify a magic packet from the NAS wakes the sleeping desktop. | Data locality + the desktop's speed, and the remote-wake path — before any Docker/Pipes. |
| **1** | `model_registry` + `deploy/train` image + entrypoint; run the container on the desktop manually; verify `predictions` + `model_registry` populated and the dashboard can load the model. | The containerized training path end-to-end. |
| **2** | NAS Dagster → `PipesDockerClient` (`DOCKER_HOST=ssh://desktop`) as the `model_predictions` asset; re-enable the monthly `model_train` schedule (now dispatches to the desktop); WoL optional. | Hands-off monthly retrain orchestrated from the NAS. |

## Wake-on-LAN (set up + verified in Phase 0; automated in Phase 2)
The desktop isn't always-on, so the NAS must be able to wake it before a job. Set this
up in Phase 0 while configuring the desktop, and verify it independently of training.

**Desktop (one-time):**
1. BIOS/UEFI → Power → enable *Wake on LAN* / *Power On by PCIE*.
2. Windows → Device Manager → Ethernet adapter → Properties → **Power Management**:
   "Allow this device to wake the computer" + "Only allow a magic packet…"; **Advanced**
   tab → "Wake on Magic Packet" = Enabled.
3. **Disable Fast Startup** (Power Options → "Choose what the power buttons do" → uncheck
   *Turn on fast startup*) — hybrid shutdown otherwise breaks WoL from a full power-off.
4. Note the NIC **MAC**; give it a DHCP reservation (static IP).

**Verify from the NAS (the Phase-0 wake test):** put the desktop to sleep, then send a
magic packet and confirm it wakes:
```python
import socket
mac = "AA:BB:CC:DD:EE:FF"                      # the desktop NIC
pkt = bytes.fromhex("ff" * 6 + mac.replace(":", "") * 16)
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
s.sendto(pkt, ("255.255.255.255", 9))          # LAN broadcast, port 9
```
(or `wakeonlan AA:BB:...`). Success = the desktop powers on and SSH (port 22) answers.

**Phase 2 automation:** the `model_predictions` asset sends the magic packet, polls
until SSH answers, dispatches the container, and can sleep the box afterward. Send the
packet from the **NAS host**, not the bridge-networked Dagster container (a bridged
container may not reach the LAN broadcast) — e.g. the asset SSHes to the NAS host to run
the sender, or the daemon runs with host networking. Across VLANs WoL needs router
config; on one flat LAN it just works. (A Windows Task Scheduler self-wake is the
no-Dagster fallback.)

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
