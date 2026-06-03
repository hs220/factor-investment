# Plan: Azure Burst Training (offload heavy model training)

> **Status: FALLBACK.** The chosen approach is local — NAS Dagster dispatching to
> the home GPU desktop via remote Docker (`plans/local-training.md`), which keeps
> everything on the LAN and avoids the DB-migration / Blob round-trip below. Keep
> this plan for the day we need cloud scale, always-on availability, or compute
> beyond the desktop.

## Why
The NAS cannot run the LightGBM walk-forward in-process: the fits saturate its
cores and starve Dagster's gRPC heartbeat (run killed after ~73 min, zero
progress), with RAM pressure from the 1.28M-row panel × 279 fits. ElasticNet fits
on the NAS; LightGBM (and any future tuning / torch) does not. We keep Dagster +
Postgres on the NAS as the **orchestrator + warehouse**, and burst the **heavy
training step** to Azure, then bring results back.

## Decision (locked with user)
Dagster (NAS) → **Azure ML command job** on an autoscaling compute cluster, via
**Dagster Pipes**. Azure ML over ACI/Batch because it gives managed burst compute
*and* a model registry that complements our artifact/manifest design. Pay per run;
the cluster scales to zero between runs.

## The networking wrinkle (and the fix)
The warehouse Postgres lives on the home LAN (`192.168.68.70:5433`) and is **not
reachable from Azure**. Rather than tunnel, **decouple compute from the home DB via
Azure Blob** (this is the "gold parquet export" already anticipated in
`architecture.md`):

```
Dagster (NAS)                         Azure
─────────────                         ─────
panel_monthly ──export gold parquet──▶ Blob: gold/panel_monthly.parquet
                                       │
                          Azure ML job ┤ reads parquet, runs train_and_deploy
                                       │ writes: Blob: runs/<ver>/{predictions.parquet,
                                       ▼                         model.joblib, manifest.json}
ingest results ◀──read Blob──────────┘
  → db.load_predictions(predictions)
  → copy model.joblib+manifest into the factor-models volume (inference reads it)
```

No inbound access to home needed; Azure only touches Blob. (Alt considered:
Tailscale tunnel to expose Postgres — rejected for now: makes cloud depend on the
home network being up.)

## Reuse — the training code is already portable
`src/models/training.py::train_and_deploy` is the single training path. The Azure
job is a **thin entrypoint** around it; it differs from the NAS only at the edges
(read parquet from Blob instead of the warehouse; write artifact/predictions to
Blob instead of the volume/DB). No model logic is duplicated.

## Components to build
1. **Gold export** — `src/data/blob.py` (azure-storage-blob) + a Dagster step/asset
   `panel_export` that writes `panel_monthly` to Blob as parquet. (Or export inside
   the Pipes launcher.)
2. **Training entrypoint** — `pipelines/train_azure.py`: read panel parquet from
   Blob → `train_and_deploy(model_name="lightgbm", tune=...)` → write
   `predictions.parquet` + `model.joblib` + `manifest.json` to Blob. Packaged in an
   image pushed to **ACR** (reuse the orchestration image + azure SDK), referenced
   by the Azure ML job.
3. **Dagster Pipes launcher** — replace the in-process `model_predictions` asset with
   one that submits the Azure ML command job (`azure-ai-ml` SDK) through a Pipes
   client, streams logs/metadata back, and waits.
4. **Results ingest** — read the run's Blob outputs → `db.load_predictions(...)` +
   place the artifact in the `factor-models` volume (or have inference read the
   artifact from Blob via the manifest). Emits the OOS metrics as asset metadata.
5. **Asset checks** unchanged (`predictions_row_count`, `predictions_ic_positive`).

## Azure resources (one-time)
- Resource group; **Azure ML workspace** + a CPU **compute cluster** (e.g.
  Standard_F16s_v2, min 0 / max 1 nodes — scales to zero).
- **Storage account + Blob container** (`gold/`, `runs/`).
- **Azure Container Registry** for the training image.
- **Service principal** (or workload identity) for Dagster→Azure auth; creds in the
  NAS `deploy/dagster/.env` (never committed), surfaced as env to the daemon.

## Phasing (prove it cheaply first)
| Phase | Work | Proves |
|---|---|---|
| **0** | Manual: export one `panel_monthly.parquet` to Blob; run `train.py` on a one-off Azure VM (or `az ml job create`) reading it; time it. | Speedup + the Blob round-trip, before any Dagster wiring. |
| **1** | `src/data/blob.py` + `pipelines/train_azure.py` + ACR image. Run the Azure ML job by hand; verify artifact + predictions land in Blob. | The cloud training path end-to-end. |
| **2** | Dagster Pipes `model_predictions` launcher + results-ingest; re-enable the `model_train` schedule (it now dispatches to Azure, NAS just orchestrates). | Hands-off monthly retrain. |

## Interim (until Phase 2)
The deployed `model_predictions` asset can't run on the NAS, and its monthly
schedule is now **STOPPED**. To keep the warehouse + dashboard populated meanwhile,
run training **off-box on the laptop** and load results:
`python -m pipelines.train --model lightgbm` then a small loader that calls
`db.load_predictions(...)` and copies the artifact to the NAS `factor-models`
volume. (Or temporarily point `model_predictions` at ElasticNet for an on-NAS run.)

## Out of scope
GPU/torch (later, same pattern with a GPU compute target), multi-horizon parallel
training, real-time inference. This plan is the burst-training lane only.

## Risks / watch-items
- **Auth/secrets** — service-principal creds on the NAS; least-privilege to the RG.
- **Image drift** — the Azure training image must pin the same lib versions as the
  artifact consumer (joblib/sklearn/lightgbm) to avoid load-time skew.
- **Cost** — cluster min-nodes=0; confirm it scales down; a stuck job shouldn't hold
  a node. Budget alert on the RG.
- **Blob as the contract** — predictions/artifact schemas in Blob must match what
  the ingest + inference expect; version under `runs/<model_version>/`.
