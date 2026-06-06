# 0004. MLflow alias pattern for champion selection

## Status
Accepted

## Context
PDAM Pump Sentinel serves pump anomaly detection from a registered model.
The production service needs a stable way to choose which version is current.
MLflow registry has two model selection mechanisms.
Legacy stages use names such as `Staging`, `Production`, and `Archived`.
Those stages are deprecated in the MLflow 3.x line.
Modern aliases are named pointers on a registered model.
Aliases can be `champion`, `challenger`, `canary`, `champion-staging`, or `champion-prod`.
The team had to pick one production selection mechanism for inference, retraining, seed scripts, and operator work.
This choice affects cold start behavior, promotion flow, audit history, and deployment wiring.
Current code already treats the MLflow registry as the source of truth for champion selection.
Local artifacts and Redis active model metadata remain support paths for development and demos.
This ADR records the choice to use MLflow aliases, with `@champion` as the production pointer.

## Decision Drivers
* Future-proofing matters because MLflow 3.x deprecates registry stages.
* Production deployment should not depend on a hard-coded numeric version.
* A model version can change without changing app configuration.
* Operators need a simple champion and challenger vocabulary.
* Multi-tenant deployments may need scoped aliases, such as `champion-staging` and `champion-prod`.
* Promotion history should stay inside the MLflow registry.
* The app should support startup resolution now and runtime polling later.
* MLflow 3.12 has the alias APIs needed by this workflow.
* The pattern maps cleanly to W&B model aliases, including the common `production` convention.

## Considered Options

### 1. MLflow stages
Use `Production` and `Staging` on MLflow model versions.
This matches older MLflow examples and is easy to explain.
It is not a good long-term choice because stages are deprecated in MLflow 3.x.
It also makes scoped names such as `champion-staging` awkward.

### 2. MLflow aliases
Use aliases on a registered model, following the MLflow registry workflow:
https://mlflow.org/docs/latest/ml/model-registry/workflow/
The app resolves `models:/PumpAD@champion` semantics through the MLflow client.
The retraining lane can promote a challenger by moving the `champion` alias.
This is the chosen option.

### 3. Direct version pinning via environment variable
Set a variable such as `PUMPAD_MODEL_VERSION=17` or bake the version into a deployment manifest.
This is simple for one demo but brittle for a training loop.
Every promotion would need a deployment or configuration change.
The version change would not leave a registry-level audit trail.

### 4. Custom database table
Store the active model pointer in a project-owned table.
The project already writes Redis active model metadata for operational state.
A custom database would duplicate registry behavior and add a second audit path.
This reinvents the wheel.

## Decision
Use MLflow aliases.
`@champion` points to the current production model version.
`@challenger` may point to a candidate model version during evaluation.
Promotion moves the `champion` alias to the accepted version.
Deployments resolve the alias at startup today.
Runtime polling may be added later for alias changes that should take effect without a process restart.
The app keeps local artifact loading as a fallback path, not as the primary production selector.
The model version number stays visible in telemetry and metadata.
It is no longer the deployment selector.

## Implementation Evidence
`ml/registry/mlflow_client.py` exposes the startup loader:

```python
def load_champion_service(
    model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
    alias: str = 'champion',
    local_model_dir: str | None = None,
) -> InferenceService | None:
```

It first calls `_load_service_from_mlflow_alias(model_name, alias)`.
If MLflow cannot resolve the alias, it falls back to local artifacts from `local_model_dir` or `PUMPAD_MODEL_DIR`.
`app/services/inference.py` implements the application cold-start chain:

```text
get_inference_service()
  -> PUMPAD_MODEL_DIR primary load
  -> load_champion_service(model_name='PumpAD', alias='champion') fallback
```

Recent fix `46bc0c3` made that cold-start fallback explicit.
`app/jobs/retraining_job.py` contains the promotion helper currently used by the retraining lane:

```python
def _promote_mlflow_champion_alias(model_name: str = DEFAULT_REGISTERED_MODEL_NAME) -> str | None:
```

It reads the `challenger` alias and writes `champion` with this call shape:

```python
client.set_registered_model_alias(model_name, 'champion', version_text)
```

`ml/registry/mlflow_client.py` uses the MLflow alias setter when training code requests an alias:

```python
client.set_registered_model_alias(registered_model_name, str(alias), str(version))
```

It also has the module-level fallback:

```python
mlflow.set_registered_model_alias(registered_model_name, str(alias), str(version))
```

The effective external API shape is:

```python
set_registered_model_alias(name: str, alias: str, version: str) -> None
```

MLflow 3.12 can return model info where `registered_model_version is None`.
The project works around that by resolving the version from the run id:

```python
client.search_model_versions(filter_string=f"run_id='{run_id}'")
```

Only after that does it fall back to latest versions or `model_info`.
`scripts/seed_initial_models.py` seeds the first PCA champion by setting `PcaTrainingConfig(alias=DEFAULT_ALIAS)` where `DEFAULT_ALIAS = 'champion'`.
It then resolves the MLflow alias version and writes active model metadata to Redis.
Recent fix `a836a0c` repaired the seed Redis initialization path.
`infra/docker-compose.dev.yml` pins MLflow to `ghcr.io/mlflow/mlflow:v3.12.0`.
Recent fix `f4d994c` added that image pin.

## Consequences

### Positive
* Deployment is decoupled from the numeric model version.
* Promotion can move `@champion` without rebuilding the app image.
* The alias vocabulary supports champion and challenger workflows.
* The same pattern can support A/B testing and canary aliases.
* Scoped aliases can support staging, production, and tenant-specific flows.
* MLflow registry events keep an audit trail for alias changes.
* The naming matches W&B model alias usage, including `production`.

### Negative
* The project requires MLflow 3.x alias support.
* The dev image is pinned to `ghcr.io/mlflow/mlflow:v3.12.0`.
* Runtime polling for alias changes is not implemented yet.
* Operators need to learn aliases instead of older stage names.
* The MLflow 3.12 version resolution quirk adds client-side fallback code.

### Neutral
* Hot-swap behavior and alias resolution are separate concerns.
* The current code can hot-swap an already loaded local artifact service.
* Alias resolution occurs during startup unless a polling loop is added.
* The hot-swap mechanism and alias resolver decouple model load from app restart once polling exists.
* Redis active model metadata remains operational state, not the registry source of truth.

## Verification
* Read `ml/registry/mlflow_client.py` for `load_champion_service(...)` and `set_registered_model_alias(...)` calls.
* Read `ml/registry/mlflow_client.py` for the MLflow 3.12 `registered_model_version is None` workaround using `search_model_versions` by `run_id`.
* Read `app/jobs/retraining_job.py` for `_promote_mlflow_champion_alias(...)` and promotion from `challenger` to `champion`.
* Read `app/services/inference.py` for `PUMPAD_MODEL_DIR` primary loading and the `load_champion_service(model_name='PumpAD', alias='champion')` fallback.
* Read `scripts/seed_initial_models.py` for initial `champion` alias seeding and Redis active model metadata.
* Read `infra/docker-compose.dev.yml` for the MLflow image pin `ghcr.io/mlflow/mlflow:v3.12.0`.

## References
* MLflow model registry workflow:
  https://mlflow.org/docs/latest/ml/model-registry/workflow/
* W&B model management:
  https://docs.wandb.ai/guides/core/registry/model-management/
