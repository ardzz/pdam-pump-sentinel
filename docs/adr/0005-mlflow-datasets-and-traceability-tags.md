# MLflow Datasets and Traceability Tags

## Status

Accepted

## Context and Problem Statement

Anomaly detection training in PDAM Pump Sentinel is not deterministic enough to audit from metrics alone.
The same model family can change when the input split changes, feature code changes, package versions change, or the host runtime changes.
Reviewers and future maintainers need to trace any MLflow run back to the exact code commit, dataset digest, package set, Python version, and host.
File paths are not enough evidence because the same path can point to different bytes over time.
A manifest path can also be copied, regenerated, or mounted differently on another host.
The audit record needs a stable digest and a visible dataset card on the run.
The project compares seven training families in one MLflow experiment, so each family needs the same provenance shape.
Those families are `pca_raw`, `pca_spectral`, `pca_enriched`, `lstm_ae`, `isolation_forest`, `xgboost_supervised`, and `lightgbm_supervised`.

## Decision Drivers

1. Reproducibility matters because anomaly metrics can move when the same command sees different data or environment state.
2. Debugging postmortems need enough metadata to explain why a run changed after a training failure, metric regression, or bad promotion.
3. Comparing runs across versions needs a common tag schema, not one off tags per trainer.
4. The MLflow Datasets API is mature enough in MLflow 3.x for dataset cards, schema capture, source capture, and stable digest display.
5. The approach should fit the existing training code without adding a separate service.
6. The project needs evidence that reviewers can inspect in the MLflow UI and in exported run metadata.
7. The policy must apply to all seven comparison families listed above.

## Considered Options

### Option 1. File path tag only

This option stores only a path tag such as `tag.dataset_path=/path/to/csv`.
It is easy to add and easy to read.
It does not compute a digest.
It does not preserve the dataframe schema.
It does not create an MLflow dataset card.
It also cannot prove that a later file at the same path is the same data used by the original run.

### Option 2. MLflow Datasets API plus traceability tags

This option logs dataframe inputs through `mlflow.data.from_pandas(df, source=path, name=...)`.
It then calls `mlflow.log_input(dataset, context='training')`, `mlflow.log_input(dataset, context='validation')`, or `mlflow.log_input(dataset, context='test')`.
MLflow computes a stable dataset digest for the logged input.
The dataset card preserves schema and source details in the run UI.
The project also writes `dataset.manifest_sha256`, `dataset.feature_mode`, and `dataset.split_strategy` tags beside the dataset card.
That gives each run both a manifest level digest and split specific dataset cards.
This is the chosen option.

### Option 3. External provenance store

This option writes dataset and environment lineage into a separate provenance database.
It could support richer lineage queries later.
It is too heavy for this project phase.
It adds another system to deploy, back up, query, and explain during the demo.
It also duplicates metadata that MLflow already attaches to the run.

### Option 4. No provenance

This option keeps metrics, params, and artifacts, but does not record dataset or environment lineage.
It has the lowest per run overhead.
It makes runs opaque.
It would leave reviewers guessing which manifest, commit, host, and dependency set produced a model.
It is not acceptable for reproducibility audits.

## Decision Outcome

Chosen option: Option 2, MLflow Datasets API plus structured traceability tags.
Every training run should log SKAB split data through MLflow Datasets when a dataframe is available.
The dataset source should point to the manifest URI plus the split name.
The dataset name should follow `skab.<split_strategy>.<split_name>`.
The dataset context should be one of `training`, `validation`, or `test`.
Every run should also receive the same traceability tag schema.
The exact traceability tags are:

1. `git.commit.sha`
2. `git.branch`
3. `git.is_dirty`
4. `python.version`
5. `host.name`
6. `package.mlflow`
7. `package.scikit-learn`
8. `package.tensorflow`
9. `package.xgboost`
10. `package.lightgbm`

Dataset cards are also part of the schema.
Each logged dataframe split gets an MLflow dataset card with a stable digest.
Each run also gets `dataset.manifest_sha256` so reviewers can compare runs at manifest scope.
The policy applies to `pca_raw`, `pca_spectral`, `pca_enriched`, `lstm_ae`, `isolation_forest`, `xgboost_supervised`, and `lightgbm_supervised`.

## Consequences

### Positive

1. Runs become reproducible enough for audits because reviewers can inspect the code commit, branch, dirty state, Python version, package versions, host, manifest digest, and dataset card.
2. The MLflow UI shows a dataset card per run, so dataset evidence is visible beside params, metrics, artifacts, signatures, and input examples.
3. The comparison experiment can compare families across the same tag schema instead of ad hoc provenance fields.
4. A metric regression can be traced back to a changed manifest digest, changed feature mode, changed split strategy, changed package version, or changed commit.
5. The approach keeps provenance close to the run record, which makes exported MLflow metadata useful without another database.

### Negative

1. Logging datasets adds slight per run overhead because MLflow computes dataset digests.
2. The implementation depends on MLflow 3.x behavior for dataset cards and model tracking features.
3. Very large dataframes could make input logging more expensive, so trainers should log split dataframes that represent the training evidence, not unrelated raw dumps.

### Neutral

1. The tag schema is standardized across all seven training families.
2. The schema does not replace params or metrics.
3. Dataset cards explain which data was used, while traceability tags explain which code and environment produced the run.

## Verification

The traceability tag helper is `set_run_traceability_tags` in `ml/registry/mlflow_client.py:66-77`.
It calls `_run_traceability_tags`, which builds the exact git, Python, host, and package tag set in `ml/registry/mlflow_client.py:630-646`.
The PCA logger calls `set_run_traceability_tags(mlflow)` in `_log_pca_training_run_to_active_run` at `ml/registry/mlflow_client.py:116-126`.
The LSTM Autoencoder logger calls the same helper in `_log_lstm_ae_training_run_to_active_run` at `ml/registry/mlflow_client.py:392-402`.
The Isolation Forest logger calls the same helper in `_log_isoforest_training_run_to_active_run` at `ml/registry/mlflow_client.py:423-433`.
The dataset helper is `log_skab_inputs_to_active_run` in `ml/registry/mlflow_client.py:179-230`.
It sets `dataset.manifest_sha256`, `dataset.feature_mode`, and `dataset.split_strategy` in `ml/registry/mlflow_client.py:205-214`.
It creates datasets with `mlflow_data.from_pandas(...)` in `ml/registry/mlflow_client.py:221-227`.
It logs inputs with `mlflow.log_input(dataset, context=context)` in `ml/registry/mlflow_client.py:228`.
The stable manifest digest helper is `_manifest_sha256` in `ml/registry/mlflow_client.py:675-681`.
The split source URI helper is `_dataset_source_uri` in `ml/registry/mlflow_client.py:684-687`.
The comparison experiment name is quoted in `scripts/train_all_for_comparison.py:26` as `DEFAULT_EXPERIMENT_NAME = 'pump_sentinel_model_comparison'`.
The comparison run tag pattern is quoted in `ComparisonPlan.tags` at `scripts/train_all_for_comparison.py:46-53` as `report_run: '1'`.
The unique run name lookup searches prior report runs with `tags.report_run = '1'` in `scripts/train_all_for_comparison.py:274-276`.
The sprint Done section records SKAB dataset cards through `mlflow.data` and run traceability tags in `docs/plans/sprint-remaining.md:31-33`.
That Done section also lists recent commits `16a1ba8` for datasets and `47126ba` for traceability tags plus signatures.

## References

1. MLflow Datasets API documentation: <https://mlflow.org/docs/latest/ml/dataset/>
2. MLflow Tracking quickstart: <https://mlflow.org/docs/latest/ml/tracking/quickstart/>
