from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.evaluation.metrics import composite_f_score, evaluate_split, event_precision_recall  # noqa: E402
from ml.training import train_supervised as supervised_training  # noqa: E402
from ml.training.train_isoforest import IsoForestTrainingConfig, train_isoforest_from_skab  # noqa: E402
from ml.training.train_lstm_ae import LstmAeTrainingConfig, train_lstm_ae_from_skab  # noqa: E402
from ml.training.train_pca import PcaTrainingConfig, train_pca_from_skab  # noqa: E402
from ml.training.train_supervised import SupervisedTrainingConfig  # noqa: E402

DEFAULT_EXPERIMENT_NAME = 'pump_sentinel_model_comparison'
DEFAULT_TRACKING_URI = 'http://localhost:5000'
DEFAULT_OUTPUT_ROOT = Path(os.getenv('PUMP_SENTINEL_COMPARISON_OUTPUT_DIR', '/tmp/pdam-pump-sentinel-model-comparison'))
UNSUPERVISED_MANIFEST = _PROJECT_ROOT / 'data' / 'skab_split_manifest.json'
SUPERVISED_MANIFEST = _PROJECT_ROOT / 'data' / 'skab_supervised_manifest.json'


@dataclass(frozen=True)
class ComparisonPlan:
    key: str
    run_name: str
    model_family: str
    feature_mode: str
    split_strategy: str
    config_factory: Callable[[Path], Any]
    trainer: Callable[[Any], Any]
    uses_active_mlflow_logging: bool = False
    needs_boosting_curves: bool = False

    @property
    def tags(self) -> dict[str, str]:
        return {
            'model_family': self.model_family,
            'feature_mode': self.feature_mode,
            'split_strategy': self.split_strategy,
            'dataset': 'skab',
            'report_run': '1',
        }

    def dry_run_payload(self) -> dict[str, Any]:
        return {'key': self.key, 'run_name': self.run_name, 'tags': self.tags}


@dataclass(frozen=True)
class RunSummary:
    run_name: str
    run_id: str
    tags: dict[str, str]
    metric_keys: list[str]


def main(argv: Sequence[str] | None = None) -> list[dict[str, Any]] | list[RunSummary]:
    parser = _parser()
    args = parser.parse_args(argv)
    plans = _selected_plans(parser, args.families)

    if args.dry_run:
        payload = {
            'dry_run': True,
            'experiment_name': args.experiment_name,
            'tracking_uri': args.mlflow_tracking_uri,
            'runs': [plan.dry_run_payload() for plan in plans],
        }
        print(json.dumps(payload, sort_keys=True))
        return payload['runs']

    mlflow, client, experiment_id = _configure_mlflow(args.mlflow_tracking_uri, args.experiment_name)
    summaries = [_run_plan(mlflow, client, experiment_id, plan, args.output_root) for plan in plans]
    print(json.dumps({'experiment_name': args.experiment_name, 'runs': [_jsonable_summary(item) for item in summaries]}, sort_keys=True))
    return summaries


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Train all SKAB surrogate model families into one MLflow experiment for report comparison.',
        epilog='Supervised in-distribution chrono 80/20 is intentionally parked; this script uses cross-group supervised runs.',
    )
    parser.add_argument('--mlflow-tracking-uri', default=os.getenv('MLFLOW_TRACKING_URI', DEFAULT_TRACKING_URI))
    parser.add_argument('--experiment-name', default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument('--families', default='all')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def _plans() -> dict[str, ComparisonPlan]:
    return {
        'pca_raw': ComparisonPlan(
            key='pca_raw',
            run_name='pca_raw',
            model_family='pca',
            feature_mode='raw',
            split_strategy='unsupervised_novel_fault',
            config_factory=lambda output_dir: PcaTrainingConfig(
                input_path=output_dir,
                output_dir=output_dir,
                split_manifest_path=UNSUPERVISED_MANIFEST,
                feature_mode='raw',
                log_mlflow=False,
            ),
            trainer=train_pca_from_skab,
        ),
        'pca_spectral': ComparisonPlan(
            key='pca_spectral',
            run_name='pca_spectral',
            model_family='pca',
            feature_mode='spectral',
            split_strategy='unsupervised_novel_fault',
            config_factory=lambda output_dir: PcaTrainingConfig(
                input_path=output_dir,
                output_dir=output_dir,
                split_manifest_path=UNSUPERVISED_MANIFEST,
                feature_mode='spectral',
                log_mlflow=False,
            ),
            trainer=train_pca_from_skab,
        ),
        'pca_enriched': ComparisonPlan(
            key='pca_enriched',
            run_name='pca_enriched',
            model_family='pca',
            feature_mode='enriched',
            split_strategy='unsupervised_novel_fault',
            config_factory=lambda output_dir: PcaTrainingConfig(
                input_path=output_dir,
                output_dir=output_dir,
                split_manifest_path=UNSUPERVISED_MANIFEST,
                feature_mode='enriched',
                log_mlflow=False,
            ),
            trainer=train_pca_from_skab,
        ),
        'lstm_ae': ComparisonPlan(
            key='lstm_ae',
            run_name='lstm_ae',
            model_family='lstm_ae',
            feature_mode='raw',
            split_strategy='unsupervised_novel_fault',
            config_factory=lambda output_dir: LstmAeTrainingConfig(
                input_path=output_dir,
                output_dir=output_dir,
                split_manifest_path=UNSUPERVISED_MANIFEST,
                log_mlflow=True,
                epochs=30,
            ),
            trainer=train_lstm_ae_from_skab,
            uses_active_mlflow_logging=True,
        ),
        'isolation_forest': ComparisonPlan(
            key='isolation_forest',
            run_name='isolation_forest',
            model_family='isolation_forest',
            feature_mode='spectral',
            split_strategy='unsupervised_novel_fault',
            config_factory=lambda output_dir: IsoForestTrainingConfig(
                input_path=output_dir,
                output_dir=output_dir,
                split_manifest_path=UNSUPERVISED_MANIFEST,
                feature_mode='spectral',
                log_mlflow=False,
            ),
            trainer=train_isoforest_from_skab,
        ),
        'xgboost_supervised': ComparisonPlan(
            key='xgboost_supervised',
            run_name='xgboost_supervised',
            model_family='xgboost',
            feature_mode='supervised_features',
            split_strategy='supervised_cross_group',
            config_factory=lambda output_dir: SupervisedTrainingConfig(
                input_path=output_dir,
                output_dir=output_dir,
                split_manifest_path=SUPERVISED_MANIFEST,
                model_type='xgboost',
                log_mlflow=False,
            ),
            trainer=_train_supervised_with_model,
            needs_boosting_curves=True,
        ),
        'lightgbm_supervised': ComparisonPlan(
            key='lightgbm_supervised',
            run_name='lightgbm_supervised',
            model_family='lightgbm',
            feature_mode='supervised_features',
            split_strategy='supervised_cross_group',
            config_factory=lambda output_dir: SupervisedTrainingConfig(
                input_path=output_dir,
                output_dir=output_dir,
                split_manifest_path=SUPERVISED_MANIFEST,
                model_type='lightgbm',
                log_mlflow=False,
            ),
            trainer=_train_supervised_with_model,
            needs_boosting_curves=True,
        ),
    }


def _selected_plans(parser: argparse.ArgumentParser, family_arg: str) -> list[ComparisonPlan]:
    plans = _plans()
    if family_arg == 'all':
        return list(plans.values())
    keys = [item.strip() for item in family_arg.split(',') if item.strip()]
    unknown = sorted(set(keys) - set(plans))
    if unknown:
        parser.error(f'unknown families: {", ".join(unknown)}')
    return [plans[key] for key in keys]


def _configure_mlflow(tracking_uri: str, experiment_name: str) -> tuple[Any, Any, str]:
    mlflow = import_module('mlflow')
    mlflow_tracking = import_module('mlflow.tracking')
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_experiment(experiment_name)
    client = mlflow_tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f'MLflow experiment was not created: {experiment_name}')
    return mlflow, client, str(experiment.experiment_id)


def _run_plan(mlflow: Any, client: Any, experiment_id: str, plan: ComparisonPlan, output_root: Path) -> RunSummary:
    run_name = _unique_run_name(client, experiment_id, plan.run_name)
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    config = plan.config_factory(output_dir)

    with mlflow.start_run(run_name=run_name) as active_run:
        run_id = str(active_run.info.run_id)
        mlflow.set_tags(plan.tags)
        result = plan.trainer(config)
        model = None
        if isinstance(result, tuple):
            result, model = result
        if model is not None and plan.needs_boosting_curves:
            supervised_training._log_boosting_eval_curves(mlflow, model, config)
        metrics = _comparison_metrics(result)
        if metrics:
            mlflow.log_metrics(metrics)
        _log_missing_params(mlflow, client, run_id, _params(config, plan.key))
        if not plan.uses_active_mlflow_logging:
            mlflow.log_artifacts(str(result.output_dir))
        return RunSummary(run_name=run_name, run_id=run_id, tags=plan.tags, metric_keys=sorted(metrics))


def _train_supervised_with_model(config: SupervisedTrainingConfig) -> tuple[Any, Any]:
    normalized = supervised_training._normalize_config(config)
    return supervised_training._fit_and_write_artifacts(normalized)


def _unique_run_name(client: Any, experiment_id: str, base_name: str) -> str:
    runs = client.search_runs([experiment_id], filter_string="tags.report_run = '1'")
    existing = {run.info.run_name for run in runs}
    if base_name not in existing:
        return base_name
    suffix = 2
    while f'{base_name}_{suffix}' in existing:
        suffix += 1
    return f'{base_name}_{suffix}'


def _comparison_metrics(result: Any) -> dict[str, float]:
    frame = _scores_frame(result)
    labels = frame['label'].to_numpy(dtype=int)
    predictions = frame['prediction'].to_numpy(dtype=int)
    scores = frame['score'].to_numpy(dtype=float)
    transient_mask = frame['changepoint'].to_numpy(dtype=int) if 'changepoint' in frame.columns else None
    split_metrics = evaluate_split(labels, predictions, scores, transient_mask=transient_mask)
    composite_metrics = composite_f_score(labels, predictions)
    event_metrics = event_precision_recall(labels, predictions)
    raw_metrics = {
        'f1': split_metrics.get('f1'),
        'precision': split_metrics.get('precision'),
        'recall': split_metrics.get('recall'),
        'false_alarm_rate': split_metrics.get('false_alarm_rate'),
        'auc': split_metrics.get('roc_auc'),
        'pr_auc': split_metrics.get('pr_auc'),
        'roc_auc': split_metrics.get('roc_auc'),
        'composite_f_score': composite_metrics.get('composite_f1'),
        'event_precision': event_metrics.get('event_precision'),
        'event_recall': event_metrics.get('event_recall'),
        'event_f1': event_metrics.get('event_f1'),
    }
    metrics: dict[str, float] = {}
    for name, value in raw_metrics.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        metric = float(value)
        if np.isfinite(metric):
            metrics[name] = metric
    return metrics


def _scores_frame(result: Any) -> Any:
    pandas = import_module('pandas')
    artifact_paths = getattr(result, 'artifact_paths')
    path = artifact_paths.get('test_scores') or artifact_paths.get('scores')
    if path is None:
        raise ValueError('training result does not include scores for comparison metrics')
    return pandas.read_csv(path)


def _params(config: Any, plan_key: str) -> dict[str, str | int | float | bool]:
    payload = asdict(config)
    payload['comparison_run_key'] = plan_key
    return {name: value for name, value in ((_param_key(k), _param_value(v)) for k, v in payload.items()) if value is not None}


def _log_missing_params(mlflow: Any, client: Any, run_id: str, params: dict[str, str | int | float | bool]) -> None:
    existing = client.get_run(run_id).data.params
    missing = {name: value for name, value in params.items() if name not in existing}
    if missing:
        mlflow.log_params(missing)


def _param_key(value: str) -> str:
    return value.replace('_path', '_path')


def _param_value(value: Any) -> str | int | float | bool | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool | int | float | str):
        return value
    return None


def _jsonable_summary(summary: RunSummary) -> dict[str, Any]:
    return {
        'metric_keys': summary.metric_keys,
        'run_id': summary.run_id,
        'run_name': summary.run_name,
        'tags': summary.tags,
    }


if __name__ == '__main__':
    main()
