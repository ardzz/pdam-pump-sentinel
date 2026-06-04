import hashlib
import sys
import types
from importlib import import_module
from types import SimpleNamespace
from typing import Any, cast


def _mlflow_client():
    return import_module('ml.registry.mlflow_client')


def _pandas():
    return import_module('pandas')


def test_log_skab_inputs_to_active_run_logs_dataset_cards_and_provenance_tags(monkeypatch, tmp_path):
    calls: dict[str, list[Any]] = {'inputs': [], 'tags': []}
    mlflow = types.ModuleType('mlflow')
    mlflow_data = types.ModuleType('mlflow.data')
    mlflow_any = cast(Any, mlflow)
    data_any = cast(Any, mlflow_data)

    def from_pandas(frame, *, source, name, targets=None):
        return SimpleNamespace(frame=frame.copy(), source=source, name=name, targets=targets)

    mlflow_any.active_run = lambda: object()
    mlflow_any.set_tags = lambda tags: calls['tags'].append(dict(tags))
    mlflow_any.log_input = lambda dataset, *, context: calls['inputs'].append((dataset, context))
    data_any.from_pandas = from_pandas
    mlflow_any.data = mlflow_data
    monkeypatch.setitem(sys.modules, 'mlflow', mlflow)
    monkeypatch.setitem(sys.modules, 'mlflow.data', mlflow_data)

    pandas = _pandas()
    train_df = pandas.DataFrame({'feature_0': [0.1, 0.2], 'label': [0, 1], 'changepoint': [0, 0]})
    val_df = pandas.DataFrame({'feature_0': [0.3], 'label': [0], 'changepoint': [1]})
    test_df = pandas.DataFrame({'feature_0': [0.4], 'label': [1], 'changepoint': [0]})
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text('{"train":["a.csv"],"validation":["b.csv"],"test":["c.csv"]}\n')

    _mlflow_client().log_skab_inputs_to_active_run(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        manifest_path=manifest_path,
        manifest_dict={'train': ['a.csv'], 'validation': ['b.csv'], 'test': ['c.csv']},
        feature_mode='enriched',
        split_strategy='supervised_cross_group',
    )

    assert len(calls['inputs']) == 3
    assert {context for _dataset, context in calls['inputs']} == {'training', 'validation', 'test'}
    assert all(len(dataset.frame) > 0 for dataset, _context in calls['inputs'])
    assert [dataset.targets for dataset, _context in calls['inputs']] == ['label', 'label', 'label']
    assert {dataset.name for dataset, _context in calls['inputs']} == {
        'skab.supervised_cross_group.train',
        'skab.supervised_cross_group.validation',
        'skab.supervised_cross_group.test',
    }
    tags = calls['tags'][0]
    assert tags['dataset.manifest_sha256'] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert tags['dataset.feature_mode'] == 'enriched'
    assert tags['dataset.split_strategy'] == 'supervised_cross_group'


def test_log_skab_inputs_to_active_run_skips_absent_validation(monkeypatch, tmp_path):
    calls = []
    mlflow = types.ModuleType('mlflow')
    mlflow_data = types.ModuleType('mlflow.data')
    mlflow_any = cast(Any, mlflow)
    data_any = cast(Any, mlflow_data)

    mlflow_any.active_run = lambda: object()
    mlflow_any.set_tags = lambda tags: None
    mlflow_any.log_input = lambda dataset, *, context: calls.append(context)
    data_any.from_pandas = lambda frame, *, source, name, targets=None: SimpleNamespace(frame=frame, source=source, name=name)
    mlflow_any.data = mlflow_data
    monkeypatch.setitem(sys.modules, 'mlflow', mlflow)
    monkeypatch.setitem(sys.modules, 'mlflow.data', mlflow_data)
    pandas = _pandas()
    frame = pandas.DataFrame({'feature_0': [1.0], 'label': [0], 'changepoint': [0]})

    _mlflow_client().log_skab_inputs_to_active_run(
        train_df=frame,
        val_df=None,
        test_df=frame,
        manifest_path=tmp_path / 'manifest.json',
        manifest_dict={},
        feature_mode='raw',
        split_strategy='unsupervised_novel_fault',
    )

    assert set(calls) == {'training', 'test'}
