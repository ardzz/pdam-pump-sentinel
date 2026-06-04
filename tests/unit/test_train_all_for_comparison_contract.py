import json
from importlib import import_module


def _train_all_for_comparison():
    return import_module('scripts.train_all_for_comparison')


def test_train_all_for_comparison_dry_run_lists_requested_runs_and_tags(capsys):
    script = _train_all_for_comparison()

    result = script.main(['--dry-run', '--families', 'pca_spectral,lstm_ae,xgboost_supervised'])

    payload = json.loads(capsys.readouterr().out)
    assert [item['run_name'] for item in payload['runs']] == ['pca_spectral', 'lstm_ae', 'xgboost_supervised']
    assert [item['run_name'] for item in result] == ['pca_spectral', 'lstm_ae', 'xgboost_supervised']
    tags_by_run = {item['run_name']: item['tags'] for item in payload['runs']}
    assert tags_by_run['pca_spectral'] == {
        'model_family': 'pca',
        'feature_mode': 'spectral',
        'split_strategy': 'unsupervised_novel_fault',
        'dataset': 'skab',
        'report_run': '1',
    }
    assert tags_by_run['lstm_ae']['model_family'] == 'lstm_ae'
    assert tags_by_run['lstm_ae']['feature_mode'] == 'raw'
    assert tags_by_run['xgboost_supervised']['model_family'] == 'xgboost'
    assert tags_by_run['xgboost_supervised']['feature_mode'] == 'supervised_features'
    assert tags_by_run['xgboost_supervised']['split_strategy'] == 'supervised_cross_group'
