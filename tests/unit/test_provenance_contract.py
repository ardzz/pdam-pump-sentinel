import hashlib
from dataclasses import dataclass
from datetime import datetime
from importlib import import_module
from pathlib import Path


def _provenance():
    return import_module('ml.utils.provenance')


@dataclass(frozen=True)
class SyntheticConfig:
    input_path: Path
    threshold: float
    features: tuple[str, ...]


def test_sha256_file_matches_standard_library_digest(tmp_path):
    provenance = _provenance()
    payload = b'skab synthetic payload\n'
    path = tmp_path / 'sensor.csv'
    path.write_bytes(payload)

    assert provenance.sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_collect_provenance_contains_hashes_versions_git_and_utc_timestamp(tmp_path):
    provenance = _provenance()
    first = tmp_path / 'a.csv'
    second = tmp_path / 'b.csv'
    first.write_text('a\n')
    second.write_text('b\n')
    config = SyntheticConfig(input_path=first, threshold=0.95, features=('pressure', 'flow'))

    result = provenance.collect_provenance(config=config, input_files=[second, first])

    parsed_time = datetime.fromisoformat(result['generated_at_utc'].replace('Z', '+00:00'))
    assert parsed_time.tzinfo is not None
    assert result['git_sha']
    assert result['git_dirty'] in {True, False, 'unknown'}
    assert result['git'] == {'sha': result['git_sha'], 'dirty': result['git_dirty']}
    assert result['library_versions']['python']
    assert {'numpy', 'pandas', 'scikit_learn', 'mlflow'} <= set(result['library_versions'])
    assert list(result['dataset_hashes']) == [str(first), str(second)]
    assert result['dataset_hashes'][str(first)] == hashlib.sha256(b'a\n').hexdigest()
    assert result['dataset_hashes'][str(second)] == hashlib.sha256(b'b\n').hexdigest()
    assert len(result['config_hash']) == 64
    assert result['config']['input_path'] == str(first)


def test_collect_provenance_config_hash_is_stable_for_mapping_order(tmp_path):
    provenance = _provenance()
    path = tmp_path / 'data.csv'
    path.write_text('payload\n')

    first = provenance.collect_provenance({'b': 2, 'a': path}, [path])
    second = provenance.collect_provenance({'a': path, 'b': 2}, [path])

    assert first['config_hash'] == second['config_hash']
    assert first['dataset_hashes'] == second['dataset_hashes']


def test_collect_provenance_marks_git_unknown_when_commands_fail(tmp_path, monkeypatch):
    provenance = _provenance()
    path = tmp_path / 'data.csv'
    path.write_text('payload\n')
    monkeypatch.setattr(provenance, '_git_output', lambda args: None)

    result = provenance.collect_provenance({'path': path}, [path])

    assert result['git_sha'] == 'unknown'
    assert result['git_dirty'] == 'unknown'
