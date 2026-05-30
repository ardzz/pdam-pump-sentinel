from importlib import import_module
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).resolve().parents[1] / 'fixtures'


def _skab_manifest():
    return import_module('ml.datasets.skab_manifest')


def _write_manifest(tmp_path: Path, content: dict) -> Path:
    import json

    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(content))
    return manifest_path


class TestLoadSkabSplitManifest:
    def test_requires_train_validation_test_keys(self, tmp_path):
        manifest = _skab_manifest()
        manifest_path = _write_manifest(tmp_path, {})

        with pytest.raises(ValueError, match='train'):
            manifest.load_skab_split_manifest(manifest_path)

    def test_requires_validation_key(self, tmp_path):
        manifest = _skab_manifest()
        manifest_path = _write_manifest(tmp_path, {'train': []})

        with pytest.raises(ValueError, match='validation'):
            manifest.load_skab_split_manifest(manifest_path)

    def test_requires_test_key(self, tmp_path):
        manifest = _skab_manifest()
        manifest_path = _write_manifest(tmp_path, {'train': [], 'validation': []})

        with pytest.raises(ValueError, match='test'):
            manifest.load_skab_split_manifest(manifest_path)

    def test_train_must_be_non_empty(self, tmp_path):
        manifest = _skab_manifest()
        manifest_path = _write_manifest(tmp_path, {'train': [], 'validation': ['a.csv'], 'test': ['b.csv']})

        with pytest.raises(ValueError, match='train'):
            manifest.load_skab_split_manifest(manifest_path)

    def test_validation_must_be_non_empty(self, tmp_path):
        manifest = _skab_manifest()
        manifest_path = _write_manifest(
            tmp_path, {'train': ['a.csv'], 'validation': [], 'test': ['b.csv']}
        )

        with pytest.raises(ValueError, match='validation'):
            manifest.load_skab_split_manifest(manifest_path)

    def test_test_may_be_empty(self, tmp_path):
        manifest = _skab_manifest()
        manifest_path = _write_manifest(tmp_path, {'train': ['a.csv'], 'validation': ['b.csv'], 'test': []})
        (tmp_path / 'a.csv').write_text('')
        (tmp_path / 'b.csv').write_text('')

        result = manifest.load_skab_split_manifest(manifest_path)

        assert result.test == []

    def test_resolves_paths_relative_to_manifest_directory(self, tmp_path):
        manifest = _skab_manifest()
        data_dir = tmp_path / 'data'
        data_dir.mkdir()
        (data_dir / 'train.csv').write_text('')
        (data_dir / 'val.csv').write_text('')
        manifest_path = _write_manifest(
            tmp_path, {'train': ['data/train.csv'], 'validation': ['data/val.csv'], 'test': []}
        )

        result = manifest.load_skab_split_manifest(manifest_path)

        assert result.train == [data_dir / 'train.csv']
        assert result.validation == [data_dir / 'val.csv']

    def test_loads_old_simple_manifest_without_metadata(self, tmp_path):
        manifest = _skab_manifest()
        (tmp_path / 'train.csv').write_text('')
        (tmp_path / 'val.csv').write_text('')
        manifest_path = _write_manifest(
            tmp_path, {'train': ['train.csv'], 'validation': ['val.csv'], 'test': []}
        )

        result = manifest.load_skab_split_manifest(manifest_path)

        assert result.train == [tmp_path / 'train.csv']
        assert result.validation == [tmp_path / 'val.csv']
        assert result._base_dir == tmp_path.resolve()
        assert result.schema_version is None
        assert result.dataset_kind is None
        assert result.name is None
        assert result.base_dir is None
        assert result.notes is None

    def test_accepts_optional_metadata_fields(self, tmp_path):
        manifest = _skab_manifest()
        (tmp_path / 'train.csv').write_text('')
        (tmp_path / 'val.csv').write_text('')
        manifest_path = _write_manifest(
            tmp_path,
            {
                'schema_version': 1,
                'dataset_kind': 'skab-demo',
                'name': 'Tiny SKAB Demo',
                'notes': 'contract-test fixture only',
                'train': ['train.csv'],
                'validation': ['val.csv'],
                'test': [],
            },
        )

        result = manifest.load_skab_split_manifest(manifest_path)

        assert result.schema_version == 1
        assert result.dataset_kind == 'skab-demo'
        assert result.name == 'Tiny SKAB Demo'
        assert result.notes == 'contract-test fixture only'
        assert result.base_dir is None

    def test_resolves_entries_relative_to_relative_base_dir(self, tmp_path):
        manifest = _skab_manifest()
        manifest_dir = tmp_path / 'manifests'
        data_dir = manifest_dir / 'demo-data'
        (data_dir / 'nested').mkdir(parents=True)
        (data_dir / 'train.csv').write_text('')
        (data_dir / 'nested' / 'val.csv').write_text('')
        manifest_path = _write_manifest(
            manifest_dir,
            {
                'base_dir': 'demo-data',
                'train': ['train.csv'],
                'validation': ['nested/val.csv'],
                'test': [],
            },
        )

        result = manifest.load_skab_split_manifest(manifest_path)

        assert result._base_dir == data_dir.resolve()
        assert result.base_dir == 'demo-data'
        assert result.train == [data_dir.resolve() / 'train.csv']
        assert result.validation == [data_dir.resolve() / 'nested' / 'val.csv']
        assert result.to_payload() == {
            'train': ['train.csv'],
            'validation': ['nested/val.csv'],
            'test': [],
        }

    def test_rejects_absolute_base_dir(self, tmp_path):
        manifest = _skab_manifest()
        manifest_path = _write_manifest(
            tmp_path,
            {
                'base_dir': str(tmp_path.resolve()),
                'train': ['train.csv'],
                'validation': ['val.csv'],
                'test': [],
            },
        )

        with pytest.raises(ValueError, match='base_dir.*relative'):
            manifest.load_skab_split_manifest(manifest_path)

    def test_rejects_base_dir_parent_escape(self, tmp_path):
        manifest = _skab_manifest()
        manifest_path = _write_manifest(
            tmp_path / 'manifests',
            {
                'base_dir': '../outside-data',
                'train': ['train.csv'],
                'validation': ['val.csv'],
                'test': [],
            },
        )

        with pytest.raises(ValueError, match='base_dir.*within manifest directory'):
            manifest.load_skab_split_manifest(manifest_path)

    def test_rejects_split_entry_escape_from_relative_base_dir(self, tmp_path):
        manifest = _skab_manifest()
        data_dir = tmp_path / 'data'
        data_dir.mkdir()
        (data_dir / 'val.csv').write_text('')
        (tmp_path / 'outside.csv').write_text('')
        manifest_path = _write_manifest(
            tmp_path,
            {
                'base_dir': 'data',
                'train': ['../outside.csv'],
                'validation': ['val.csv'],
                'test': [],
            },
        )

        with pytest.raises(ValueError, match='within manifest directory'):
            manifest.load_skab_split_manifest(manifest_path)

    def test_reports_missing_files_under_relative_base_dir(self, tmp_path):
        manifest = _skab_manifest()
        data_dir = tmp_path / 'data'
        data_dir.mkdir()
        (data_dir / 'val.csv').write_text('')
        manifest_path = _write_manifest(
            tmp_path,
            {
                'base_dir': 'data',
                'train': ['missing.csv'],
                'validation': ['val.csv'],
                'test': [],
            },
        )

        with pytest.raises(ValueError, match='missing file referenced in manifest'):
            manifest.load_skab_split_manifest(manifest_path)

    def test_rejects_duplicate_files_with_relative_base_dir(self, tmp_path):
        manifest = _skab_manifest()
        data_dir = tmp_path / 'data'
        data_dir.mkdir()
        (data_dir / 'shared.csv').write_text('')
        manifest_path = _write_manifest(
            tmp_path,
            {
                'base_dir': 'data',
                'train': ['shared.csv'],
                'validation': ['nested/../shared.csv'],
                'test': [],
            },
        )

        with pytest.raises(ValueError, match='duplicate'):
            manifest.load_skab_split_manifest(manifest_path)

    def test_rejects_duplicate_files_across_splits(self, tmp_path):
        manifest = _skab_manifest()
        (tmp_path / 'shared.csv').write_text('')
        manifest_path = _write_manifest(
            tmp_path,
            {'train': ['shared.csv'], 'validation': ['shared.csv'], 'test': []},
        )

        with pytest.raises(ValueError, match='duplicate'):
            manifest.load_skab_split_manifest(manifest_path)

    def test_rejects_duplicate_files_between_train_and_test(self, tmp_path):
        manifest = _skab_manifest()
        (tmp_path / 'shared.csv').write_text('')
        manifest_path = _write_manifest(
            tmp_path, {'train': ['shared.csv'], 'validation': ['a.csv'], 'test': ['shared.csv']}
        )
        (tmp_path / 'a.csv').write_text('')

        with pytest.raises(ValueError, match='duplicate'):
            manifest.load_skab_split_manifest(manifest_path)

    def test_rejects_missing_files(self, tmp_path):
        manifest = _skab_manifest()
        manifest_path = _write_manifest(
            tmp_path, {'train': ['missing.csv'], 'validation': ['a.csv'], 'test': []}
        )
        (tmp_path / 'a.csv').write_text('')

        with pytest.raises(ValueError, match='missing'):
            manifest.load_skab_split_manifest(manifest_path)

    def test_rejects_absolute_paths(self, tmp_path):
        manifest = _skab_manifest()
        outside = tmp_path.parent / 'outside.csv'
        outside.write_text('')
        manifest_path = _write_manifest(
            tmp_path, {'train': [str(outside)], 'validation': ['a.csv'], 'test': []}
        )
        (tmp_path / 'a.csv').write_text('')

        with pytest.raises(ValueError, match='relative'):
            manifest.load_skab_split_manifest(manifest_path)

    def test_rejects_parent_directory_escape(self, tmp_path):
        manifest = _skab_manifest()
        outside = tmp_path.parent / 'outside.csv'
        outside.write_text('')
        manifest_path = _write_manifest(
            tmp_path, {'train': ['../outside.csv'], 'validation': ['a.csv'], 'test': []}
        )
        (tmp_path / 'a.csv').write_text('')

        with pytest.raises(ValueError, match='within manifest directory'):
            manifest.load_skab_split_manifest(manifest_path)

    def test_rejects_non_string_entries(self, tmp_path):
        manifest = _skab_manifest()
        manifest_path = _write_manifest(tmp_path, {'train': [1], 'validation': ['a.csv'], 'test': []})
        (tmp_path / 'a.csv').write_text('')

        with pytest.raises(ValueError, match='strings'):
            manifest.load_skab_split_manifest(manifest_path)

    def test_rejects_unknown_keys(self, tmp_path):
        manifest = _skab_manifest()
        (tmp_path / 'a.csv').write_text('')
        (tmp_path / 'b.csv').write_text('')
        manifest_path = _write_manifest(
            tmp_path,
            {
                'train': ['a.csv'],
                'validation': ['b.csv'],
                'test': [],
                'extra_key': ['c.csv'],
            },
        )

        with pytest.raises(ValueError, match='unknown'):
            manifest.load_skab_split_manifest(manifest_path)


class TestSkabSplitManifestSerializable:
    def test_produces_stable_json_serializable_payload(self, tmp_path):
        manifest = _skab_manifest()
        (tmp_path / 'train1.csv').write_text('')
        (tmp_path / 'val1.csv').write_text('')
        manifest_path = _write_manifest(
            tmp_path, {'train': ['train1.csv'], 'validation': ['val1.csv'], 'test': []}
        )

        result = manifest.load_skab_split_manifest(manifest_path)
        payload = result.to_payload()

        assert payload == {
            'train': ['train1.csv'],
            'validation': ['val1.csv'],
            'test': [],
        }

    def test_payload_is_json_serializable(self, tmp_path):
        import json

        manifest = _skab_manifest()
        (tmp_path / 'a.csv').write_text('')
        (tmp_path / 'b.csv').write_text('')
        manifest_path = _write_manifest(
            tmp_path, {'train': ['a.csv'], 'validation': ['b.csv'], 'test': []}
        )

        result = manifest.load_skab_split_manifest(manifest_path)
        payload = result.to_payload()

        assert json.dumps(payload) is not None


class TestSkabSplitManifestDataclass:
    def test_manifest_has_train_validation_test_attributes(self, tmp_path):
        manifest = _skab_manifest()
        (tmp_path / 'a.csv').write_text('')
        (tmp_path / 'b.csv').write_text('')
        manifest_path = _write_manifest(
            tmp_path, {'train': ['a.csv'], 'validation': ['b.csv'], 'test': []}
        )

        result = manifest.load_skab_split_manifest(manifest_path)

        assert hasattr(result, 'train')
        assert hasattr(result, 'validation')
        assert hasattr(result, 'test')
        assert isinstance(result.train, list)
        assert isinstance(result.validation, list)
        assert isinstance(result.test, list)
        assert all(isinstance(p, Path) for p in result.train)
        assert all(isinstance(p, Path) for p in result.validation)
        assert all(isinstance(p, Path) for p in result.test)
