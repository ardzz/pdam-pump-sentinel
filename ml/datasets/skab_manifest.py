from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

_SPLIT_KEYS = {'train', 'validation', 'test'}
_METADATA_KEYS = {'schema_version', 'dataset_kind', 'name', 'base_dir', 'notes'}


@dataclass
class SkabSplitManifest:
    train: list[Path]
    validation: list[Path]
    test: list[Path]
    _base_dir: Path = field(default=Path('.'), repr=False, compare=False)
    schema_version: Any | None = None
    dataset_kind: Any | None = None
    name: Any | None = None
    base_dir: str | None = None
    notes: Any | None = None

    def to_payload(self) -> dict[str, list[str]]:
        return {
            'train': [str(p.relative_to(self._base_dir)) for p in self.train],
            'validation': [str(p.relative_to(self._base_dir)) for p in self.validation],
            'test': [str(p.relative_to(self._base_dir)) for p in self.test],
        }


def load_skab_split_manifest(path: Path) -> SkabSplitManifest:
    raw = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise ValueError('manifest must be a JSON object')

    allowed_keys = _SPLIT_KEYS | _METADATA_KEYS
    unknown_keys = set(raw.keys()) - allowed_keys
    if unknown_keys:
        raise ValueError(f'unknown keys in manifest: {sorted(unknown_keys)}')

    for key in ('train', 'validation', 'test'):
        if key not in raw:
            raise ValueError(f'missing required key in manifest: {key}')

    train_entries = _validated_entries(raw['train'], 'train')
    validation_entries = _validated_entries(raw['validation'], 'validation')
    test_entries = _validated_entries(raw['test'], 'test')

    if not train_entries:
        raise ValueError('train split must be non-empty')
    if not validation_entries:
        raise ValueError('validation split must be non-empty')

    base_dir = _resolve_manifest_base_dir(path, raw.get('base_dir'))

    train_paths = [_resolve_manifest_entry(base_dir, entry) for entry in train_entries]
    validation_paths = [_resolve_manifest_entry(base_dir, entry) for entry in validation_entries]
    test_paths = [_resolve_manifest_entry(base_dir, entry) for entry in test_entries]

    all_paths = [*train_paths, *validation_paths, *test_paths]
    seen: set[Path] = set()
    for p in all_paths:
        resolved = p.resolve()
        if resolved in seen:
            raise ValueError(f'duplicate file across splits: {p}')
        seen.add(resolved)

    for p in all_paths:
        if not p.exists():
            raise ValueError(f'missing file referenced in manifest: {p}')

    return SkabSplitManifest(
        train=train_paths,
        validation=validation_paths,
        test=test_paths,
        _base_dir=base_dir,
        schema_version=raw.get('schema_version'),
        dataset_kind=raw.get('dataset_kind'),
        name=raw.get('name'),
        base_dir=raw.get('base_dir'),
        notes=raw.get('notes'),
    )


def _validated_entries(value: object, split_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f'{split_name} split must be a list of strings')
    if not all(isinstance(entry, str) for entry in value):
        raise ValueError(f'{split_name} split entries must be strings')
    return cast(list[str], value)


def _resolve_manifest_base_dir(manifest_path: Path, value: object) -> Path:
    manifest_dir = manifest_path.resolve().parent
    if value is None:
        return manifest_dir
    if not isinstance(value, str):
        raise ValueError('manifest base_dir must be a relative path string')
    base_path = Path(value)
    if base_path.is_absolute():
        raise ValueError('manifest base_dir must be a relative path')
    resolved = (manifest_dir / base_path).resolve()
    if not resolved.is_relative_to(manifest_dir):
        raise ValueError('manifest base_dir must stay within manifest directory')
    return resolved


def _resolve_manifest_entry(base_dir: Path, entry: str) -> Path:
    entry_path = Path(entry)
    if entry_path.is_absolute():
        raise ValueError('manifest file entries must be relative paths')
    resolved = (base_dir / entry_path).resolve()
    if not resolved.is_relative_to(base_dir):
        raise ValueError('manifest file entries must stay within manifest directory')
    return base_dir / entry_path
