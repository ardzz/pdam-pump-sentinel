from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

_LIBRARY_PACKAGES = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'scikit_learn': 'scikit-learn',
    'mlflow': 'mlflow',
}


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def collect_provenance(config: Any, input_files: Iterable[str | Path]) -> dict[str, Any]:
    """Collect deterministic run provenance for data/config driven ML jobs."""

    git_sha = _git_output(['rev-parse', 'HEAD']) or 'unknown'
    git_status = _git_output(['status', '--porcelain'])
    git_dirty: bool | str = 'unknown' if git_status is None else bool(git_status)
    normalized_config = _json_safe(config)

    return {
        'generated_at_utc': datetime.now(UTC).isoformat(timespec='seconds').replace('+00:00', 'Z'),
        'git_sha': git_sha,
        'git_dirty': git_dirty,
        'git': {'sha': git_sha, 'dirty': git_dirty},
        'library_versions': _library_versions(),
        'dataset_hashes': _dataset_hashes(input_files),
        'config': normalized_config,
        'config_hash': _sha256_json(normalized_config),
    }


def _dataset_hashes(input_files: Iterable[str | Path]) -> dict[str, str]:
    paths = sorted((Path(path) for path in input_files), key=lambda path: str(path))
    return {str(path): sha256_file(path) for path in paths}


def _library_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {'python': platform.python_version()}
    for name, package in _LIBRARY_PACKAGES.items():
        try:
            versions[name] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git_output(args: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(
            ['git', *args],
            cwd=Path.cwd(),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    return result.stdout.strip()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    if isinstance(value, set | frozenset):
        return [_json_safe(item) for item in sorted(value, key=repr)]
    if isinstance(value, bool | int | str) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    item = getattr(value, 'item', None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass
    return str(value)
