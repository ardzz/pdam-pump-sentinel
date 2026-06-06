from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

LOGGER = logging.getLogger('PDAM.screenshots')

_PLAYWRIGHT_CACHE_CANDIDATES = (
    Path.home() / '.cache' / 'ms-playwright',
    Path('/usr/local/share/ms-playwright'),
)
_CHROMIUM_BINARY_HINTS = (
    'chromium-headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell',
    'chromium-*/chrome-linux64/chrome',
    'chrome-headless-shell-linux64/chrome-headless-shell',
)
_SYSTEM_BINARY_HINTS = (
    'chromium',
    'chromium-browser',
    'google-chrome',
    'google-chrome-stable',
    'chrome',
)


@dataclass(frozen=True)
class Target:
    label: str
    url: str
    description: str
    wait_seconds: int = 4


DEFAULT_TARGETS: tuple[Target, ...] = (
    Target(
        label='mlflow-home',
        url='http://localhost:5000/',
        description='MLflow experiments overview',
        wait_seconds=4,
    ),
    Target(
        label='mlflow-experiments',
        url='http://localhost:5000/#/experiments',
        description='MLflow experiments list',
        wait_seconds=4,
    ),
    Target(
        label='mlflow-models',
        url='http://localhost:5000/#/models',
        description='MLflow registered models list (PumpAD versions + champion alias)',
        wait_seconds=4,
    ),
    Target(
        label='mlflow-pumpad',
        url='http://localhost:5000/#/models/PumpAD',
        description='PumpAD model detail with version table + aliases',
        wait_seconds=5,
    ),
    Target(
        label='grafana-routemq',
        url='http://localhost:13000/d/pumpad-observability/pdam-pump-sentinel-routemq-observability?from=now-1h&to=now&timezone=browser&refresh=10s&kiosk=tv',
        description='Grafana RouteMQ observability dashboard (kiosk mode)',
        wait_seconds=8,
    ),
    Target(
        label='streamlit-home',
        url='http://localhost:8501/',
        description='Streamlit dashboard landing page',
        wait_seconds=5,
    ),
    Target(
        label='streamlit-live-sensors',
        url='http://localhost:8501/live_sensors',
        description='Streamlit Live Sensors page (anomaly status + score)',
        wait_seconds=5,
    ),
    Target(
        label='streamlit-anomaly-history',
        url='http://localhost:8501/anomaly_history',
        description='Streamlit Anomaly History page (score timeline)',
        wait_seconds=5,
    ),
    Target(
        label='streamlit-model-registry',
        url='http://localhost:8501/model_registry',
        description='Streamlit Model Registry page (name/version/activated_at)',
        wait_seconds=5,
    ),
    Target(
        label='streamlit-drift-reports',
        url='http://localhost:8501/drift_reports',
        description='Streamlit Drift & Training page',
        wait_seconds=5,
    ),
)


def _discover_chrome_binary() -> Path:
    env_override = os.getenv('PDAM_CHROME_BIN')
    if env_override:
        candidate = Path(env_override)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f'PDAM_CHROME_BIN={env_override!r} does not exist')

    for cache in _PLAYWRIGHT_CACHE_CANDIDATES:
        if not cache.exists():
            continue
        for hint in _CHROMIUM_BINARY_HINTS:
            for match in sorted(cache.glob(hint), reverse=True):
                if match.is_file() and os.access(match, os.X_OK):
                    return match

    for hint in _SYSTEM_BINARY_HINTS:
        resolved = shutil.which(hint)
        if resolved:
            return Path(resolved)

    raise FileNotFoundError(
        'no chrome / chromium / chrome-headless-shell binary located. '
        'Install one or set PDAM_CHROME_BIN to a valid executable.'
    )


def _capture(chrome: Path, target: Target, out_path: Path, window: str, headless_arg: str) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(chrome),
        headless_arg,
        '--no-sandbox',
        '--disable-gpu',
        '--disable-dev-shm-usage',
        '--hide-scrollbars',
        f'--window-size={window}',
        f'--virtual-time-budget={max(target.wait_seconds, 2) * 1000}',
        f'--screenshot={out_path}',
        target.url,
    ]
    LOGGER.info('capture %s → %s', target.label, out_path.name)
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(target.wait_seconds, 5) + 20,
            cwd=str(_PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired:
        LOGGER.error('  ✗ %s timed out', target.label)
        return False

    if result.returncode != 0:
        LOGGER.error('  ✗ %s exit=%s stderr=%s', target.label, result.returncode, result.stderr.strip()[:200])
        return False

    if not out_path.exists():
        LOGGER.error('  ✗ %s output missing', target.label)
        return False

    size_kb = out_path.stat().st_size // 1024
    LOGGER.info('  ✓ %s saved (%s KB)', target.label, size_kb)
    return True


def _resolve_headless_flag(chrome: Path) -> str:
    name = chrome.name.lower()
    if 'headless-shell' in name:
        return '--headless'
    return '--headless=new'


def _resolve_targets(target_filter: Sequence[str] | None) -> tuple[Target, ...]:
    if not target_filter:
        return DEFAULT_TARGETS
    wanted = {t.lower() for t in target_filter}
    selected = tuple(t for t in DEFAULT_TARGETS if t.label.lower() in wanted)
    missing = wanted - {t.label.lower() for t in selected}
    if missing:
        known = ', '.join(t.label for t in DEFAULT_TARGETS)
        raise SystemExit(f'unknown targets: {sorted(missing)!r}. available: {known}')
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    parser = argparse.ArgumentParser(description='Capture demo screenshots via headless Chrome.')
    parser.add_argument('--tag', default=os.getenv('SCREENSHOT_TAG', 'capture'))
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=Path(os.getenv('SCREENSHOT_OUT_DIR', 'docs/presentation/screenshots')),
    )
    parser.add_argument(
        '--targets',
        nargs='*',
        default=None,
        help='restrict to specific target labels (default: all). use --list to see options.',
    )
    parser.add_argument('--list', action='store_true', help='list available targets and exit')
    parser.add_argument(
        '--window',
        default=os.getenv('SCREENSHOT_WINDOW', '1920,1080'),
        help='chrome --window-size value (default 1920,1080)',
    )
    parser.add_argument(
        '--no-timestamp',
        action='store_true',
        help='omit timestamp suffix in filenames',
    )
    args = parser.parse_args(argv)

    if args.list:
        print('Available targets:')
        for target in DEFAULT_TARGETS:
            print(f'  {target.label:<28} {target.description}')
            print(f'    url: {target.url}')
        return 0

    chrome = _discover_chrome_binary()
    headless_flag = _resolve_headless_flag(chrome)
    LOGGER.info('using chrome binary: %s (%s)', chrome, headless_flag)

    selected = _resolve_targets(args.targets)
    stamp = '' if args.no_timestamp else datetime.now(timezone.utc).strftime('-%Y%m%dT%H%M%SZ')

    passed = 0
    for target in selected:
        out_name = f'{args.tag}-{target.label}{stamp}.png'
        out_path = args.out_dir / out_name
        if _capture(chrome, target, out_path, args.window, headless_flag):
            passed += 1
    total = len(selected)
    LOGGER.info('captured %d/%d', passed, total)
    print(f'\n=== Screenshots: {passed}/{total} captured under {args.out_dir} (tag={args.tag}) ===')
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
