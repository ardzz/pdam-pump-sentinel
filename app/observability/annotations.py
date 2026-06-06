from __future__ import annotations

import logging
import os
import time

import requests

logger = logging.getLogger('PDAM.observability')


def post_annotation(text: str, tags: list[str], time_ms: int | None = None) -> None:
    grafana_url = os.getenv('GRAFANA_URL', 'http://localhost:13000').rstrip('/')
    api_key = os.getenv('GRAFANA_API_KEY')
    headers = {}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    body = {
        'text': text,
        'tags': list(tags),
        'time': time_ms if time_ms is not None else int(time.time() * 1000),
    }
    try:
        response = requests.post(f'{grafana_url}/api/annotations', json=body, headers=headers, timeout=3)
        response.raise_for_status()
    except Exception as exc:
        logger.warning('could not post grafana annotation: %s', exc, exc_info=False)
