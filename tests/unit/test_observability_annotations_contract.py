from __future__ import annotations

from typing import Any

from app.observability import annotations


class _Response:
    def raise_for_status(self) -> None:
        return None


def test_post_annotation_constructs_request_with_bearer_header(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, json: dict[str, Any], headers: dict[str, str], timeout: int) -> _Response:
        calls.append({'url': url, 'json': json, 'headers': headers, 'timeout': timeout})
        return _Response()

    monkeypatch.setenv('GRAFANA_URL', 'http://grafana.example')
    monkeypatch.setenv('GRAFANA_API_KEY', 'secret')
    monkeypatch.setattr(annotations.requests, 'post', fake_post)

    annotations.post_annotation('hello', ['drift'], time_ms=123)

    assert calls == [
        {
            'url': 'http://grafana.example/api/annotations',
            'json': {'text': 'hello', 'tags': ['drift'], 'time': 123},
            'headers': {'Authorization': 'Bearer secret'},
            'timeout': 3,
        }
    ]


def test_post_annotation_omits_authorization_header_when_unset(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, json: dict[str, Any], headers: dict[str, str], timeout: int) -> _Response:
        calls.append({'url': url, 'json': json, 'headers': headers, 'timeout': timeout})
        return _Response()

    monkeypatch.setenv('GRAFANA_URL', 'http://grafana.example/')
    monkeypatch.delenv('GRAFANA_API_KEY', raising=False)
    monkeypatch.setattr(annotations.requests, 'post', fake_post)

    annotations.post_annotation('hello', ['model'], time_ms=456)

    assert calls[0]['url'] == 'http://grafana.example/api/annotations'
    assert 'Authorization' not in calls[0]['headers']
    assert calls[0]['json'] == {'text': 'hello', 'tags': ['model'], 'time': 456}


def test_post_annotation_never_raises_on_connection_error(monkeypatch) -> None:
    def fake_post(url: str, json: dict[str, Any], headers: dict[str, str], timeout: int) -> _Response:
        raise annotations.requests.exceptions.ConnectionError('down')

    monkeypatch.delenv('GRAFANA_API_KEY', raising=False)
    monkeypatch.setattr(annotations.requests, 'post', fake_post)

    annotations.post_annotation('hello', ['drift'], time_ms=789)
