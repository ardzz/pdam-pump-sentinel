.PHONY: help install dev dev-down run queue-work replay inject-drift retrain dashboard mlflow test lint format ps logs

help:
	@echo "PDAM Pump Sentinel - available targets:"
	@echo "  install      Install dependencies via uv"
	@echo "  dev          Start infra (Redis + MySQL + Mosquitto + MLflow)"
	@echo "  dev-down     Stop infra"
	@echo "  run          Run RouteMQ application"
	@echo "  queue-work   Run queue worker"
	@echo "  replay       Replay SKAB dataset to MQTT"
	@echo "  inject-drift Inject synthetic sensor drift (demo)"
	@echo "  retrain      Manually trigger retraining"
	@echo "  dashboard    Launch Streamlit dashboard"
	@echo "  mlflow       Launch MLflow UI"
	@echo "  test         Run tests"
	@echo "  lint         Run ruff lint"
	@echo "  format       Run ruff format"
	@echo "  ps           Show running services"
	@echo "  logs         Tail service logs"

install:
	uv sync

dev:
	docker compose -f infra/docker-compose.dev.yml up -d

dev-down:
	docker compose -f infra/docker-compose.dev.yml down

run:
	uv run python -m bootstrap.app

queue-work:
	uv run routemq queue-work --queue default

replay:
	uv run python scripts/replay_skab.py

inject-drift:
	uv run python scripts/inject_drift.py

retrain:
	uv run python scripts/trigger_retrain.py

dashboard:
	uv run streamlit run dashboard/app.py

mlflow:
	uv run mlflow ui --host 0.0.0.0 --port 5000

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

ps:
	docker compose -f infra/docker-compose.dev.yml ps

logs:
	docker compose -f infra/docker-compose.dev.yml logs -f
