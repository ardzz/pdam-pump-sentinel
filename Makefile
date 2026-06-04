.PHONY: help install dev dev-down run queue-work replay inject-drift retrain dashboard mlflow mlflow-report skab-eda train-pca train-pca-split test lint format ps logs

PYTHON ?= uv run python
SKAB_EDA ?= $(PYTHON) scripts/generate_skab_eda.py
SKAB_EDA_INPUT ?= tests/fixtures/skab_tiny.csv
SKAB_EDA_SPLIT_MANIFEST ?=
SKAB_EDA_OUTPUT_DIR ?= /tmp/pdam-skab-eda
SKAB_EDA_EXTRA_ARGS ?=
TRAIN_PCA ?= $(PYTHON) -m ml.training.train_pca
PCA_INPUT ?= tests/fixtures/skab_tiny.csv
PCA_OUTPUT_DIR ?= /tmp/pdam-pca-smoke
PCA_SPLIT_MANIFEST ?=
PCA_SPLIT_OUTPUT_DIR ?= /tmp/pdam-pca-split
PCA_WINDOW_SIZE ?= 1
PCA_STRIDE ?= 1
PCA_THRESHOLD_QUANTILE ?= 0.95
PCA_COMMON_ARGS ?= --window-size $(PCA_WINDOW_SIZE) --stride $(PCA_STRIDE) --threshold-quantile $(PCA_THRESHOLD_QUANTILE)
PCA_EXTRA_ARGS ?=
PCA_SPLIT_ARGS ?= $(PCA_SPLIT_OUTPUT_DIR) --split-manifest $(PCA_SPLIT_MANIFEST)
MLFLOW_REPORT_PORT ?= 5050
MLFLOW_REPORT_BACKEND ?= sqlite:///data/mlflow_live.db
MLFLOW_REPORT_ARTIFACTS ?= $(CURDIR)/data/mlflow_server_artifacts

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
	@echo "  mlflow       Launch MLflow UI (legacy file store)"
	@echo "  mlflow-report Launch MLflow report server against data/mlflow_live.db on MLFLOW_REPORT_PORT (default 5050)"
	@echo "  skab-eda     Generate SKAB EDA report artifacts"
	@echo "  train-pca    Train PCA from a single CSV smoke input"
	@echo "  train-pca-split Train PCA from PCA_SPLIT_MANIFEST"
	@echo "  test         Run tests"
	@echo "  lint         Run ruff lint"
	@echo "  format       Run ruff format"
	@echo "  ps           Show running services"
	@echo "  logs         Tail service logs"

install:
	uv sync

dev:
	docker compose -f infra/docker-compose.dev.yml --env-file infra/.env up -d

dev-down:
	docker compose -f infra/docker-compose.dev.yml --env-file infra/.env down

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

mlflow-report:
	uv run mlflow server \
		--host 0.0.0.0 \
		--port $(MLFLOW_REPORT_PORT) \
		--backend-store-uri $(MLFLOW_REPORT_BACKEND) \
		--serve-artifacts \
		--artifacts-destination $(MLFLOW_REPORT_ARTIFACTS)

skab-eda:
	$(SKAB_EDA) --input $(SKAB_EDA_INPUT) --output-dir $(SKAB_EDA_OUTPUT_DIR) $(if $(SKAB_EDA_SPLIT_MANIFEST),--split-manifest $(SKAB_EDA_SPLIT_MANIFEST),) $(SKAB_EDA_EXTRA_ARGS)

train-pca:
	$(TRAIN_PCA) $(PCA_INPUT) $(PCA_OUTPUT_DIR) $(PCA_COMMON_ARGS) $(PCA_EXTRA_ARGS)

train-pca-split:
	@if [ -z "$(PCA_SPLIT_MANIFEST)" ]; then echo "Set PCA_SPLIT_MANIFEST=path/to/manifest.json"; exit 2; fi
	$(TRAIN_PCA) $(PCA_SPLIT_ARGS) $(PCA_COMMON_ARGS) $(PCA_EXTRA_ARGS)

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

ps:
	docker compose -f infra/docker-compose.dev.yml --env-file infra/.env ps

logs:
	docker compose -f infra/docker-compose.dev.yml --env-file infra/.env logs -f
