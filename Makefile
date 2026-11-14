.PHONY: help setup data train-baseline mlflow test lint format clean

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

help:
	@echo "make setup           create .venv and install ML + dev dependencies"
	@echo "make data            download IEEE-CIS data from Kaggle into data/raw"
	@echo "make train-baseline  train logistic regression baseline (logged to MLflow)"
	@echo "make mlflow          start MLflow UI on http://localhost:5000"
	@echo "make test            run pytest"
	@echo "make lint            ruff check"
	@echo "make format          ruff format"

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r ml/requirements.txt -r requirements-dev.txt
	.venv/bin/pre-commit install

data:
	bash scripts/download_data.sh

train-baseline:
	$(PY) -m ml.src.train --model logreg

mlflow:
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

test:
	$(PY) -m pytest

lint:
	.venv/bin/ruff check .

format:
	.venv/bin/ruff format .

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
