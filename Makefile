.PHONY: help setup data train train-baseline tune mlflow test lint format clean

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
PARAMS ?= ml/params/xgb_best.json
TRIALS ?= 30
TUNE_MIN ?= 60

help:
	@echo "make setup           create .venv and install ML + dev dependencies"
	@echo "make data            download IEEE-CIS data from Kaggle into data/raw"
	@echo "make train           train XGBoost (uses $(PARAMS) if present), register in MLflow"
	@echo "make tune            Optuna search: TRIALS=$(TRIALS), TUNE_MIN=$(TUNE_MIN) minutes max"
	@echo "make train-baseline  train the week 1 logistic regression baseline"
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
	PATH="$(CURDIR)/.venv/bin:$$PATH" bash scripts/download_data.sh

train:
	$(PY) -m ml.src.train --model xgb $(if $(wildcard $(PARAMS)),--params $(PARAMS))

tune:
	$(PY) -m ml.src.tune --trials $(TRIALS) --timeout-min $(TUNE_MIN) --out $(PARAMS)

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
