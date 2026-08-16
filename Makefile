# NanoVLM / microvlm local workflow. Always `conda activate microvlm` first.
SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help setup test smoke format lint check-env

help:
	@echo "Targets: setup test smoke format lint"
	@echo "Activate the existing conda env first: conda activate microvlm"

check-env:
	@python -c "import os,sys; e=os.environ.get('CONDA_DEFAULT_ENV',''); sys.exit(0) if e=='microvlm' else sys.exit('ERROR: conda env must be microvlm (got ' + repr(e) + '). Run: conda activate microvlm')"

setup: check-env
	pip install -e ".[dev]"

test: check-env
	pytest

smoke: check-env
	pytest tests/integration/test_smoke_end_to_end.py

format: check-env
	black src tests

lint: check-env
	ruff check src tests
