.PHONY: setup install test train

setup:
	uv venv
	uv sync

install:
	uv sync

test:
	uv run pytest

train:
	uv run python -m telecom_churn.train