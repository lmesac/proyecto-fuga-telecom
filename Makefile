
.PHONY: setup install train predict app test lint check

setup:
	uv sync

install:
	uv sync

train:
	uv run python -m telecom_churn.train

predict:
	uv run python -m telecom_churn.predict

app:
	uv run streamlit run app.py

test:
	uv run pytest -v

lint:
	uv run ruff check src tests

check: lint test