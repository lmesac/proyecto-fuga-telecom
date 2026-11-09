
.PHONY: setup install train predict app test lint check

setup:
	uv sync --locked

install:
	uv sync --locked

train:
	uv run python -m telecom_churn.train

predict:
	uv run python -m telecom_churn.predict

app:
	uv run streamlit run app/app_j.py

test:
	uv run pytest -v

lint:
	uv run ruff check src tests

check: lint test