.PHONY: setup train test app
setup:
	uv sync --locked
train:
	uv run python -m telecom_churn.train
test:
	uv run pytest -q
app:
	uv run streamlit run app/streamlit_app.py
