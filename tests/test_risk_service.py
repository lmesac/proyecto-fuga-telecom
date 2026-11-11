import pandas as pd
import pytest

from telecom_churn.presentation.risk import (
    classify_risk,
    get_recommendation,
)
from telecom_churn.services.churn_service import evaluate_customers
from telecom_churn.data import load_data


def test_classify_low_risk():
    assert classify_risk(0.20) == "Bajo"


def test_classify_medium_risk():
    assert classify_risk(0.50) == "Medio"


def test_classify_high_risk():
    assert classify_risk(0.80) == "Alto"


def test_invalid_probability():
    with pytest.raises(ValueError):
        classify_risk(1.20)


def test_recommendation():
    recommendation = get_recommendation("Alto")

    assert recommendation == (
        "Priorizar al cliente para una acción de retención."
    )


def test_evaluate_customers_adds_risk_columns(temporary_pipeline):
    # El servicio usa el modelo temporal y un cliente reservado para inferencia.
    df = temporary_pipeline.customers

    customer = df.drop(
        columns=["CustomerID", "Churn", "ChurnReason"],
        errors="ignore",
    ).head(1)

    result = evaluate_customers(customer)

    assert "Probabilidad_Fuga" in result.columns
    assert "Nivel_Riesgo" in result.columns
    assert "Recomendacion" in result.columns

    assert result.loc[0, "Nivel_Riesgo"] in {
        "Bajo",
        "Medio",
        "Alto",
    }

    probability = result.loc[0, "Probabilidad_Fuga"]

    assert 0 <= probability <= 1