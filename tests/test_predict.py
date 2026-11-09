"""Pruebas del recorrido de guardado, carga y predicción del paquete."""
import numpy as np
import pandas as pd
import pytest

from telecom_churn import predict
from telecom_churn.config import CLASSIFICATION_THRESHOLD


def test_saved_pipeline_preserves_predictions(temporary_pipeline):
    customers = temporary_pipeline.customers
    # La referencia se calcula con el modelo en memoria y filas ajenas a su ajuste.
    expected = temporary_pipeline.pipeline.predict_proba(
        customers[temporary_pipeline.features]
    )[:, 1]
    # La función pública carga el archivo temporal y aplica el preprocesador guardado.
    result = predict.predict_dataframe(customers)
    np.testing.assert_allclose(result["Probabilidad_Fuga"], expected)
    np.testing.assert_array_equal(
        result["Prediccion_Fuga"], (expected >= CLASSIFICATION_THRESHOLD).astype(int)
    )
    assert result["CustomerID"].tolist() == customers["CustomerID"].tolist()


def test_csv_prediction_round_trip(temporary_pipeline, tmp_path):
    input_path = tmp_path / "customers.csv"
    output_path = tmp_path / "predictions.csv"
    # Se prueba el punto de entrada para archivos usando rutas temporales.
    temporary_pipeline.customers.to_csv(input_path, index=False)
    returned = predict.predict_file(input_path, output_path)
    saved = pd.read_csv(output_path)
    pd.testing.assert_frame_equal(saved, returned, check_exact=False)
    assert len(saved) == 30
    assert saved["Probabilidad_Fuga"].between(0, 1).all()


def test_missing_feature_has_clear_error(temporary_pipeline):
    customers = temporary_pipeline.customers.drop(columns="Contract")
    with pytest.raises(ValueError, match="Faltan variables requeridas.*Contract"):
        predict.predict_dataframe(customers)


def test_missing_model_has_clear_error(tmp_path, monkeypatch):
    # El caso de archivo ausente también se simula fuera de models/.
    monkeypatch.setattr(predict, "MODEL_PATH", tmp_path / "absent.pkl")
    with pytest.raises(FileNotFoundError, match="No se encontró el modelo"):
        predict.predict_dataframe(pd.DataFrame({"Contract": ["Month-to-month"]}))
