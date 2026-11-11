"""Recursos de prueba que se crean y guardan en directorios temporales."""
from types import SimpleNamespace

import joblib
import pytest
from sklearn.model_selection import train_test_split

from telecom_churn import predict
from telecom_churn.data import load_data
from telecom_churn.features import prepare_features, identify_feature_types
from telecom_churn.model import build_pipeline


@pytest.fixture
def temporary_pipeline(tmp_path, monkeypatch):
    """Entrena una regresión logística pequeña y reserva filas para inferencia."""
    data = load_data()
    # La estratificación asegura ejemplos de ambas clases en la muestra.
    training, held_out = train_test_split(
        data, train_size=300, test_size=30, stratify=data["Churn"], random_state=42
    )
    X, y = prepare_features(training)
    pipeline = build_pipeline(*identify_feature_types(X), model_name="Logistic Regression")
    pipeline.fit(X, y)

    # El archivo del modelo pertenece exclusivamente a esta prueba.
    model_path = tmp_path / "model.pkl"
    joblib.dump(pipeline, model_path)
    # predict.py importa la ruta directamente: cambiamos esa referencia solo durante el test.
    # Pytest restaura el valor original al finalizar.
    monkeypatch.setattr(predict, "MODEL_PATH", model_path)
    return SimpleNamespace(
        pipeline=pipeline, model_path=model_path, customers=held_out, features=X.columns
    )
