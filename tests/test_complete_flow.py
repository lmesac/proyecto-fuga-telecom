"""Pruebas de transformación e inferencia reales, con un modelo pequeño."""
import joblib
import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from telecom_churn.data import load_data
from telecom_churn.features import preprocess_data
from telecom_churn.predict import predict_churn

def test_features_missing_and_unknown():
    # Se ajusta con una muestra y se prueba una categoría que no apareció en ella.
    X, y, prep = preprocess_data(load_data().head(300))
    assert "ChurnReason" not in X and "CustomerID" not in X and "Churn" not in X
    prep.fit(X)
    new = X.iloc[[0]].copy()
    new["Contract"] = "categoria_nueva"
    new["MonthlyCharges"] = np.nan
    assert np.isfinite(prep.transform(new)).all()

def test_train_save_predict(tmp_path):
    # Verifica el recorrido completo sin depender de un modelo guardado previamente.
    X, y, prep = preprocess_data(load_data().head(400))
    model = Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=1000))])
    model.fit(X.iloc[:300], y.iloc[:300])
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    result = predict_churn(X.iloc[300:], path)
    assert len(result) == 100
    assert result.probabilidad_abandono.between(0, 1).all()
    with pytest.raises(ValueError, match="Faltan columnas"):
        predict_churn(X.iloc[300:].drop(columns="Contract"), path)

def test_app_starts():
    # Streamlit ejecuta la interfaz sin abrir un navegador ni publicar un servicio.
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(__import__("pathlib").Path("app/streamlit_app.py").resolve())).run(timeout=30)
    assert not app.exception
