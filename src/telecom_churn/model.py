
"""
Módulo de construcción de modelos de Machine Learning.

Define los algoritmos de clasificación y construye los pipelines que integran el preprocesamiento con cada modelo.

El entrenamiento, evaluación y almacenamiento se realizan en train.py.
"""

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier
)

from telecom_churn.features import build_preprocessor


def get_models():
    """
    Define los tres modelos de clasificación que se van a comparar.
    """

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

        "HistGradientBoosting": HistGradientBoostingClassifier(
            random_state=42
        )
    }

    return models


def build_pipeline(
    numeric_features,
    categorical_features,
    model_name="Logistic Regression"
):
    """
    Construye un pipeline completo de Machine Learning.

    Integra el preprocesamiento y el algoritmo seleccionado.

    El pipeline todavía no está entrenado.
    """

    models = get_models()

    if model_name not in models:
        raise ValueError(
            f"Modelo no válido: {model_name}. "
            f"Modelos disponibles: {list(models.keys())}"
        )

    if not numeric_features and not categorical_features:
        raise ValueError(
            "Debe proporcionarse al menos una variable predictora."
        )

    # Construir el preprocesador
    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features
    )

    # Integrar preprocesamiento y algoritmo
    full_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", models[model_name])
        ]
    )

    return full_pipeline