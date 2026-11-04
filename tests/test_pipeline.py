
import pandas as pd

from sklearn.model_selection import train_test_split

from telecom_churn.config import (
    TARGET_COLUMN,
    ID_COLUMN,
    EXCLUDED_COLUMNS
)

from telecom_churn.data import load_data

from telecom_churn.features import (
    prepare_features,
    identify_feature_types
)

from telecom_churn.model import (
    get_models,
    build_pipeline
)


def test_data_loading():
    """Verifica que el dataset se cargue correctamente."""

    df = load_data()

    assert not df.empty
    assert df.shape == (10000, 35)
    assert TARGET_COLUMN in df.columns


def test_feature_preparation():
    """Verifica la preparación de variables predictoras."""

    df = load_data()

    X, y = prepare_features(df)

    assert not X.empty
    assert not y.empty

    assert TARGET_COLUMN not in X.columns
    assert ID_COLUMN not in X.columns

    for column in EXCLUDED_COLUMNS:
        assert column not in X.columns

    assert set(y.unique()).issubset({0, 1})


def test_feature_identification():
    """Verifica la identificación de variables numéricas y categóricas."""

    df = load_data()
    X, y = prepare_features(df)

    numeric_features, categorical_features = (
        identify_feature_types(X)
    )

    assert len(numeric_features) > 0
    assert len(categorical_features) > 0

    assert (
        len(numeric_features) + len(categorical_features)
        == X.shape[1]
    )


def test_model_definitions():
    """Verifica que existan los tres algoritmos."""

    models = get_models()

    assert isinstance(models, dict)
    assert len(models) >= 3


def test_pipeline_training():
    """Verifica que el pipeline pueda entrenarse y predecir."""

    df = load_data()
    X, y = prepare_features(df)

    numeric_features, categorical_features = (
        identify_feature_types(X)
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=300,
        random_state=42,
        stratify=y
    )

    pipeline = build_pipeline(
        numeric_features,
        categorical_features,
        "Logistic Regression"
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    assert len(predictions) == len(y_test)
    assert set(predictions).issubset({0, 1})

    probabilities = pipeline.predict_proba(X_test)

    assert probabilities.shape == (len(X_test), 2)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()