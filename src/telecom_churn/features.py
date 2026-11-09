
"""
Módulo de preparación y transformación de variables.

Responsabilidades:
1. Separar variables predictoras y variable objetivo.
2. Excluir identificadores y variables no disponibles para la predicción.
3. Identificar variables numéricas y categóricas.
4. Construir el preprocesador de Scikit-learn.

"""

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from telecom_churn.config import (
    TARGET_COLUMN,
    EXCLUDED_COLUMNS
)


def prepare_features(
    df,
    target_col=TARGET_COLUMN,
    excluded_columns=None
):
    """
    Separa las variables predictoras (X) de la variable objetivo (y).

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset original.

    target_col : str
        Nombre de la variable objetivo.

    excluded_columns : list, optional
        Columnas que deben excluirse de las variables predictoras.

    Returns
    -------
    X : pandas.DataFrame
        Variables predictoras.

    y : pandas.Series
        Variable objetivo codificada como 0 y 1.
    """

    if df.empty:
        raise ValueError("El dataset está vacío.")

    if target_col not in df.columns:
        raise ValueError(
            f"No se encontró la variable objetivo: {target_col}"
        )

    # Verificar que la variable objetivo no tenga valores nulos
    if df[target_col].isna().any():
        raise ValueError(
            f"La variable objetivo {target_col} contiene valores nulos."
        )

    # Normalizar los valores de la variable objetivo
    target_values = df[target_col].astype(str).str.strip().str.lower()

    # Equivalencias permitidas para la clasificación binaria
    target_mapping = {
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0,
        "1.0": 1,
        "0.0": 0,
        "true": 1,
        "false": 0
    }

    y = target_values.map(target_mapping)

    # No convertir silenciosamente categorías desconocidas en 0
    if y.isna().any():
        invalid_values = target_values[y.isna()].unique().tolist()

        raise ValueError(
            "Se encontraron valores no reconocidos en la variable "
            f"objetivo {target_col}: {invalid_values}"
        )

    y = y.astype(int)

    # Comprobar que existan las dos clases
    if y.nunique() != 2:
        raise ValueError(
            "La variable objetivo debe contener las dos clases: "
            "0 (No fuga) y 1 (Fuga)."
        )

    # Definir las columnas que deben excluirse
    if excluded_columns is None:
        excluded_columns = EXCLUDED_COLUMNS

    columns_to_drop = [
        col for col in [target_col] + excluded_columns
        if col in df.columns
    ]

    # Construir las variables predictoras
    X = df.drop(columns=columns_to_drop).copy()

    if X.empty:
        raise ValueError(
            "No existen variables predictoras después de las exclusiones."
        )

    return X, y


def identify_feature_types(X):
    """
    Identifica automáticamente las variables numéricas y categóricas.

    Parameters
    ----------
    X : pandas.DataFrame
        Variables predictoras.

    Returns
    -------
    numeric_features : list
        Columnas numéricas.

    categorical_features : list
        Columnas categóricas.
    """

    # Identificar variables numéricas
    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # Identificar variables categóricas
    categorical_features = X.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    # Verificar que todas las columnas estén clasificadas
    classified_columns = set(
        numeric_features + categorical_features
    )

    unclassified_columns = [
        col for col in X.columns
        if col not in classified_columns
    ]

    if unclassified_columns:
        raise ValueError(
            "Existen variables con tipos de datos no contemplados: "
            f"{unclassified_columns}"
        )

    if not numeric_features and not categorical_features:
        raise ValueError(
            "No se identificaron variables predictoras válidas."
        )

    return numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features):
    """
    Construye el preprocesador de variables numéricas y categóricas.

    La imputación, el escalado y la codificación se ajustarán
    exclusivamente con los datos utilizados en pipeline.fit().
    """

    # Transformación de variables numéricas
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # Transformación de variables categóricas
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="missing"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    # Integración de los transformadores
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ],
        remainder="drop"
    )

    return preprocessor