"""
Servicios para preparar datos de clientes utilizados por la aplicación.
"""

import pandas as pd

from telecom_churn.data import load_data


EXCLUDED_COLUMNS = {
    "CustomerID",
    "Churn",
    "ChurnReason",
}


def get_reference_data() -> pd.DataFrame:
    """
    Carga el dataset utilizado como referencia por la aplicación.
    """
    return load_data()


def category_options(
    column: str,
    reference_df: pd.DataFrame,
) -> list:
    """
    Obtiene las categorías válidas de una variable categórica.
    """
    return sorted(
        reference_df[column]
        .dropna()
        .unique()
        .tolist()
    )


def default_value(
    column: str,
    reference_df: pd.DataFrame,
):
    """
    Obtiene un valor de referencia para una variable.

    Para variables numéricas utiliza la mediana.
    Para variables categóricas utiliza la moda.
    """

    if pd.api.types.is_numeric_dtype(
        reference_df[column]
    ):
        return reference_df[column].median()

    return reference_df[column].mode().iloc[0]


def build_base_customer(
    reference_df: pd.DataFrame,
) -> dict:
    """
    Construye un cliente base utilizando valores de referencia
    del dataset.
    """

    customer = {}

    for column in reference_df.columns:

        if column in EXCLUDED_COLUMNS:
            continue

        customer[column] = default_value(
            column,
            reference_df,
        )

    return customer


def complete_batch_data(
    df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Completa las columnas faltantes de un archivo masivo
    utilizando valores de referencia.
    """

    completed_df = df.copy()

    base_customer = build_base_customer(
        reference_df
    )

    for column, value in base_customer.items():

        if column not in completed_df.columns:
            completed_df[column] = value

    return completed_df