"""
Servicio para evaluar el riesgo de fuga de clientes.
"""

import pandas as pd

from telecom_churn.predict import predict_dataframe
from telecom_churn.presentation.risk import (
    classify_risk,
    get_recommendation,
)


def evaluate_customers(
    df: pd.DataFrame,
    threshold: float | None = None,
) -> pd.DataFrame:

    predictions = predict_dataframe(
        df,
        threshold=threshold,
    )

    results = predictions.copy()

    results["Nivel_Riesgo"] = results[
        "Probabilidad_Fuga"
    ].apply(classify_risk)

    results["Recomendacion"] = results[
        "Nivel_Riesgo"
    ].apply(get_recommendation)

    return results