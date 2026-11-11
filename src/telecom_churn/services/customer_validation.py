"""
Validaciones para los datos ingresados en la aplicación.
"""

import pandas as pd


def validate_customer_data(df: pd.DataFrame) -> None:
    """
    Valida reglas básicas antes de generar predicciones.

    Lanza ValueError cuando encuentra datos inválidos.
    """

    if df.empty:
        raise ValueError("El archivo no contiene registros.")

    numeric_rules = {
        "TenureMonths": (0, None),
        "MonthlyCharges": (0, None),
        "SupportTickets": (0, None),
        "Complaints": (0, None),
        "NPS": (0, 10),
        "CSAT": (1, 5),
    }

    for column, (minimum, maximum) in numeric_rules.items():

        if column not in df.columns:
            continue

        numeric_values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        invalid_type = (
            numeric_values.isna()
            & df[column].notna()
        )

        if invalid_type.any():
            rows = (
                invalid_type[invalid_type]
                .index
                .tolist()
            )

            raise ValueError(
                f"La columna '{column}' contiene valores no numéricos "
                f"en las filas: {rows}"
            )

        if minimum is not None:
            below_minimum = numeric_values < minimum

            if below_minimum.any():
                raise ValueError(
                    f"La columna '{column}' debe ser mayor o igual a "
                    f"{minimum}."
                )

        if maximum is not None:
            above_maximum = numeric_values > maximum

            if above_maximum.any():
                raise ValueError(
                    f"La columna '{column}' debe ser menor o igual a "
                    f"{maximum}."
                )


def validate_known_columns(
    df: pd.DataFrame,
    allowed_columns: set,
) -> None:
    """
    Verifica que el archivo no contenga columnas desconocidas.
    """

    unknown_columns = [
        column
        for column in df.columns
        if column not in allowed_columns
    ]

    if unknown_columns:
        raise ValueError(
            "El archivo contiene columnas no reconocidas: "
            f"{unknown_columns}"
        )