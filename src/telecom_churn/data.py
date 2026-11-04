
"""
Módulo de carga y validación de datos.
"""

import pandas as pd

from telecom_churn.config import DATA_FILE, TARGET_COLUMN


def load_data():
    """
    Carga el dataset original y realiza validaciones básicas.
    """

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    if df.empty:
        raise ValueError("El dataset está vacío.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"No existe la variable objetivo: {TARGET_COLUMN}"
        )

    print(f"Dataset cargado correctamente: {df.shape}")

    return df