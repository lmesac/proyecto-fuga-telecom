"""Carga y validación de los datos originales, sin alterar el archivo fuente."""
import pandas as pd
from .config import DATA_FILE

def load_data(filepath=DATA_FILE):
    # La validación evita entrenar con etiquetas desconocidas o clientes repetidos.
    df = pd.read_csv(filepath)
    if not {"CustomerID", "Churn"}.issubset(df.columns):
        raise ValueError("Faltan CustomerID o Churn")
    if df.CustomerID.isna().any() or df.CustomerID.duplicated().any():
        raise ValueError("Los identificadores deben ser únicos y completos")
    if not df.Churn.isin(["Yes", "No"]).all():
        raise ValueError("Churn debe contener únicamente Yes o No")
    return df
