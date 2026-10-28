import pandas as pd
from telecom_churn.config import PROCESSED_DATA_DIR, MODELS_DIR

def predict_churn(new_data_path: str):
    """
    Carga el dataset nuevo y genera las probabilidades de fuga de clientes.
    """
    print("1. Cargando datos para predicción...")
    df = pd.read_csv(new_data_path)
    
    # Aquí puedes integrar la lógica para cargar tu modelo entrenado y aplicar predict_proba
    print(f"Dataset cargado con {df.shape[0]} registros listos para scoring.")
    return df

if __name__ == "__main__":
    print("Módulo de predicción listo.")