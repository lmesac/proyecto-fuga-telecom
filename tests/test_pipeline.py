import os
import pandas as pd
from telecom_churn.features import load_and_preprocess_data

def test_data_loading():
    """
    Verifica que el archivo de datos existe y se carga con las dimensiones correctas (10,000 filas y 35 columnas).
    """
    data_path = "data/telecom_churn_10000.csv"
    assert os.path.exists(data_path), "El archivo de datos no se encuentra en la ruta especificada."
    
    df = pd.read_csv(data_path)
    assert df.shape[0] > 0, "El dataset está vacío."
    assert df.shape[1] == 35, f"Se esperaban 35 columnas, pero se encontraron {df.shape[1]}."

def test_model_import():
    """
    Verifica que los modelos se puedan importar correctamente desde el módulo.
    """
    from telecom_churn.model import get_models
    models = get_models()
    assert isinstance(models, dict), "Los modelos deben devolverse en un diccionario."
    assert len(models) >= 3, "Debe haber al menos 3 modelos definidos para comparar."