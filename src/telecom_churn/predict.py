
"""
Módulo para realizar predicciones de fuga
sobre nuevos registros de clientes.
"""

import joblib
import pandas as pd

from telecom_churn.config import (
    MODEL_PATH,
    ID_COLUMN,
    TARGET_COLUMN,
    EXCLUDED_COLUMNS,
    CLASSIFICATION_THRESHOLD
)


def predict_dataframe(df, threshold=None):

    # Validar que el dataset no esté vacío
    if df.empty:
        raise ValueError("El archivo no contiene registros.")

    # Cargar el modelo entrenado
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    # Recuperar las variables utilizadas durante el entrenamiento
    required_features = list(model.feature_names_in_)

    # Identificar variables faltantes
    missing_features = [
        col for col in required_features
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Faltan variables requeridas: {missing_features}"
        )

    # Preparar los datos conservando el orden de entrenamiento
    X = df[required_features].copy()

    # Obtener probabilidades
    probabilities = model.predict_proba(X)[:, 1]

    # Aplicar umbral
    if threshold is None:
        threshold = CLASSIFICATION_THRESHOLD

    predictions = (probabilities >= threshold).astype(int)

    # Construir resultados
    results = pd.DataFrame()

    if ID_COLUMN in df.columns:
        results[ID_COLUMN] = df[ID_COLUMN].values

    results["Probabilidad_Fuga"] = probabilities
    results["Prediccion_Fuga"] = predictions

    results["Resultado"] = results["Prediccion_Fuga"].map({
        0: "No fuga",
        1: "Posible fuga"
    })

    return results


def predict_file(input_path, output_path="reports/predictions.csv"):

    df = pd.read_csv(input_path)

    results = predict_dataframe(df)

    results.to_csv(output_path, index=False)

    print("\nPredicciones generadas correctamente.")
    print(results.head())
    print(f"\nArchivo guardado en: {output_path}")

    return results