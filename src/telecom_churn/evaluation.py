
"""
Evaluación de diferentes umbrales de clasificación
para el modelo de predicción de fuga de clientes.
"""

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from telecom_churn.config import (
    MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE
)

from telecom_churn.data import load_data
from telecom_churn.features import prepare_features


def evaluate_thresholds():

    # 1. Cargar dataset y preparar variables
    df = load_data()
    X, y = prepare_features(df)

    # 2. Separar entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # 3. Cargar el mejor modelo guardado
    model = joblib.load(MODEL_PATH)

    # 4. Generar predicciones de validación cruzada
    # Cada registro recibe una probabilidad de un modelo
    # que no fue entrenado con ese mismo registro.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    probabilities = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1
    )[:, 1]

    # 5. Evaluar diferentes umbrales
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50,
                  0.55, 0.60, 0.65, 0.70]

    results = []

    for threshold in thresholds:

        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y_train, predictions, zero_division=0
        )

        recall = recall_score(
            y_train, predictions, zero_division=0
        )

        f1 = f1_score(
            y_train, predictions, zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            y_train, predictions, labels=[0, 1]
        ).ravel()

        results.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp
        })

    # 6. Crear tabla de resultados
    results_df = pd.DataFrame(results)

    # 7. Guardar resultados
    reports_dir = MODEL_PATH.parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    output_path = reports_dir / "threshold_evaluation.csv"

    results_df.to_csv(output_path, index=False)

    # 8. Mostrar resultados
    print("\nEVALUACIÓN DE UMBRALES")
    print("=" * 75)

    print(
        results_df.round(4).to_string(index=False)
    )

    print("\nArchivo generado:")
    print(output_path)

    print(
        "\nNota: los resultados corresponden a predicciones "
        "de validación cruzada sobre el conjunto de entrenamiento."
    )


if __name__ == "__main__":
    evaluate_thresholds()