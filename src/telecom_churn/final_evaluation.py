
"""
Evaluación final del modelo en el conjunto de prueba.

El umbral 0.35 fue seleccionado previamente
utilizando validación cruzada sobre entrenamiento.
"""

import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from telecom_churn.config import (
    MODEL_PATH,
    METRICS_PATH,
    RANDOM_STATE,
    TEST_SIZE
)

from telecom_churn.data import load_data
from telecom_churn.features import prepare_features


def final_evaluation():

    # 1. Cargar y preparar los datos
    df = load_data()
    X, y = prepare_features(df)

    # 2. Reproducir la separación original
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # 3. Cargar el modelo entrenado
    model = joblib.load(MODEL_PATH)

    # 4. Obtener probabilidades de fuga
    probabilities = model.predict_proba(X_test)[:, 1]

    # 5. Evaluar el umbral seleccionado
    threshold = 0.35
    predictions = (probabilities >= threshold).astype(int)

    # 6. Calcular métricas
    results = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, probabilities)
    }

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    results["confusion_matrix"] = {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    }

    # 7. Mostrar resultados
    print("\nEVALUACIÓN FINAL DEL MODELO")
    print("=" * 50)

    for metric, value in results.items():
        print(f"{metric}: {value}")

    print("\nEvaluación realizada sobre el conjunto de prueba.")
    print(f"Registros evaluados: {len(X_test)}")


if __name__ == "__main__":
    final_evaluation()