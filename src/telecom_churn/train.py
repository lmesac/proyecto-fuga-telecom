
"""
Módulo de entrenamiento y evaluación de modelos.

Proceso:
1. Cargar el dataset.
2. Preparar variables predictoras y objetivo.
3. Identificar variables numéricas y categóricas.
4. Separar entrenamiento y prueba.
5. Comparar tres modelos mediante validación cruzada.
6. Evaluar los modelos con el conjunto de prueba.
7. Seleccionar el ganador utilizando F1 de validación cruzada.
8. Guardar el modelo y sus métricas.

El conjunto de prueba no se utiliza para seleccionar
el modelo ganador.
"""

import json

import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from telecom_churn.config import (
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    MODELS_DIR,
    MODEL_PATH,
    REPORTS_DIR,
    METRICS_PATH
)

from telecom_churn.data import load_data

from telecom_churn.features import (
    prepare_features,
    identify_feature_types
)

from telecom_churn.model import (
    get_models,
    build_pipeline
)


def train_models():
    """
    Ejecuta el entrenamiento, comparación y evaluación
    de los tres modelos de clasificación.
    """

    print("\n" + "=" * 60)
    print("PROYECTO TELECOM CHURN - ENTRENAMIENTO")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Cargar dataset
    # --------------------------------------------------

    df = load_data()

    print("\nDimensiones del dataset:", df.shape)

    # --------------------------------------------------
    # 2. Preparar variables predictoras y objetivo
    # --------------------------------------------------

    X, y = prepare_features(df)

    print("\nVariables predictoras:", X.shape[1])
    print("Registros:", X.shape[0])

    print("\nDistribución de la variable objetivo:")
    print(y.value_counts())

    # --------------------------------------------------
    # 3. Identificar tipos de variables
    # --------------------------------------------------

    numeric_features, categorical_features = (
        identify_feature_types(X)
    )

    print("\nVariables numéricas:")
    print(numeric_features)

    print("\nVariables categóricas:")
    print(categorical_features)

    # --------------------------------------------------
    # 4. Separar entrenamiento y prueba
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nDistribución de datos:")
    print("Entrenamiento:", X_train.shape)
    print("Prueba:", X_test.shape)

    # --------------------------------------------------
    # 5. Configurar validación cruzada
    # --------------------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    models = get_models()

    results = {}
    trained_pipelines = {}

    # --------------------------------------------------
    # 6. Entrenar y evaluar los tres modelos
    # --------------------------------------------------

    for model_name in models:

        print("\n" + "-" * 60)
        print(f"Evaluando modelo: {model_name}")
        print("-" * 60)

        # Construir pipeline completo
        pipeline = build_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            model_name=model_name
        )

        # Validación cruzada únicamente con entrenamiento
        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="f1",
            n_jobs=1
        )

        mean_cv_f1 = float(cv_scores.mean())
        std_cv_f1 = float(cv_scores.std())

        print(
            f"F1 validación cruzada: "
            f"{mean_cv_f1:.4f} (+/- {std_cv_f1:.4f})"
        )

        # Entrenar con el conjunto completo de entrenamiento
        pipeline.fit(X_train, y_train)

        # Evaluar con datos no utilizados en el entrenamiento
        predictions = pipeline.predict(X_test)

        probabilities = pipeline.predict_proba(X_test)[:, 1]

        # Calcular métricas
        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )

        average_precision = average_precision_score(
            y_test,
            probabilities
        )

        matrix = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        )

        # Guardar resultados del modelo
        results[model_name] = {
            "cv_f1_mean": mean_cv_f1,
            "cv_f1_std": std_cv_f1,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "average_precision": float(average_precision),
            "confusion_matrix": matrix.tolist(),
            "classification_report": classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=["No fuga", "Fuga"],
                output_dict=True,
                zero_division=0
            )
        }

        trained_pipelines[model_name] = pipeline

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1:        {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")

        print("\nMatriz de confusión:")
        print(matrix)

    # --------------------------------------------------
    # 7. Seleccionar el mejor modelo
    # --------------------------------------------------

    best_model_name = max(
        results,
        key=lambda name: results[name]["cv_f1_mean"]
    )

    best_pipeline = trained_pipelines[best_model_name]

    print("\n" + "=" * 60)
    print("SELECCIÓN DEL MEJOR MODELO")
    print("=" * 60)

    print("Modelo seleccionado:", best_model_name)

    print(
        "F1 promedio de validación cruzada:",
        f"{results[best_model_name]['cv_f1_mean']:.4f}"
    )

    # --------------------------------------------------
    # 8. Crear directorios de almacenamiento
    # --------------------------------------------------

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # 9. Guardar el pipeline ganador
    # --------------------------------------------------

    joblib.dump(
        best_pipeline,
        MODEL_PATH
    )

    print("\nModelo guardado en:")
    print(MODEL_PATH)

    # --------------------------------------------------
    # 10. Guardar métricas y resultados
    # --------------------------------------------------

    report = {
        "project": "Telecom Churn Prediction",
        "target": TARGET_COLUMN,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "selection_metric": "cv_f1_mean",
        "validation_folds": 5,
        "training_records": int(len(X_train)),
        "test_records": int(len(X_test)),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "best_model": best_model_name,
        "models": results
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nMétricas guardadas en:")
    print(METRICS_PATH)

    # --------------------------------------------------
    # 11. Resumen comparativo
    # --------------------------------------------------

    comparison = pd.DataFrame({
        model_name: {
            "CV F1": results[model_name]["cv_f1_mean"],
            "Accuracy": results[model_name]["accuracy"],
            "Precision": results[model_name]["precision"],
            "Recall": results[model_name]["recall"],
            "F1 prueba": results[model_name]["f1"],
            "ROC-AUC": results[model_name]["roc_auc"]
        }
        for model_name in results
    }).T

    print("\nCOMPARACIÓN FINAL DE MODELOS:")
    print(comparison.round(4).to_string())

    print("\nEntrenamiento finalizado correctamente.")

    return results


if __name__ == "__main__":
    train_models()