# Predicción de abandono Telecom

Integrantes: Jorge Diaz, Federico Martinez y Lida Mesa.

Versión de trabajo de Federico. Objetivo: comparar modelos para identificar abandono observado y apoyar la priorización de revisión de clientes. No demuestra eficacia de acciones de retención ni predicción a un plazo futuro.

## Datos
10.000 registros sintéticos y 35 variables, generados desde cero en Python según la documentación del equipo. IBM Telco Customer Churn (7.043 registros y 21 variables) fue una referencia conceptual: sus registros originales no se incorporaron. Consultar [dataset-card.md](dataset-card.md) para la fuente documental, las reglas declaradas y los pendientes de reproducibilidad y licencia. Se excluyen identificador, etiqueta y motivo de abandono: 32 predictores.

## Ejecutar
Python 3.12 y uv. Desde la raíz:

```sh
uv sync --locked
uv run python -m telecom_churn.train
uv run pytest -q
uv run streamlit run app/streamlit_app.py
```

En Windows, si uv no está en PATH, usar `.venv/Scripts/uv.exe`. Alternativamente usar el Python del entorno para los comandos `-m`. Makefile ofrece setup, train, test y app.

## Diseño y resultados
Separación estratificada 60% entrenamiento, 20% validación y 20% prueba, semilla 42. Imputación y codificación se ajustan dentro del pipeline. Se comparan tres modelos y baseline. Selección por ROC-AUC de validación, luego reajuste con 80% y evaluación final en test. Umbral fijo 0,5.

Resultados de la ejecución local en reports/metrics.json. Ganador: Logistic Regression

Métricas test:
```json
{
  "accuracy": 0.6995,
  "precision": 0.6666666666666666,
  "recall": 0.5587529976019184,
  "f1": 0.6079582517938682,
  "roc_auc": 0.7509409282179745,
  "confusion_matrix": [
    [
      933,
      233
    ],
    [
      368,
      466
    ]
  ]
}
```
El modelo se guarda con el preprocesador en models/churn.joblib. La aplicación recibe CSV o permite seleccionar un cliente del dataset.

## Limitaciones
No hay fechas, horizonte de abandono ni evidencia de que satisfacción y valores estimados se midieran antes de la baja. Es una evaluación académica sobre una simulación: las relaciones y métricas no constituyen evidencia del comportamiento de clientes reales. Una mejor regresión logística no demuestra causalidad ni relaciones necesariamente lineales. Probabilidades no calibradas para uso operativo. El conjunto de prueba no debe usarse para ajustar hiperparámetros.

## Estructura
src/telecom_churn contiene carga, preparación, modelos, entrenamiento y predicción; app contiene interfaz; tests valida carga, procesamiento e inferencia; notebooks/02_eda_federico.ipynb contiene EDA con Plotly. REQUISITOS.md identifica pendientes técnicos y del equipo.

## Guía de lectura
1. OBJETIVO.md: propósito y criterios de éxito.
2. dataset-card.md: datos y dudas pendientes.
3. notebooks/02_eda_federico.ipynb: preguntas, cálculos y conclusiones.
4. HALLAZGOS.md: evidencia y decisiones.
5. GUIA_DEL_CODIGO.md: función de cada componente.
6. MODEL_CARD.md: resultados y límites.
7. reports/EVALUACION.md: estabilidad, umbrales e interpretación.
