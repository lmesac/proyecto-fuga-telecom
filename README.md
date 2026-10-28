# Proyecto Final: Predicción de Fuga de Clientes (Telecom Churn)

Proyecto de Ciencia de Datos enfocado en la predicción de la fuga de clientes (*Churn*) utilizando un conjunto de datos de **10,000 registros y 35 variables** (sociodemográficas, operativas, financieras y métricas de experiencia del cliente como NPS y CSAT).

---

## 📁 Estructura del Proyecto
```text
proyecto-fuga-telecom/
├── .gitignore
├── Makefile
├── README.md                 <-- Documentación principal e interpretación de resultados
├── pyproject.toml
├── data/
│   └── telecom_churn_10000.csv
├── notebooks/
│   └── 01_analisis_exploratorio.ipynb
├── src/
│   └── telecom_churn/
│       ├── __init__.py
│       ├── config.py
│       ├── features.py       <-- Limpieza e imputación de las 35 variables (sin data leakage)
│       ├── model.py          <-- Definición de los 3 modelos evaluados
│       ├── predict.py
│       └── train.py          <-- Script principal de entrenamiento y evaluación
└── tests/



## 📊 Resultados y Evaluación de Modelos (Churn Prediction)

A continuación se presentan los resultados obtenidos tras entrenar y evaluar los tres modelos predictivos utilizando las **35 variables** del dataset de clientes (excluyendo la variable de fuga directa para evitar *data leakage*):

### 1. Resumen de Métricas por Modelo

| Modelo | Accuracy (Exactitud) | Precision (Precisión) | Recall (Sensibilidad) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.6995** | **0.6667** | **0.5588** | **0.6080** | **0.7509** |
| **Random Forest** | 0.6815 | 0.6513 | 0.5084 | 0.5710 | 0.7290 |
| **HistGradientBoosting** | 0.6810 | 0.6373 | 0.5456 | 0.5879 | 0.7313 |

---

### 2. Interpretación de Resultados y Comportamiento de los Modelos

* **Regresión Logística (Logistic Regression):**
  * **Comportamiento:** Se posiciona como el **mejor modelo de los tres**, destacando en todas las métricas clave (Accuracy de ~70%, F1-Score de 0.608 y un ROC-AUC de 0.751).
  * **Interpretación de Negocio:** Al ser un modelo lineal, demuestra que las relaciones entre las variables operativas, financieras y de satisfacción del cliente (`NPS`, `CSAT`, contratos, cargos) guardan una tendencia lineal directa con la probabilidad de abandono. Logra el mejor balance detectando correctamente a los clientes en riesgo sin disparar los falsos positivos.

* **Random Forest:**
  * **Comportamiento:** Muestra un rendimiento competitivo con una Accuracy del 68.1% y un ROC-AUC de 0.729, aunque presenta el **Recall más bajo (50.8%)**.
  * **Interpretación de Negocio:** Aunque es un modelo potente basado en árboles de decisión múltiples, en este dataset tabular tiende a ser un poco más conservador, dejando escapar a la mitad de los clientes que verdaderamente terminan fugándose.

* **HistGradientBoosting:**
  * **Comportamiento:** Obtiene un desempeño muy similar a Random Forest en exactitud (~68.1%) y un ROC-AUC de 0.731, con un desempeño de Recall ligeramente superior (54.5%).
  * **Interpretación de Negocio:** Al ser un algoritmo iterativo de refuerzo de gradiente, captura interacciones no lineales complejas entre las variables de servicio, ofreciendo un buen balance analítico, aunque ligeramente por debajo de la Regresión Logística en este escenario particular.

---

### 3. Conclusión y Modelo Ganador

El modelo seleccionado como **Ganador para Producción y Estrategia de CX** es la **Regresión Logística**. 

* **¿Por qué se selecciona?:** Obtiene la mayor capacidad de discriminación general (**ROC-AUC de 0.7509**) y el mejor equilibrio entre precisión y sensibilidad (**F1-Score de 0.6080**). 
* **Valor para el Negocio (Customer Experience):** Este modelo permite a la compañía identificar con mayor certeza el grupo de clientes con alto riesgo de abandono. Esto optimiza el presupuesto de retención (evitando dar incentivos a clientes estables) y maximiza la captura oportuna de usuarios propensos a cancelar el servicio.