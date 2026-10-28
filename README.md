# Proyecto Final: Predicción de Fuga de Clientes (Telecom Churn)

Proyecto de Ciencia de Datos enfocado en la predicción de la fuga de clientes (*Churn*) utilizando un conjunto de datos de **10,000 registros y 35 variables** (sociodemográficas, operativas, financieras y métricas de experiencia del cliente como NPS y CSAT).

## Descripcion de las variables**

Variable; Descripción
CustomerID: Identificador único del cliente
Age: Edad del cliente
SeniorCitizen: Indicador de cliente senior
Gender: Género
Partner: Tiene pareja
Dependents: Tiene dependientes
TenureMonths: Meses de permanencia
TenureCohort: Rango de permanencia
PhoneService: Servicio telefónico
MultipleLines: Líneas múltiples
InternetService: Tipo de servicio de internet
OnlineSecurity: Seguridad online
OnlineBackup: Backup online
DeviceProtection: Protección del dispositivo
TechSupport: Soporte técnico
StreamingTV: Streaming TV
StreamingMovies: Streaming películas
ServiceCount: Cantidad de servicios
Contract: Tipo de contrato
PaperlessBilling: Facturación electrónica
AutoPay: Pago automático
PaymentMethod: Método de pago
MonthlyCharges: Cargo mensual

## Origen base: IBM y Kagle
El dataset de Telecom Churn con exactamente 10,000 registros y 35 variables es una variante específica generada de forma sintética o enriquecida a partir de los conjuntos de datos históricos originales de 7043 registros y 21 variables relacionados con la pérdidas de clientes liberados originalmente por IBM Sample Data Sets.

El dataset esta compuesto por 4 dimensiones claves:

1. Datos Demográficos (Originales + Nuevos): Género, si tiene pareja, dependientes, rango de edad y se suelen añadir variables como región, estado civil o nivel de ingresos estimado.

2. Servicios Contratados (Desglosados): En lugar de agruparlos, se añaden columnas individuales de tipo binario (Sí/No) para cada servicio: línea telefónica, múltiples líneas, tipo de internet (DSL, Fibra óptica), seguridad en línea, soporte técnico, streaming de TV y streaming de películas.

3. Métricas de Cuenta y Pago: Antigüedad (tenure), tipo de contrato (mensual, anual, bianual), facturación electrónica, método de pago, cargos mensuales y cargos totales.

4. Variables de Comportamiento Añadidas (para llegar a 35): En las versiones de 35 variables se inyectan indicadores analíticos adicionales de IBM Cognos como el Score de Satisfacción, el Churn Score (puntuación de propensión) y el CLTV (Customer Lifetime Value / Valor de vida del cliente), además de métricas de uso como volumen de datos consumidos, minutos de llamadas de voz y SMS enviados.

## Objetivo principal:

Servir de laboratorio para entrenar algoritmos de clasificación (como Regresión Logística, Random Forest o XGBoost) que predigan si un cliente cancelará o no su suscripción antes de que suceda

---

## 📁 Estructura del Proyecto
proyecto-fuga-telecom/
│
├── .github/
│   └── workflows/
│       └── ci.yml                  # Flujo de Integración Continua (CI) con GitHub Actions y pytest
│
├── data/
│   └── telecom_churn_10000.csv     # Dataset con 10,000 registros y 35 variables de telecomunicaciones
│
├── notebooks/
│   └── 01_analisis_exploratorio.ipynb # Cuaderno Jupyter con el EDA y validación de datos
│
├── src/
│   └── telecom_churn/              # Paquete principal de Python
│       ├── __init__.py             # Inicialización del paquete y rutas base
│       ├── config.py               # Rutas de directorios y configuración global
│       ├── features.py             # Limpieza, preprocesamiento y prevención de Data Leakage
│       ├── model.py                # Definición de modelos (Regresión Logística, Random Forest, HistGradientBoosting)
│       └── predict.py              # Módulo de scoring e inferencia para nuevos clientes
│
├── tests/
│   └── test_pipeline.py            # Pruebas unitarias automatizadas con pytest
│
├── .gitignore                      # Exclusión de archivos pesados, caché y entornos virtuales
├── Makefile                        # Automatización de tareas (setup, test, train)
├── pyproject.toml                  # Configuración del proyecto y dependencias gestionadas con uv
└── README.md                       # Documentación principal, contexto e interpretación de resultados



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

### 🛠️ Arquitectura y Componentes
El proyecto está desarrollado bajo una arquitectura modular y buenas prácticas de MLOps:
- **`src/telecom_churn/`**: Contiene la lógica de negocio desacoplada, facilitando la mantenibilidad y la reutilización de código.
- **`tests/`**: Pruebas unitarias ejecutadas mediante `pytest` para garantizar la estabilidad del pipeline.
- **`.github/workflows/ci.yml`**: Automatización de pruebas en la nube (CI/CD) para asegurar que el código se mantenga validado y verde.
- **`uv`**: Gestor de dependencias moderno de ultra-rápido rendimiento utilizado para la sincronización del entorno.