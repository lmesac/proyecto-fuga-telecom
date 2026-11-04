# Predicción de fuga de clientes en telecomunicaciones

Proyecto académico de Ciencia de Datos orientado a la exploración de datos y evaluación de modelos de clasificación para analizar la fuga de clientes (*Customer Churn*) en el sector de telecomunicaciones.

## 1. Información del proyecto

* **Programa:** Especialización en Ciencia de Datos e Inteligencia Artificial
* **Institución:** Universidad de Medellín
* **Módulo:** Introducción a la Ciencia de Datos
* **Periodo:** 2026-II
* **Integrantes:** Jorge Díaz, Federico Martínez y Lida Mesa
* **Repositorio:** [lmesac/proyecto-fuga-telecom](https://github.com/lmesac/proyecto-fuga-telecom)

## 2. Problema y objetivo

La fuga de clientes puede afectar los ingresos recurrentes, los costos de adquisición y retención y la relación de una empresa con sus clientes. Identificar características asociadas con el abandono puede ayudar a estudiar perfiles de riesgo y orientar decisiones de retención.

El proyecto desarrolla un flujo reproducible de Ciencia de Datos que comprende inspección y preparación de datos, análisis exploratorio, visualización, entrenamiento y evaluación de modelos de clasificación.

La pregunta que orienta el análisis es:

> ¿Qué características de los clientes están asociadas con la fuga y qué capacidad tienen los modelos de clasificación para identificar clientes con mayor riesgo de abandono?

La variable objetivo es `Churn`. La clase positiva representa a los clientes que abandonan el servicio y la clase negativa a quienes permanecen.

## 3. Procedencia y construcción del dataset

### 3.1 Fuente de referencia

Se tomó como referencia conceptual y estructural el conjunto de datos **IBM Telco Customer Churn**, publicado en [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). La versión de referencia contiene 7.043 registros y 21 variables relacionadas con características demográficas, servicios contratados, condiciones comerciales, facturación y abandono.

**Los registros originales de Kaggle no se incorporaron al archivo utilizado en este proyecto.** El dataset final se generó sintéticamente desde cero en Python, tomando como orientación la estructura y las dimensiones de información del conjunto de referencia.

### 3.2 Construcción del conjunto sintético

El archivo `telecom_churn_10000.csv` contiene:

* **10.000 registros sintéticos.**
* **35 variables.**
* Identificadores secuenciales desde `TELCO-00001` hasta `TELCO-10000`.
* Una semilla aleatoria (`np.random.seed(42)`) para facilitar la reproducibilidad de la generación.

La generación utilizó distribuciones aleatorias y reglas de simulación para representar características demográficas, permanencia, servicios, contratación, facturación y comportamiento de los clientes.

Se tomaron como referencia conceptual las 21 dimensiones del conjunto IBM Telco Customer Churn, con adaptaciones de nombres y categorías, y se incorporaron 14 variables adicionales:

| Dimensión                 | Variables adicionales                                                |
| ------------------------- | -------------------------------------------------------------------- |
| Demográfica y permanencia | `Age`, `TenureCohort`                                                |
| Servicios                 | `ServiceCount`                                                       |
| Comercial                 | `AutoPay`, `DiscountRate`                                            |
| Atención y operación      | `SupportTickets`, `Complaints`, `CustomerContacts`, `ServiceOutages` |
| Experiencia               | `NPS`, `CSAT`                                                        |
| Valor del cliente         | `EstimatedCLV`, `AvgTicketCost`                                      |
| Abandono                  | `ChurnReason`                                                        |

Algunas variables se generaron mediante distribuciones aleatorias y otras se calcularon a partir de características del registro. La variable `Churn` se construyó mediante una función logística con un componente aleatorio. `ChurnReason` se asignó mediante reglas condicionales asociadas a la clasificación de abandono.

Durante la generación se incorporaron valores faltantes de manera controlada, aproximadamente en el 1,5 % de los registros de `PaymentMethod`, `MonthlyCharges`, `NPS` y `CSAT`.

### 3.3 Alcance y limitaciones de los datos

El conjunto es **completamente sintético**: no representa registros reales de clientes ni una muestra estadística de una empresa de telecomunicaciones. Sus distribuciones, relaciones entre variables y resultados de abandono dependen de las reglas utilizadas en la simulación.

En consecuencia, el proyecto permite desarrollar ejercicios académicos de exploración, preparación de datos y modelamiento, pero sus resultados no deben interpretarse como evidencia empírica del comportamiento real del mercado. Las relaciones encontradas tampoco demuestran causalidad.

## 4. Contenido del dataset

Las 35 variables cubren las siguientes dimensiones:

| Dimensión                  | Variables                                                                                                                                                                 |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Identificación             | `CustomerID`                                                                                                                                                              |
| Demográfica                | `Age`, `Gender`, `SeniorCitizen`, `Partner`, `Dependents`                                                                                                                 |
| Permanencia                | `TenureMonths`, `TenureCohort`                                                                                                                                            |
| Servicios                  | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `ServiceCount` |
| Contratación y facturación | `Contract`, `PaperlessBilling`, `AutoPay`, `PaymentMethod`, `MonthlyCharges`, `DiscountRate`, `TotalCharges`                                                              |
| Atención y operación       | `SupportTickets`, `Complaints`, `CustomerContacts`, `ServiceOutages`                                                                                                      |
| Experiencia y valor        | `NPS`, `CSAT`, `EstimatedCLV`, `AvgTicketCost`                                                                                                                            |
| Resultado                  | `Churn`, `ChurnReason`                                                                                                                                                    |

El archivo de datos utilizado por el proyecto es `data/telecom_churn_10000.csv`.

## 5. Metodología

El trabajo se organiza en las siguientes etapas:

### 5.1 Inspección y comprensión

Se revisan las dimensiones del conjunto, los nombres y tipos de variables, las primeras y últimas observaciones, las distribuciones, los valores faltantes, los duplicados y la consistencia de las categorías.

Se utilizan, entre otras, funciones de Pandas como:

* `head()` y `tail()`.
* `info()` y `describe()`.
* `isnull()` y `duplicated()`.
* `value_counts()`.

### 5.2 Limpieza y transformación

Se revisan los tipos de datos, las cadenas y categorías, los valores faltantes, los duplicados y la variable objetivo `Churn`. También se examinan posibles valores atípicos mediante criterios como el rango intercuartílico (IQR).

La identificación de valores atípicos no implica eliminarlos automáticamente: cada tratamiento debe justificarse según la naturaleza de la variable y su pertinencia analítica.

### 5.3 Análisis exploratorio y operaciones con Pandas

El análisis incluye:

* **Univariado:** distribución de edad, permanencia, cargos, valor estimado, categorías y variable objetivo.
* **Bivariado:** relaciones entre fuga y contrato, permanencia, cargos, satisfacción, reclamaciones, contactos e interrupciones.
* **Multivariado:** correlaciones entre variables numéricas, comparaciones conjuntas y tablas cruzadas.

Se emplean operaciones de Pandas como `loc`, `iloc`, `groupby()`, `pivot_table()`, `crosstab()` y `merge()`, además de estructuras de agrupación jerárquica o MultiIndex cuando corresponde.

El proyecto trabaja con un único archivo de entrada. Por ello, no se utiliza `concat()` para integrar fuentes originales distintas.

### 5.4 Visualización

Se utilizan dos bibliotecas principales:

* **Seaborn:** histogramas, gráficos de barras, diagramas de caja y mapas de calor.
* **Plotly Express:** gráficos interactivos para explorar categorías, permanencia, cargos, satisfacción y fuga.

El proyecto también incluye una aplicación desarrollada con Streamlit (`app.py`) para consultar los resultados y facilitar la exploración de la información.

### 5.5 Modelamiento y evaluación

Se entrenan y comparan tres algoritmos de clasificación:

1. Regresión Logística (`LogisticRegression`).
2. Bosque Aleatorio (`RandomForestClassifier`).
3. HistGradientBoosting (`HistGradientBoostingClassifier`).

La evaluación considera Accuracy, Precision, Recall, F1-Score y ROC-AUC. La comparación se complementa con validación cruzada utilizando F1-Score.

## 6. Prevención de fuga de información (Data Leakage)

Para reducir el riesgo de que el modelo utilice información que revele el resultado que intenta predecir:

* `CustomerID` se excluye como identificador, porque no constituye una característica explicativa del cliente.
* `ChurnReason` se excluye como predictor porque está asociado directamente con el resultado de fuga.
* Se debe revisar que las variables predictoras estén disponibles antes del momento en que se realizaría la predicción.

Existe una limitación que debe mantenerse explícita: el dataset no contiene información temporal suficiente para confirmar si variables como `NPS`, `CSAT`, `SupportTickets`, `Complaints`, `CustomerContacts`, `ServiceOutages`, `TotalCharges` y `EstimatedCLV` fueron medidas antes o después del evento de fuga.

Por tanto, excluir `ChurnReason` no garantiza que todos los predictores estén libres de Data Leakage. Una aplicación real requeriría definir una fecha de corte y verificar la disponibilidad de cada variable antes del evento que se busca anticipar.

## 7. Limitación del análisis temporal

El archivo no contiene una fecha calendario ni una marca temporal que permita establecer cuándo se registró cada observación o cuándo ocurrió la fuga.

`TenureMonths` representa la duración de permanencia del cliente, pero no equivale a una fecha de contratación, registro o abandono.

Por esta razón, es posible comparar la permanencia entre grupos, pero no realizar análisis cronológicos, remuestreo por periodos, extracción de componentes de calendario, tendencias históricas ni ventanas móviles sobre una secuencia temporal real.

Esta limitación evita presentar como análisis temporal operaciones que el dataset no permite sustentar.

## 8. Resultados de los modelos

Los siguientes resultados corresponden a la evaluación documentada en el conjunto de prueba, con el umbral de clasificación predeterminado de 0,50:

| Modelo               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| -------------------- | -------: | --------: | -----: | -------: | ------: |
| Regresión Logística  |   0,6995 |    0,6662 | 0,5600 |   0,6085 |  0,7514 |
| Random Forest        |   0,6760 |    0,6356 | 0,5228 |   0,5737 |  0,7294 |
| HistGradientBoosting |   0,6800 |    0,6359 | 0,5444 |   0,5866 |  0,7284 |

La validación cruzada reportó los siguientes F1-Score promedio:

| Modelo               | F1-Score promedio |
| -------------------- | ----------------: |
| Regresión Logística  |            0,6009 |
| Random Forest        |            0,5586 |
| HistGradientBoosting |            0,5883 |

La Regresión Logística obtuvo el mejor F1-Score y ROC-AUC en la comparación documentada, por lo que fue seleccionada como el mejor modelo experimental de este ejercicio.

### 8.1 Evaluación de un umbral de 0,35

Adicionalmente, se evaluó un umbral de clasificación de 0,35 para la Regresión Logística:

| Métrica   | Resultado |
| --------- | --------: |
| Accuracy  |   65,80 % |
| Precision |   56,51 % |
| Recall    |   78,06 % |
| F1-Score  |   65,56 % |
| ROC-AUC   |   75,14 % |

Matriz de confusión:

| Resultado                 | Cantidad |
| ------------------------- | -------: |
| Verdaderos negativos (TN) |      665 |
| Falsos positivos (FP)     |      501 |
| Falsos negativos (FN)     |      183 |
| Verdaderos positivos (TP) |      651 |

Al reducir el umbral aumentó el Recall, es decir, se identificó una mayor proporción de los clientes clasificados como casos de fuga. Al mismo tiempo, aumentaron los falsos positivos.

El umbral de 0,35 es un escenario de evaluación orientado a la sensibilidad; no se ha demostrado que sea el umbral óptimo para el negocio. Su selección operativa debería considerar el costo de las campañas de retención, el valor del cliente, la capacidad de atención y el costo de no intervenir oportunamente.

> Los resultados anteriores deben contrastarse con los reportes de la ejecución final del pipeline antes de utilizarlos como resultados definitivos de una nueva versión del proyecto.

## 9. Interpretación desde el negocio y Customer Experience

La analítica predictiva puede apoyar la priorización de acciones preventivas, la identificación de segmentos que requieren atención diferenciada y el análisis conjunto de permanencia, satisfacción, reclamaciones y comportamiento del servicio.

Sin embargo, una probabilidad de fuga no explica por sí sola la causa del abandono ni determina cuál es la acción de retención adecuada.

El modelo debe considerarse un insumo para la toma de decisiones, complementado con análisis del contexto, validación de la información y seguimiento de los resultados de las acciones implementadas.

La naturaleza sintética del dataset limita cualquier extrapolación a clientes reales. Antes de una aplicación empresarial serían necesarias validaciones con datos reales, controles de calidad, revisión de disponibilidad temporal de predictores y evaluación de impacto operativo y económico.

## 10. Estructura del repositorio

```text
proyecto-fuga-telecom/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   └── telecom_churn_10000.csv
├── notebooks/
│   └── 01_analisis_exploratorio.ipynb
├── models/
│   └── best_model.pkl
├── reports/
│   ├── metrics.json
│   └── threshold_evaluation.csv
├── src/
│   └── telecom_churn/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── evaluation.py
│       ├── features.py
│       ├── final_evaluation.py
│       ├── model.py
│       ├── predict.py
│       └── train.py
├── tests/
│   └── test_pipeline.py
├── app.py
├── dataset-card.md
├── .gitignore
├── Makefile
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── README.md
```

| Componente                   | Función                                                                             |
| ---------------------------- | ----------------------------------------------------------------------------------- |
| `data/`                      | Almacena el conjunto de datos utilizado.                                            |
| `notebooks/`                 | Contiene el análisis exploratorio y las visualizaciones.                            |
| `src/telecom_churn/`         | Contiene los módulos de carga, preparación, entrenamiento, predicción y evaluación. |
| `models/`                    | Almacena el modelo entrenado.                                                       |
| `reports/`                   | Contiene métricas y resultados de evaluación.                                       |
| `tests/`                     | Contiene las pruebas automatizadas.                                                 |
| `app.py`                     | Implementa la aplicación Streamlit.                                                 |
| `.github/workflows/ci.yml`   | Configura la integración continua con GitHub Actions.                               |
| `Makefile`                   | Automatiza tareas frecuentes.                                                       |
| `pyproject.toml` y `uv.lock` | Definen el proyecto y permiten reproducir las dependencias mediante `uv`.           |
| `dataset-card.md`            | Amplía la documentación del conjunto de datos.                                      |

## 11. Instalación y ejecución

### 11.1 Requisitos

* Python en la versión compatible con `pyproject.toml`.
* Git.
* `uv`.
* Visual Studio Code y las extensiones Python y Jupyter, si se trabajará con el notebook.

### 11.2 Clonar el repositorio

```bash
git clone https://github.com/lmesac/proyecto-fuga-telecom.git
cd proyecto-fuga-telecom
```

### 11.3 Crear y sincronizar el entorno

```bash
uv sync --locked
```

Este comando sincroniza el entorno virtual con las dependencias registradas en el proyecto y en `uv.lock`.

### 11.4 Ejecutar el notebook

Abrir `notebooks/01_analisis_exploratorio.ipynb` en Visual Studio Code, seleccionar el intérprete Python del entorno virtual `.venv` y ejecutar las celdas en orden.

### 11.5 Ejecutar verificaciones

```bash
make check
```

También pueden ejecutarse por separado:

```bash
make lint
make test
```

### 11.6 Entrenar el modelo

```bash
make train
```

### 11.7 Ejecutar la aplicación

```bash
make app
```

La aplicación se ejecuta mediante Streamlit.

## 12. Calidad y reproducibilidad

El proyecto utiliza `pytest` para ejecutar pruebas automatizadas y Ruff para verificar la calidad del código. La integración continua se configura mediante GitHub Actions para ejecutar las verificaciones definidas cuando se realizan cambios en las ramas configuradas.

El archivo `uv.lock` registra las versiones resueltas de las dependencias y permite sincronizar un entorno consistente mediante `uv sync --locked`.

La carpeta `.venv` es local y no debe versionarse en GitHub.

## 13. Conclusiones

El proyecto desarrolla un flujo de Ciencia de Datos para explorar y analizar la fuga de clientes en un escenario sintético de telecomunicaciones, desde la inspección y preparación de los datos hasta la evaluación de modelos de clasificación.

Entre los modelos documentados, la Regresión Logística presentó el mejor desempeño general. La evaluación del umbral mostró que es posible aumentar el Recall, aunque esto implica aceptar más falsos positivos. La elección de un umbral operativo debe responder a criterios de negocio y no únicamente a métricas estadísticas.

Desde Customer Experience, la analítica predictiva puede contribuir a anticipar riesgos y priorizar intervenciones. No obstante, la utilidad de sus resultados depende de la calidad de los datos, la disponibilidad oportuna de las variables, la pertinencia de las acciones y la medición de sus efectos.

Dado que el conjunto de datos es sintético y no contiene una referencia temporal real, los resultados tienen alcance académico y exploratorio. No constituyen evidencia del comportamiento de clientes reales ni demuestran relaciones causales.

---

**Repositorio:** https://github.com/lmesac/proyecto-fuga-telecom
