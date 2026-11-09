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

### 2.1 Uso previsto y criterios de evaluación

El usuario previsto es un analista de atención y retención. La decisión que se busca apoyar es priorizar qué clientes revisar, a partir de la probabilidad estimada de abandono y del contexto de atención.

El objetivo analítico es comparar clasificadores sobre el dataset sintético mediante un procedimiento reproducible. El criterio implementado para seleccionar el modelo es el mayor F1 promedio en validación cruzada sobre entrenamiento. El desempeño final se describe con precisión, recall, F1, exactitud y ROC-AUC en prueba.

Como criterio de negocio para una evaluación posterior, se propone medir cuántos abandonos se identifican dentro de la cantidad de clientes que el equipo pueda contactar. La capacidad de atención, los costos y una meta de retención deben acordarse antes de evaluar una intervención real. En este ejercicio, esos resultados de negocio quedan pendientes de validación.

Los principales riesgos son el alcance sintético del dataset, la disponibilidad temporal de las variables y la ausencia de costos operativos para justificar una política de intervención.

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

La identificación de valores atípicos requiere revisar su naturaleza antes de decidir un tratamiento. En el flujo actual se conservan las filas del archivo y el preprocesamiento se realiza mediante un `ColumnTransformer` dentro del `Pipeline`.

| Tipo de variable | Tratamiento implementado |
| --- | --- |
| Numérica | Completar faltantes con la mediana y estandarizar con `StandardScaler`. |
| Categórica | Completar faltantes con la categoría `missing` y codificar con `OneHotEncoder`, que admite categorías nuevas mediante `handle_unknown="ignore"`. |

Se excluyen `CustomerID`, `ChurnReason` y la etiqueta `Churn` de las entradas del modelo. Quedan 32 predictores antes de codificarlos. La etiqueta se transforma a 0 para permanencia y 1 para abandono.

Las medianas, el escalado y la codificación se ajustan con entrenamiento. Durante la validación cruzada, el pipeline se vuelve a ajustar dentro de cada partición. El pipeline guardado reúne estas transformaciones y el clasificador para utilizarlos en predicción.

### 5.3 Análisis exploratorio y operaciones con Pandas

El análisis incluye:

* **Univariado:** distribución de edad, permanencia, cargos, valor estimado, categorías y variable objetivo.
* **Bivariado:** relaciones entre fuga y contrato, permanencia, cargos, satisfacción, reclamaciones, contactos e interrupciones.
* **Multivariado:** correlaciones entre variables numéricas, comparaciones conjuntas y tablas cruzadas.

Se emplean operaciones de Pandas como `loc`, `iloc`, `groupby()`, `pivot_table()`, `crosstab()` y `merge()`, además de estructuras de agrupación jerárquica o MultiIndex cuando corresponde.

El proyecto trabaja con un único archivo de entrada. Por ello, no se utiliza `concat()` para integrar fuentes originales distintas.

#### Hallazgos del análisis exploratorio

| Hallazgo en los datos sintéticos | Interpretación para el proyecto |
| --- | --- |
| 4.169 abandonos (41,69%) y 5.831 permanencias (58,31%). | Mantener particiones estratificadas y valorar la detección de abandonos junto a la exactitud. |
| Abandono de 50,13% en contratos mensuales (5.526 registros) y 29,74% en contratos de dos años (2.021). | Evaluar el aporte del contrato como predictor. |
| Antigüedad mediana de 26 meses entre abandonos y 40 entre permanencias. | Examinar antigüedad y sus relaciones con cargos acumulados y variables derivadas. |
| 150 faltantes en cada una de cuatro columnas, que afectan a 588 filas distintas, y cero filas duplicadas. | Conservar filas y tratar faltantes dentro del pipeline. |

Estos hallazgos describen el CSV completo y las asociaciones de la simulación. Los tamaños de grupo y los límites de interpretación se desarrollan en el [notebook del EDA](notebooks/01_analisis_exploratorio.ipynb).

### 5.4 Visualización

Se utilizan dos bibliotecas principales:

* **Seaborn:** histogramas, gráficos de barras, diagramas de caja y mapas de calor.
* **Plotly Express:** gráficos interactivos para explorar categorías, permanencia, cargos, satisfacción y fuga.

El proyecto también incluye una aplicación desarrollada con Streamlit (`app/app_j.py`) para consultar los resultados y facilitar la exploración de la información.

### 5.5 Modelamiento y evaluación

Se entrenan y comparan tres algoritmos de clasificación:

1. Regresión Logística (`LogisticRegression`).
2. Bosque Aleatorio (`RandomForestClassifier`).
3. HistGradientBoosting (`HistGradientBoostingClassifier`).

Se realiza una separación estratificada con semilla 42, destinando 8.000 registros (80%) a entrenamiento y 2.000 (20%) a prueba. Sobre entrenamiento se ejecuta validación cruzada estratificada de cinco particiones, con F1 como criterio de selección.

Cada candidato se ajusta también con los 8.000 registros de entrenamiento y se evalúa sobre prueba. El código selecciona el ganador mediante `cv_f1_mean` y guarda su pipeline. Las métricas de prueba se utilizan para reportar el desempeño final.

La evaluación considera Accuracy, Precision, Recall, F1-Score y ROC-AUC. Al revisar futuras mejoras, las decisiones de selección deben apoyarse en entrenamiento y validación, reservando prueba para la evaluación final.

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

Los siguientes resultados corresponden a la evaluación documentada en el conjunto de prueba, con el umbral de clasificación de 0,50 utilizado en la comparación de modelos:

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

La Regresión Logística fue seleccionada por su mayor F1 promedio de validación cruzada en entrenamiento, 0,6009. También obtuvo el mayor F1 y ROC-AUC de prueba entre los modelos del reporte. Los resultados completos están en [reports/metrics.json](reports/metrics.json).

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

#### Diferencia entre los reportes de evaluación

El archivo [reports/threshold_evaluation.csv](reports/threshold_evaluation.csv) contiene los resultados de la comparación de umbrales mediante validación cruzada sobre los 8.000 registros del conjunto de entrenamiento. En cambio, las métricas y la matriz de confusión presentadas en esta sección corresponden a la evaluación final del umbral 0,35 sobre los 2.000 registros del conjunto de prueba. Por esta diferencia en los conjuntos de evaluación, sus resultados no deben compararse como si provinieran de la misma muestra.

Al reducir el umbral aumentó el Recall, es decir, se identificó una mayor proporción de los clientes clasificados como casos de fuga. Al mismo tiempo, aumentaron los falsos positivos.

El umbral de 0,35 es un escenario de evaluación orientado a la sensibilidad; no se ha demostrado que sea el umbral óptimo para el negocio. Su selección operativa debería considerar el costo de las campañas de retención, el valor del cliente, la capacidad de atención y el costo de no intervenir oportunamente.

> Los resultados anteriores deben contrastarse con los reportes de la ejecución final del pipeline antes de utilizarlos como resultados definitivos de una nueva versión del proyecto.

### 8.2 Umbral de clasificación y niveles de riesgo

La probabilidad estimada se utiliza con reglas distintas según la salida que se consulta.

| Uso | Regla actual |
| --- | --- |
| Comparación de los tres modelos | Métricas de clasificación con umbral 0,50. |
| Predicción de abandono | `Probabilidad_Fuga >= 0,35` produce `Prediccion_Fuga = 1` y el resultado «Posible fuga». Por debajo, produce 0 y «No fuga». |
| Riesgo bajo | Probabilidad menor que 0,35. |
| Riesgo medio | Probabilidad mayor o igual que 0,35 y menor que 0,65. |
| Riesgo alto | Probabilidad mayor o igual que 0,65. |

Por ejemplo, una probabilidad de 0,50 se clasifica como «Posible fuga» y riesgo «Medio». La clasificación binaria señala abandono estimado y el nivel de riesgo ayuda a ordenar la atención.

El umbral de predicción se define en `src/telecom_churn/config.py`. Los límites de riesgo están en `src/telecom_churn/presentation/risk.py`. Son reglas de la demostración cuya utilidad comercial requiere validación con capacidad de atención, costos y resultados de las intervenciones.

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
│       ├── train.py
│       ├── presentation/
│       │   ├── __init__.py
│       │   └── risk.py
│       └── services/
│           ├── __init__.py
│           └── churn_service.py
├── tests/
│   ├── test_pipeline.py
│   └── test_risk_service.py
├── app/
│   └── app_j.py
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
| `app/app_j.py` | Implementa la aplicación Streamlit para evaluación individual y masiva del riesgo de abandono. |
| `src/telecom_churn/services/` | Combina la predicción con la clasificación de riesgo y las recomendaciones. |
| `src/telecom_churn/presentation/` | Define los niveles de riesgo y sus recomendaciones. |
| `.github/workflows/ci.yml`   | Configura la integración continua con GitHub Actions.                               |
| `Makefile`                   | Automatiza tareas frecuentes.                                                       |
| `pyproject.toml` y `uv.lock` | Definen el proyecto y permiten reproducir las dependencias mediante `uv`.           |
| `dataset-card.md`            | Amplía la documentación del conjunto de datos.                                      |

## 11. Instalación y ejecución

### 11.1 Requisitos

* Python 3.12, utilizado por la integración continua. El proyecto declara compatibilidad desde Python 3.10 en `pyproject.toml`.
* Git.
* `uv`.
* `make` para utilizar los comandos del Makefile. En Windows puede utilizarse la alternativa directa con `uv` indicada en cada paso.
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

Este comando sincroniza el entorno virtual con las dependencias registradas en el proyecto y en `uv.lock`. Ejecutar los comandos siguientes desde la raíz del repositorio, donde se encuentran `pyproject.toml` y `Makefile`.

En la versión actual se necesita `data/telecom_churn_10000.csv`, incluido en el repositorio. El procedimiento de generación reproducible está pendiente de incorporación por el equipo. La [ficha del dataset](dataset-card.md) describe su procedencia y alcance.

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

Alternativa directa en PowerShell o en una terminal sin `make`:

```sh
uv run ruff check src tests
uv run pytest -v
```

### 11.6 Entrenar el modelo

```bash
make train
```

Alternativa directa:

```sh
uv run python -m telecom_churn.train
```

El entrenamiento guarda el pipeline ganador en `models/best_model.pkl` y la comparación de modelos en `reports/metrics.json`. Al repetirlo, se actualizan ambos archivos.

### 11.7 Ejecutar la aplicación

```bash
make app
```

Alternativa directa:

```sh
uv run streamlit run app/app_j.py
```

Abrir la dirección indicada como **Local URL**, normalmente `http://localhost:8501`. Mantener la terminal abierta durante el uso y pulsar `Ctrl+C` para detener el servicio. Se necesitan el CSV del proyecto y el modelo entrenado.

### 11.8 Utilizar la aplicación

**Evaluación individual**

1. Introducir antigüedad, contrato, cargo mensual, servicio de internet, método de pago, pago automático, tickets, quejas, NPS y CSAT.
2. Pulsar **Evaluar riesgo**.
3. Consultar la probabilidad estimada de fuga, el nivel de riesgo y la recomendación.
4. Abrir **Ver datos enviados al modelo** para revisar el registro completo utilizado.

**Ejemplo ilustrativo de resultado**

Si la aplicación muestra una probabilidad de fuga del **50 %**, el resultado se interpreta así.

| Salida | Valor del ejemplo | Interpretación |
| --- | --- | --- |
| Probabilidad de fuga | 50 % | Estimación de abandono generada por el modelo para los datos enviados. |
| Nivel de riesgo | Medio | La probabilidad está entre 35 % inclusive y 65 % exclusivo. |
| Recomendación | Revisar la experiencia del cliente y considerar una acción preventiva de retención. | Orientación asociada al nivel medio en la demostración. |

Con el umbral de clasificación de 35 %, este registro también se clasifica como **Posible fuga** en la salida del servicio de predicción. El formulario muestra la probabilidad, el nivel de riesgo y la recomendación.

El 50 % se utiliza para ilustrar la lectura del resultado. El valor que aparezca al probar la app dependerá de los datos del cliente y del modelo guardado. Su interpretación tiene el alcance académico del dataset sintético. La sección 8.2 detalla los umbrales utilizados.

Las otras variables se completan con medianas numéricas y modas categóricas calculadas a partir del CSV de referencia completo. Por ello, el resultado depende tanto de los datos introducidos como de los valores completados. La revisión del registro permite comprobar si representa el escenario que se desea analizar, especialmente en campos relacionados como antigüedad, cohorte y cargos acumulados.

**Evaluación masiva**

1. Cargar un CSV con nombres de columnas compatibles con el dataset. Para una prueba del flujo, puede seleccionarse `data/telecom_churn_10000.csv`.
2. Pulsar **Analizar clientes**.
3. Consultar el resumen por niveles de riesgo, el ranking y los diez registros con mayor puntuación.
4. Descargar los resultados mediante **Descargar resultados**.

Conviene proporcionar las 32 variables predictoras. La aplicación actual completa columnas ausentes con valores de referencia del CSV. Las celdas vacías de columnas existentes se procesan con el imputador del modelo. `CustomerID` permite identificar los resultados, mientras `Churn` y `ChurnReason` se excluyen de la predicción.

La prueba con el CSV completo demuestra el funcionamiento de la interfaz. Las métricas de desempeño se calculan mediante la separación de entrenamiento y prueba descrita en la sección 5.5.

### 11.9 Ejecutar las evaluaciones complementarias

Después de entrenar el modelo, ejecutar:

```sh
uv run python -m telecom_churn.evaluation
uv run python -m telecom_churn.final_evaluation
```

| Comando | Resultado |
| --- | --- |
| `telecom_churn.evaluation` | Compara umbrales con predicciones de validación cruzada sobre las 8.000 filas de entrenamiento y actualiza `reports/threshold_evaluation.csv`. |
| `telecom_churn.final_evaluation` | Evalúa el modelo guardado con umbral 0,35 sobre las 2.000 filas de prueba e imprime las métricas y la matriz de confusión en la terminal. |

La interpretación de las métricas debe conservar el conjunto evaluado y el umbral utilizado, para poder comparar resultados equivalentes.

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
