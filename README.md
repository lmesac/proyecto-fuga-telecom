# Proyecto Final: Predicción de Fuga de Clientes en Telecomunicaciones (Telecom Churn)

## 1. Información general

**Programa:** Especialización en Ciencia de Datos e Inteligencia Artificial
**Institución:** Universidad de Medellín
**Módulo:** Introducción a la Ciencia de Datos
**Periodo:** 2026-II

**Integrantes:**

* Jorge Díaz
* Federico Martínez
* Lida Mesa

## 2. Descripción del proyecto

Este proyecto desarrolla un flujo de Ciencia de Datos orientado al análisis y la predicción de la fuga de clientes (*Customer Churn*) en el sector de telecomunicaciones.

El objetivo es explorar las características de los clientes, identificar relaciones entre variables sociodemográficas, comerciales, financieras y de experiencia, y entrenar modelos de clasificación que permitan estimar la probabilidad de abandono del servicio.

## 2.1 Origen y transformación del conjunto de datos

El proyecto parte del conjunto de datos Telco Customer Churn, publicado por Blastchar en Kaggle y basado en IBM Sample Data Sets. La fuente original contiene 7.043 registros y 21 columnas, con información demográfica, servicios contratados, características de facturación y la variable objetivo Churn.

Para ampliar el alcance del análisis, se construyó una versión enriquecida sintéticamente del conjunto de datos, con 10.000 registros y 35 columnas. La transformación incorporó nuevas dimensiones relacionadas con servicios, antigüedad, comportamiento operativo, satisfacción y valor del cliente, entre ellas NPS, CSAT, tickets de soporte, quejas, contactos, interrupciones del servicio y valor estimado del ciclo de vida del cliente.

Esta ampliación tiene como propósito académico explorar relaciones entre la fuga de clientes y variables adicionales de experiencia y operación. Las variables y registros sintéticos deben interpretarse como datos simulados y no como observaciones reales de clientes.

Por tanto, los resultados del análisis exploratorio y de los modelos predictivos corresponden exclusivamente al conjunto de datos enriquecido utilizado en el proyecto. No deben interpretarse como estimaciones representativas del mercado real de telecomunicaciones.


El proyecto integra herramientas de Python, Pandas, Seaborn, Plotly y Streamlit, bajo una estructura modular que facilita la reproducibilidad, las pruebas y la evolución del análisis.

## 3. Contexto y problema de negocio

La fuga de clientes representa un desafío para las empresas de telecomunicaciones porque afecta los ingresos recurrentes, los costos de adquisición y retención, la estabilidad de la cartera y la relación con los clientes.

Identificar oportunamente los perfiles asociados con una mayor probabilidad de abandono puede ayudar a priorizar acciones de retención y orientar decisiones comerciales y de experiencia del cliente.

El problema analítico se formula como una tarea de clasificación binaria:

* **Variable objetivo:** `Churn`.
* **Clase positiva:** cliente que abandona el servicio.
* **Clase negativa:** cliente que permanece.

La pregunta central del proyecto es:

**¿Qué características de los clientes están asociadas con la fuga y qué capacidad tienen los modelos de clasificación para identificar clientes con mayor riesgo de abandono?**

Los resultados deben interpretarse como evidencia exploratoria y predictiva sobre el conjunto de datos utilizado, no como una demostración de causalidad.

## 4. Fuente y características del conjunto de datos

### 4.1 Archivo utilizado

* **Archivo:** `data/telecom_churn_10000.csv`
* **Registros:** 10.000
* **Variables:** 35
* **Formato:** CSV
* **Variable objetivo:** `Churn`

### 4.2 Procedencia y trazabilidad

Se identifica como referencia relacionada el conjunto histórico Telco Customer Churn publicado en Kaggle:

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

### 4.3 Dimensiones de información

El conjunto de datos contiene cuatro grupos principales de información:

1. **Características sociodemográficas:** edad, género, pareja y dependientes.
2. **Servicios contratados:** telefonía, internet, seguridad, soporte y servicios de entretenimiento.
3. **Información comercial y financiera:** antigüedad, contratos, cargos, descuentos y métodos de pago.
4. **Comportamiento y experiencia:** contactos, reclamaciones, interrupciones, satisfacción, valor estimado del cliente y fuga.

### 4.4 Diccionario de variables

| Variable         | Descripción                             | Tipo                  |
| ---------------- | --------------------------------------- | --------------------- |
| CustomerID       | Identificador del cliente               | Identificador         |
| Age              | Edad del cliente, en años               | Numérica              |
| SeniorCitizen    | Indicador de cliente senior             | Numérica/binaria      |
| Gender           | Género                                  | Categórica            |
| Partner          | Indica si tiene pareja                  | Categórica            |
| Dependents       | Indica si tiene dependientes            | Categórica            |
| TenureMonths     | Meses de permanencia                    | Numérica              |
| TenureCohort     | Grupo o rango de permanencia            | Categórica            |
| PhoneService     | Servicio telefónico                     | Categórica            |
| MultipleLines    | Disponibilidad de múltiples líneas      | Categórica            |
| InternetService  | Tipo de servicio de internet            | Categórica            |
| OnlineSecurity   | Servicio de seguridad en línea          | Categórica            |
| OnlineBackup     | Servicio de respaldo en línea           | Categórica            |
| DeviceProtection | Protección de dispositivos              | Categórica            |
| TechSupport      | Servicio de soporte técnico             | Categórica            |
| StreamingTV      | Servicio de televisión por streaming    | Categórica            |
| StreamingMovies  | Servicio de películas por streaming     | Categórica            |
| ServiceCount     | Cantidad de servicios contratados       | Numérica              |
| Contract         | Tipo de contrato                        | Categórica            |
| PaperlessBilling | Facturación electrónica                 | Categórica            |
| AutoPay          | Pago automático                         | Categórica            |
| PaymentMethod    | Método de pago                          | Categórica            |
| MonthlyCharges   | Cargos mensuales                        | Numérica              |
| DiscountRate     | Tasa o proporción de descuento          | Numérica              |
| TotalCharges     | Cargos acumulados                       | Numérica              |
| SupportTickets   | Cantidad de solicitudes de soporte      | Numérica              |
| Complaints       | Cantidad de reclamaciones               | Numérica              |
| CustomerContacts | Cantidad de contactos del cliente       | Numérica              |
| ServiceOutages   | Cantidad de interrupciones del servicio | Numérica              |
| NPS              | Indicador Net Promoter Score            | Numérica              |
| CSAT             | Indicador de satisfacción del cliente   | Numérica              |
| EstimatedCLV     | Valor de vida estimado del cliente      | Numérica              |
| AvgTicketCost    | Costo promedio por ticket               | Numérica              |
| Churn            | Indicador de fuga del cliente           | Categórica / objetivo |
| ChurnReason      | Motivo asociado a la fuga               | Categórica            |

## 5. Objetivos

### 5.1 Objetivo general

Desarrollar un flujo reproducible de Ciencia de Datos para explorar las características de los clientes de telecomunicaciones, identificar factores asociados con la fuga y evaluar modelos de clasificación que permitan estimar el riesgo de abandono.

### 5.2 Objetivos específicos

* Comprender la estructura, las características y la calidad del conjunto de datos.
* Identificar y documentar valores faltantes, duplicados, inconsistencias y posibles valores atípicos.
* Limpiar y transformar las variables utilizando Pandas.
* Aplicar técnicas de análisis exploratorio univariado, bivariado y multivariado.
* Utilizar operaciones avanzadas de Pandas para agrupar, combinar, indexar y resumir información.
* Construir visualizaciones estáticas e interactivas con Seaborn y Plotly.
* Entrenar y comparar modelos de clasificación.
* Evaluar el desempeño de los modelos y analizar el efecto de diferentes umbrales de clasificación.
* Comunicar los resultados mediante un tablero y conclusiones orientadas al negocio.

## 6. Metodología de trabajo

El desarrollo se organiza en las siguientes etapas:

### 6.1 Comprensión e inspección de los datos

Se realiza la lectura del archivo CSV y la revisión de:

* Dimensiones del conjunto de datos.
* Nombres y tipos de variables.
* Registros iniciales y finales.
* Distribución de variables numéricas y categóricas.
* Valores faltantes.
* Registros duplicados.
* Consistencia de las categorías.

Se utilizan funciones de Pandas como `head()`, `tail()`, `info()`, `describe()`, `isnull()`, `duplicated()` y `value_counts()`.

### 6.2 Limpieza y transformación

El proceso contempla:

* Normalización de cadenas de texto y categorías.
* Revisión de tipos de datos.
* Identificación de valores faltantes y duplicados.
* Normalización de la variable objetivo `Churn`.
* Creación de una representación binaria de la variable objetivo para análisis estadístico.
* Revisión de valores atípicos mediante criterios como el rango intercuartílico (IQR).

La identificación de valores atípicos no implica su eliminación automática. Cada decisión debe justificarse según la naturaleza de la variable y su pertinencia para el análisis.

### 6.3 Prevención de Data Leakage

Para evitar que el modelo utilice información que revele directamente el resultado que intenta predecir:

* `CustomerID` se excluye como identificador, porque no representa una característica explicativa del cliente.
* `ChurnReason` se excluye como predictor, porque describe información asociada al resultado de fuga.
* Se revisa la disponibilidad temporal de las variables utilizadas para entrenar el modelo.

Existe una limitación importante: no se cuenta con información suficiente para confirmar si variables como `NPS`, `CSAT`, `SupportTickets`, `Complaints`, `CustomerContacts`, `ServiceOutages`, `TotalCharges` y `EstimatedCLV` fueron medidas antes o después del evento de fuga.

Por tanto, la exclusión de `ChurnReason` no garantiza por sí sola que todo el conjunto de predictores esté libre de fuga de información. Para una implementación real sería necesario establecer una fecha de corte y verificar que cada predictor esté disponible antes del momento de predicción.

### 6.4 Análisis exploratorio de datos (EDA)

El análisis se desarrolla en tres niveles:

**Univariado**

* Distribución de edad y permanencia.
* Distribución de cargos y valor estimado del cliente.
* Frecuencia de categorías de contrato, servicios y métodos de pago.
* Distribución de la variable objetivo.

**Bivariado**

* Relación entre contrato y fuga.
* Relación entre permanencia y fuga.
* Comparación de cargos mensuales entre clientes que permanecen y clientes que abandonan.
* Comparación de NPS y CSAT según la condición de fuga.
* Comportamiento de reclamaciones, contactos e interrupciones del servicio frente a la fuga.

**Multivariado**

* Correlaciones entre variables numéricas.
* Comparación conjunta de variables de experiencia, permanencia y facturación.
* Tablas cruzadas para analizar la relación entre variables categóricas y la fuga.

Las visualizaciones y tablas del notebook permiten identificar patrones y formular hipótesis para posteriores análisis.

### 6.5 Pandas avanzado

El notebook incorpora operaciones de Pandas para demostrar el manejo y análisis estructurado de los datos:

* `loc` e `iloc`: selección de registros y variables.
* `groupby()`: agrupación y agregación de información.
* `pivot_table()`: construcción de tablas dinámicas.
* `crosstab()`: análisis de frecuencias y relaciones entre categorías.
* MultiIndex o agrupaciones jerárquicas: análisis por más de una dimensión.
* `merge()`: combinación de tablas derivadas cuando corresponde.

No se utiliza `concat()` para integrar múltiples fuentes originales porque el proyecto trabaja con un único archivo de entrada. Esta decisión evita generar combinaciones artificiales de datos sin una necesidad analítica.

### 6.6 Limitación del análisis temporal

La guía del proyecto solicita trabajar con una variable de fecha o tiempo y desarrollar análisis de patrones temporales.

El archivo utilizado no contiene una variable de fecha calendario ni una marca temporal que permita establecer cuándo ocurrió cada observación o evento.

`TenureMonths` representa el tiempo de permanencia acumulado del cliente, pero no equivale a una fecha de registro, fecha de contratación o fecha de fuga.

Por esta razón, el proyecto puede analizar la permanencia como variable de duración y comparar sus valores entre grupos, pero no puede realizar de forma válida:

* Conversión de una fecha real a `datetime`.
* Construcción de un índice temporal.
* Remuestreo por días, meses o años.
* Extracción de componentes calendario.
* Cálculo de tendencias cronológicas.
* Comparaciones entre periodos históricos.
* Ventanas móviles (`rolling`) sobre una secuencia temporal real.

Esta limitación se documenta para no presentar como análisis temporal una operación que el dataset no permite sustentar.

## 7. Visualización y análisis de resultados

### 7.1 Seaborn

Se utiliza Seaborn para construir visualizaciones estáticas que facilitan la interpretación de las variables:

* Histogramas para variables numéricas.
* Gráficos de barras para variables categóricas.
* Diagramas de caja para comparar distribuciones y examinar valores atípicos.
* Mapas de calor para explorar correlaciones.
* Gráficos comparativos de variables de experiencia y comportamiento según la fuga.

### 7.2 Plotly

Se utiliza Plotly Express para construir visualizaciones interactivas que permiten explorar relaciones entre variables mediante:

* Información contextual al pasar el cursor (*hover*).
* Colores para diferenciar categorías.
* Comparaciones entre grupos.
* Exploración de permanencia, cargos, satisfacción y fuga.

### 7.3 Dashboard

El proyecto incorpora una interfaz desarrollada con Streamlit en `app.py`.

El tablero permite consultar los resultados del proyecto y facilita la exploración de información relacionada con la fuga de clientes y el comportamiento del modelo.

La visualización de indicadores y resultados debe interpretarse de acuerdo con los datos y métricas efectivamente calculados por el proyecto.

## 8. Modelamiento predictivo

Se entrenaron y compararon tres algoritmos de clasificación:

1. Regresión Logística (`Logistic Regression`).
2. Bosque Aleatorio (`Random Forest`).
3. HistGradientBoosting (`HistGradientBoostingClassifier`).

La evaluación contempla métricas de clasificación y capacidad de discriminación:

* Accuracy: proporción de predicciones correctas.
* Precision: proporción de predicciones positivas que realmente corresponden a clientes que abandonan.
* Recall: proporción de clientes que abandonan y que fueron identificados por el modelo.
* F1-Score: equilibrio entre precisión y recall.
* ROC-AUC: capacidad de discriminación entre las clases.

La selección del modelo se apoya en la comparación de resultados y en la evaluación de F1 mediante validación cruzada.

## 9. Resultados de la evaluación de modelos

### 9.1 Comparación de modelos

Los resultados de prueba obtenidos con el umbral de clasificación predeterminado de 0,50 son:

| Modelo               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| -------------------- | -------: | --------: | -----: | -------: | ------: |
| Regresión Logística  |   0,6995 |    0,6662 | 0,5600 |   0,6085 |  0,7514 |
| Random Forest        |   0,6760 |    0,6356 | 0,5228 |   0,5737 |  0,7294 |
| HistGradientBoosting |   0,6800 |    0,6359 | 0,5444 |   0,5866 |  0,7284 |

*Nota: los valores se presentan redondeados a cuatro decimales. Deben corresponder a los reportes generados por la ejecución final del pipeline.*

### 9.2 Interpretación comparativa

**Regresión Logística**

Presentó el mejor resultado general entre los tres modelos evaluados, con un F1-Score de 0,6085 y un ROC-AUC de 0,7514.

También obtuvo el mejor equilibrio entre precisión y sensibilidad según el umbral predeterminado de 0,50.

Su resultado la convierte en el modelo seleccionado para continuar el análisis experimental.

**Random Forest**

Obtuvo un F1-Score de 0,5737 y un ROC-AUC de 0,7294.

Su recall fue de 0,5228, el menor entre los tres modelos, lo que indica que identificó una proporción inferior de los clientes que efectivamente abandonaron el servicio.

**HistGradientBoosting**

Obtuvo un F1-Score de 0,5866 y un ROC-AUC de 0,7284.

Su recall de 0,5444 fue superior al de Random Forest, aunque su desempeño general fue inferior al de la Regresión Logística.

### 9.3 Selección del modelo

La Regresión Logística fue seleccionada como el mejor modelo experimental de este ejercicio, teniendo en cuenta su desempeño comparativo y los resultados de validación cruzada.

La validación cruzada reportó los siguientes F1-Score promedio:

| Modelo               | F1-Score promedio en validación cruzada |
| -------------------- | --------------------------------------: |
| Regresión Logística  |                                  0,6009 |
| Random Forest        |                                  0,5586 |
| HistGradientBoosting |                                  0,5883 |

La selección no significa que el modelo esté listo para producción. Su implementación real requeriría validación temporal, análisis de costos, evaluación de estabilidad y verificación de la disponibilidad de los predictores antes del evento de fuga.

## 10. Evaluación del umbral de clasificación

Además de comparar los modelos con el umbral predeterminado de 0,50, se evaluó el efecto de utilizar un umbral de 0,35 para la Regresión Logística.

El cambio busca aumentar la sensibilidad del modelo para identificar una mayor proporción de clientes que podrían abandonar el servicio.

Resultados obtenidos en el conjunto de prueba con umbral de 0,35:

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

La reducción del umbral incrementó el recall, pero también aumentó la cantidad de falsos positivos.

Desde la perspectiva del negocio, esto significa que se identifica una mayor proporción de clientes que abandonan, aunque también se clasifica como potencialmente riesgoso a un número importante de clientes que no abandonan.

La elección de un umbral operativo debe considerar el costo de las campañas de retención, el valor del cliente, la capacidad de atención y el costo de no intervenir oportunamente. Por ello, 0,35 se considera un escenario de evaluación orientado a sensibilidad, no un umbral óptimo de negocio demostrado.

## 11. Interpretación desde Customer Experience y negocio

Los resultados permiten plantear una oportunidad para integrar analítica predictiva y gestión de experiencia del cliente.

La identificación de clientes con mayor probabilidad de fuga puede apoyar:

* La priorización de acciones preventivas de retención.
* La revisión de condiciones comerciales y contractuales.
* La identificación de segmentos que requieren atención diferenciada.
* El análisis conjunto de permanencia, satisfacción, reclamaciones y comportamiento de servicio.
* La asignación de recursos de retención de acuerdo con el riesgo y el valor potencial del cliente.

Sin embargo, una probabilidad de fuga no explica por sí sola la causa del abandono ni determina cuál es la acción de retención adecuada.

El modelo debe ser un insumo para la toma de decisiones, complementado con análisis del contexto, validación de información y seguimiento de los resultados de las acciones implementadas.

## 12. Limitaciones del proyecto

Se identifican las siguientes limitaciones:

1. **Ausencia de fecha calendario:** el conjunto no permite desarrollar análisis cronológicos, remuestreo ni tendencias históricas.
2. **Disponibilidad temporal de predictores:** no se conoce con precisión si todas las variables de experiencia y comportamiento fueron registradas antes del evento de fuga.
3. **Umbral de decisión:** no se cuenta con una matriz de costos del negocio que permita establecer el umbral económicamente óptimo.
4. **Interpretación causal:** las relaciones observadas no demuestran que una variable cause directamente la fuga.

Estas limitaciones deben considerarse antes de trasladar los resultados a una operación real.

## 13. Estructura del proyecto

```text
proyecto-fuga-telecom/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   └── telecom_churn_10000.csv
│
├── notebooks/
│   └── 01_analisis_exploratorio.ipynb
│
├── models/
│   └── best_model.pkl
│
├── reports/
│   ├── metrics.json
│   └── threshold_evaluation.csv
│
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
│
├── tests/
│   └── test_pipeline.py
│
├── app.py
├── dataset-card.md
├── .gitignore
├── Makefile
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── README.md
```

### 13.1 Descripción de componentes

| Componente                              | Función                                                                |
| --------------------------------------- | ---------------------------------------------------------------------- |
| `data/`                                 | Almacena el conjunto de datos utilizado.                               |
| `notebooks/`                            | Contiene el análisis exploratorio y las visualizaciones.               |
| `src/telecom_churn/config.py`           | Centraliza parámetros y rutas del proyecto.                            |
| `src/telecom_churn/data.py`             | Gestiona la lectura y preparación de los datos.                        |
| `src/telecom_churn/features.py`         | Contiene la preparación de variables y el preprocesamiento.            |
| `src/telecom_churn/model.py`            | Define y entrena los modelos predictivos.                              |
| `src/telecom_churn/evaluation.py`       | Gestiona la evaluación de los modelos.                                 |
| `src/telecom_churn/final_evaluation.py` | Ejecuta la evaluación final y los análisis asociados.                  |
| `src/telecom_churn/predict.py`          | Contiene la lógica de predicción.                                      |
| `models/`                               | Almacena el artefacto del modelo entrenado.                            |
| `reports/`                              | Contiene reportes de métricas y evaluación de umbrales.                |
| `tests/`                                | Contiene las pruebas automatizadas del proyecto.                       |
| `app.py`                                | Implementa la interfaz de Streamlit.                                   |
| `.github/workflows/ci.yml`              | Configura la integración continua con GitHub Actions.                  |
| `Makefile`                              | Automatiza tareas frecuentes.                                          |
| `pyproject.toml`                        | Define los metadatos, dependencias y configuración del proyecto.       |
| `requirements.txt`                      | Documenta las dependencias para instalación mediante pip.              |
| `uv.lock`                               | Registra versiones resueltas para reproducir el entorno con uv.        |
| `dataset-card.md`                       | Documenta las características y consideraciones del conjunto de datos. |

## 14. Requisitos e instalación

### 14.1 Requisitos previos

* Python 3.10 o superior.
* Git.
* uv como gestor de dependencias.
* Visual Studio Code con las extensiones Python y Jupyter para trabajar con el notebook.

### 14.2 Clonar el repositorio

```bash
git clone https://github.com/lmesac/proyecto-fuga-telecom.git
cd proyecto-fuga-telecom
```

### 14.3 Crear y sincronizar el entorno

```bash
uv sync --locked
```

Este comando crea o actualiza el entorno virtual y sincroniza las dependencias definidas por el proyecto.

### 14.4 Ejecutar el notebook

Abrir en Visual Studio Code:

```text
notebooks/01_analisis_exploratorio.ipynb
```

Seleccionar el intérprete Python del entorno virtual `.venv` y ejecutar las celdas en orden.

### 14.5 Ejecutar pruebas

```bash
make check
```

Este comando ejecuta las verificaciones de calidad del código y las pruebas automatizadas.

También es posible ejecutar cada tarea por separado:

```bash
make lint
make test
```

### 14.6 Entrenar el modelo

```bash
make train
```

El entrenamiento ejecuta el módulo correspondiente y genera los artefactos definidos por la configuración del proyecto.

### 14.7 Ejecutar la aplicación

```bash
make app
```

La aplicación se ejecuta mediante Streamlit y permite consultar la interfaz del proyecto.

## 15. Pruebas y reproducibilidad

El proyecto utiliza `pytest` para verificar componentes del pipeline.

Las pruebas contemplan:

* Carga de datos.
* Preparación de variables.
* Identificación de características.
* Definición de modelos.
* Entrenamiento del pipeline.

La calidad del código se verifica mediante Ruff.

La integración continua se configura mediante GitHub Actions, con el propósito de ejecutar las verificaciones automatizadas cuando se realizan cambios en las ramas configuradas.

El archivo `uv.lock` permite mantener versiones de dependencias consistentes al sincronizar el entorno mediante uv.

La carpeta `.venv` no debe incluirse en la entrega ni versionarse en GitHub.

## 16. Conclusiones

El proyecto permitió desarrollar un flujo de Ciencia de Datos para comprender y analizar la fuga de clientes en telecomunicaciones, desde la inspección y preparación de los datos hasta el entrenamiento y evaluación de modelos predictivos.

Entre los modelos evaluados, la Regresión Logística presentó el mejor desempeño general en las métricas comparadas y fue seleccionada para continuar el análisis.

La evaluación de umbrales mostró que es posible aumentar la sensibilidad del modelo, aunque esto implica incrementar los falsos positivos. Por tanto, la selección de un umbral debe responder a criterios operativos y económicos, no únicamente estadísticos.

Desde la perspectiva de Customer Experience, la analítica predictiva puede ayudar a anticipar riesgos y priorizar intervenciones, pero su valor depende de la calidad de los datos, la oportunidad de las variables, la pertinencia de las acciones y la medición de sus resultados.

## 17. Repositorio

Código fuente y documentación:

https://github.com/lmesac/proyecto-fuga-telecom
