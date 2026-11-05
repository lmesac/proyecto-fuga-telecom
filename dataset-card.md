# Dataset Card: Telecom Churn

## 1. Descripción general

Este conjunto de datos contiene **10.000 registros sintéticos y 35 variables**, diseñados con fines académicos para explorar y analizar la fuga de clientes (*Customer Churn*) en el sector de telecomunicaciones.

El objetivo es estudiar las características demográficas, comerciales, operativas, financieras y de experiencia que pueden estar asociadas con la fuga de clientes.

**Importante:** los registros no corresponden a clientes reales ni representan una muestra estadística de una empresa de telecomunicaciones.

## 2. Procedencia y construcción

Se tomó como referencia conceptual y estructural el conjunto de datos IBM Telco Customer Churn, publicado en Kaggle, que contiene 7.043 registros y 21 variables.

Los registros originales no fueron incorporados al dataset de este proyecto. Los 10.000 registros finales fueron generados sintéticamente desde cero mediante Python, utilizando distribuciones aleatorias y reglas de simulación.

La generación utilizó la semilla `np.random.seed(42)` para facilitar su reproducibilidad.

## 3. Estructura y variables

El conjunto contiene 35 variables agrupadas en las siguientes dimensiones:

* **Identificación:** `CustomerID`.
* **Demográficas:** `Age`, `Gender`, `SeniorCitizen`, `Partner`, `Dependents`.
* **Permanencia:** `TenureMonths`, `TenureCohort`.
* **Servicios contratados:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `ServiceCount`.
* **Comerciales y facturación:** `Contract`, `PaperlessBilling`, `AutoPay`, `PaymentMethod`, `MonthlyCharges`, `DiscountRate`, `TotalCharges`.
* **Atención y operación:** `SupportTickets`, `Complaints`, `CustomerContacts`, `ServiceOutages`.
* **Experiencia y valor:** `NPS`, `CSAT`, `EstimatedCLV`, `AvgTicketCost`.
* **Resultado:** `Churn`, `ChurnReason`.

## 4. Variable objetivo

La variable objetivo es `Churn`, que identifica si el registro corresponde a un cliente que abandona el servicio (`Yes`) o permanece (`No`).

La variable `ChurnReason` representa una categoría asociada al motivo simulado de abandono. Por su relación directa con el resultado, se excluye del entrenamiento predictivo para evitar Data Leakage.

## 5. Valores faltantes y preprocesamiento

Durante la generación se incorporaron valores faltantes de manera controlada, aproximadamente en el 1,5 % de los registros de las variables `PaymentMethod`, `MonthlyCharges`, `NPS` y `CSAT`.

El tratamiento de valores faltantes se realiza dentro del pipeline de preprocesamiento:

* Variables numéricas: imputación mediante mediana.
* Variables categóricas: imputación mediante una constante.

`CustomerID` se conserva para identificación, pero se excluye como variable predictora.

## 6. Limitaciones y uso previsto

Este dataset está destinado exclusivamente a ejercicios académicos de exploración, limpieza, visualización y modelamiento.

Sus principales limitaciones son:

* No contiene información real de clientes.
* No representa estadísticamente el comportamiento de una empresa o del mercado.
* Las relaciones entre variables dependen de las reglas utilizadas para generar los datos.
* No permite establecer relaciones causales.
* No contiene una fecha calendario que permita realizar análisis cronológicos reales.

Por estas razones, los resultados de los modelos no deben interpretarse como evidencia de capacidad predictiva sobre clientes reales ni utilizarse directamente para tomar decisiones comerciales.

## 7. Archivo fuente

`data/telecom_churn_10000.csv`

**Tamaño:** 10.000 registros y 35 variables.
