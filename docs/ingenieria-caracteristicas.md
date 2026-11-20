# Ingeniería de características

## 1. Objetivo

La ingeniería de características del proyecto busca construir y preparar variables que permitan desarrollar un ejercicio de clasificación binaria de fuga de clientes (`Churn`).

El proceso comprende dos componentes: la generación de variables derivadas en el dataset sintético y la transformación de variables para su utilización en los modelos.

## 2. Variables derivadas en la generación del dataset

El archivo `data/generar_dataset.py` construye las siguientes variables a partir de reglas programadas:

| Variable | Regla de generación |
|---|---|
| `SeniorCitizen` | Toma el valor 1 cuando la edad es igual o superior a 65 años; en los demás casos toma 0. |
| `TenureCohort` | Agrupa `TenureMonths` en intervalos de permanencia: 1-12, 13-24, 25-36, 37-48, 49-60 y 61-72 meses. |
| `ServiceCount` | Cuenta los servicios activos mediante condiciones sobre telefonía, líneas múltiples, internet y servicios adicionales. |
| `TotalCharges` | Se calcula a partir de `MonthlyCharges`, `TenureMonths` y `DiscountRate`, aplicando la fórmula `MonthlyCharges × TenureMonths × (1 - DiscountRate × 0.5)`. |
| `EstimatedCLV` | Se calcula multiplicando el cargo mensual por los meses de permanencia y un factor aleatorio entre 0.8 y 1.2. |

Estas variables son construcciones sintéticas y no necesariamente equivalen a indicadores calculados con información operacional real.

## 3. Construcción de la variable objetivo

La variable `Churn` se genera mediante una probabilidad calculada con una función logística:

`churn_prob = 1 / (1 + exp(-z))`

Donde:

`z = -2.5 + 0.03 × MonthlyCharges - 0.04 × TenureMonths + 0.3 × SupportTickets + 0.4 × Complaints`

Posteriormente, se utiliza una distribución binomial para asignar los valores `Yes` o `No`.

Por tanto, la relación entre estas variables y la fuga está incorporada en las reglas de generación del dataset. No constituye una relación causal demostrada ni una estimación obtenida de clientes reales.

## 4. Construcción de la variable ChurnReason

El generador construye `ChurnReason` mediante condiciones sobre el resultado de fuga y variables como tickets de soporte, interrupciones, cargos mensuales y reclamaciones.

Esta variable se excluye explícitamente de las variables predictoras en `config.py`, junto con `CustomerID`, para evitar utilizar información que no corresponde al conjunto de predictores del modelo.

## 5. Preparación para el modelamiento

El módulo `src/telecom_churn/features.py` implementa tres funciones:

- `prepare_features()`: separa X e y, codifica la variable objetivo y excluye las columnas configuradas.
- `identify_feature_types()`: identifica las variables numéricas y categóricas.
- `build_preprocessor()`: construye los pipelines de imputación, escalado y codificación.

El tratamiento definido es:

**Variables numéricas**
- Imputación con la mediana.
- Estandarización mediante `StandardScaler`.

**Variables categóricas**
- Imputación con el valor constante `missing`.
- Codificación One-Hot mediante `OneHotEncoder`.
- Manejo de categorías desconocidas con `handle_unknown="ignore"`.

Las transformaciones se integran mediante `ColumnTransformer` y pipelines de Scikit-learn.

## 6. Prevención de fuga de información

La configuración excluye:

- `CustomerID`: identificador del registro, no una característica predictiva.
- `ChurnReason`: variable relacionada con el resultado de fuga.

La variable objetivo `Churn` se separa de X y se transforma en valores binarios: 1 para fuga y 0 para no fuga.

Además, el preprocesador está diseñado para ajustarse dentro del pipeline de entrenamiento, de modo que las estadísticas de imputación y escalado se aprendan exclusivamente con los datos utilizados en el ajuste.

## 7. Alcance

Las variables derivadas y las relaciones utilizadas para generar `Churn` responden a reglas sintéticas. En consecuencia, su utilidad es académica y metodológica. No deben interpretarse como evidencia de comportamiento real ni utilizarse directamente para decisiones comerciales.

