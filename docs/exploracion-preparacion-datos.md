# Exploración y preparación de datos

## 1. Fuente de información

El proyecto utiliza el archivo `data/telecom_churn_10000.csv`, un dataset sintético de telecomunicaciones con 10.000 registros y 35 variables.

El generador se encuentra en `data/generar_dataset.py` y utiliza Python, NumPy y Pandas. La semilla aleatoria establecida es 42, con el propósito de permitir la reproducción de los datos bajo las mismas condiciones de ejecución.

## 2. Estructura del dataset

El conjunto de datos contiene variables relacionadas con:

- Identificación y características demográficas.
- Antigüedad y permanencia del cliente.
- Servicios contratados.
- Contratos y facturación.
- Cargos y descuentos.
- Contactos, reclamaciones e interrupciones del servicio.
- Indicadores de experiencia: NPS y CSAT.
- Valor estimado del cliente y costo promedio de atención.
- Fuga del cliente y motivo asociado.

La variable objetivo es `Churn`.

## 3. Carga y validación

El módulo `src/telecom_churn/data.py` implementa la función `load_data()`, que:

1. Verifica que exista el archivo definido en la configuración.
2. Carga el CSV mediante Pandas.
3. Comprueba que el dataset no esté vacío.
4. Verifica que exista la variable objetivo `Churn`.
5. Retorna el DataFrame cargado.

Estas validaciones comprueban condiciones básicas de disponibilidad y estructura. No constituyen una validación estadística integral del dataset.

## 4. Preparación de variables

La función `prepare_features()` del módulo `src/telecom_churn/features.py` separa las variables predictoras de la variable objetivo.

La preparación contempla:

- Validar que el DataFrame no esté vacío.
- Verificar la existencia de la variable objetivo.
- Comprobar que la variable objetivo no tenga valores nulos.
- Normalizar los valores de `Churn` y convertirlos a clasificación binaria: 1 para fuga y 0 para no fuga.
- Rechazar categorías de la variable objetivo que no estén contempladas.
- Excluir las columnas `CustomerID` y `ChurnReason` de las variables predictoras.
- Retornar las variables predictoras (X) y la variable objetivo (y).

La exclusión de `ChurnReason` evita utilizar como predictor una variable asociada al resultado de fuga.

## 5. Tratamiento de variables

La función `identify_feature_types()` clasifica automáticamente las columnas en numéricas y categóricas según sus tipos de datos.

La función `build_preprocessor()` construye un preprocesador mediante `ColumnTransformer` y pipelines de Scikit-learn:

| Tipo de variable | Tratamiento |
|---|---|
| Numérica | Imputación de valores faltantes mediante la mediana y escalado con `StandardScaler`. |
| Categórica | Imputación con el valor constante `missing` y codificación mediante `OneHotEncoder`. |

El codificador utiliza `handle_unknown="ignore"` para manejar categorías no observadas durante el ajuste.

El diseño establece que la imputación, el escalado y la codificación se ajusten dentro de `pipeline.fit()`, utilizando los datos de entrenamiento y evitando ajustar las transformaciones con información del conjunto de prueba.

## 6. Consideraciones metodológicas

El dataset es sintético. Sus distribuciones y relaciones entre variables dependen de las reglas definidas en el generador.

Por tanto, los resultados de la exploración y del modelamiento deben interpretarse dentro del alcance académico del ejercicio. No representan evidencia sobre el comportamiento de clientes reales ni permiten establecer relaciones causales.

