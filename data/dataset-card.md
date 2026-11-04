# Dataset Card - Telecom Churn

## Descripción

Este dataset se utiliza para desarrollar un modelo de clasificación orientado a predecir la fuga de clientes de una compañía de telecomunicaciones.

## Dimensiones

- Registros: 10.000
- Columnas: 35
- Variable objetivo: `Churn`

## Distribución de la variable objetivo

La variable `Churn` indica si un cliente abandonó o no el servicio.

- `No`: 5.831 clientes (58,31 %)
- `Yes`: 4.169 clientes (41,69 %)

La distribución de clases es relativamente balanceada, por lo que inicialmente no se considera necesario aplicar técnicas de sobremuestreo o submuestreo.

## Calidad de los datos

### Duplicados

No se identificaron registros duplicados.

- Duplicados encontrados: 0

### Valores faltantes

Se identificaron valores faltantes en las siguientes variables:

| Variable | Valores faltantes |
|---|---:|
| PaymentMethod | 150 |
| MonthlyCharges | 150 |
| NPS | 150 |
| CSAT | 150 |

El resto de las variables no presenta valores faltantes.

## Variables excluidas del modelo

### CustomerID

`CustomerID` funciona como identificador único del cliente y no representa una característica de comportamiento útil para la predicción.

Por esta razón, debe excluirse de las variables predictoras.

### ChurnReason

`ChurnReason` describe el motivo asociado al abandono del cliente.

Esta variable presenta riesgo de **data leakage**, ya que contiene información directamente relacionada con el resultado que se intenta predecir.

Por esta razón, debe excluirse del entrenamiento.

## Estrategia preliminar de tratamiento de faltantes

La imputación deberá realizarse dentro del pipeline de preprocesamiento y ajustarse únicamente con los datos de entrenamiento.

Estrategia inicial:

- `MonthlyCharges`: mediana
- `NPS`: mediana
- `CSAT`: mediana
- `PaymentMethod`: moda

Esta estrategia deberá validarse durante el desarrollo del pipeline.

## Fuente y licencia

La fuente externa original del dataset y su licencia todavía deben ser verificadas por el equipo.

No se debe afirmar que el dataset pertenece a IBM, Kaggle u otra fuente específica hasta contar con un enlace o referencia verificable.

## Consideraciones

- Los datos contienen variables numéricas y categóricas.
- El problema corresponde a clasificación binaria.
- Se debe evitar fuga de información durante el preprocesamiento.
- El split entre entrenamiento, validación y prueba debe realizarse antes de ajustar imputadores, codificadores o escaladores.
- Las transformaciones deben formar parte de un pipeline reproducible.
