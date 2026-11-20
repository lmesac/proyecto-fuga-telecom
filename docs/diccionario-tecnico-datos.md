# Documentación técnica y diccionario de datos

## 1. Identificación del dataset

| Característica | Descripción |
|---|---|
| Archivo fuente | `data/telecom_churn_10000.csv` |
| Registros | 10.000 |
| Variables | 35 |
| Variable objetivo | `Churn` |
| Identificador | `CustomerID` |
| Naturaleza | Dataset sintético |
| Generador | `data/generar_dataset.py` |
| Semilla aleatoria | 42 |

## 2. Diccionario de variables

El siguiente diccionario describe las variables a partir de sus nombres y de las reglas visibles en el generador.

| Variable | Descripción |
|---|---|
| `CustomerID` | Identificador sintético del cliente. |
| `Age` | Edad del cliente. |
| `SeniorCitizen` | Indicador derivado: 1 cuando la edad es igual o superior a 65 años; 0 en otro caso. |
| `Gender` | Género codificado en el generador como M o F. |
| `Partner` | Indicador de pareja, con valores Yes o No. |
| `Dependents` | Indicador de dependientes, con valores Yes o No. |
| `TenureMonths` | Antigüedad del cliente en meses. |
| `TenureCohort` | Grupo de permanencia derivado de `TenureMonths`. |
| `PhoneService` | Disponibilidad de servicio telefónico. |
| `MultipleLines` | Indicador de líneas múltiples. |
| `InternetService` | Tipo de servicio de internet. |
| `OnlineSecurity` | Servicio de seguridad en línea. |
| `OnlineBackup` | Servicio de respaldo en línea. |
| `DeviceProtection` | Servicio de protección de dispositivos. |
| `TechSupport` | Servicio de soporte técnico. |
| `StreamingTV` | Servicio de televisión por streaming. |
| `StreamingMovies` | Servicio de películas por streaming. |
| `ServiceCount` | Conteo derivado de servicios activos. |
| `Contract` | Tipo de contrato. |
| `PaperlessBilling` | Indicador de facturación electrónica o sin papel. |
| `AutoPay` | Indicador de pago automático. |
| `PaymentMethod` | Método de pago. |
| `MonthlyCharges` | Cargo mensual. |
| `DiscountRate` | Tasa de descuento aplicada en la generación. |
| `TotalCharges` | Cargo total calculado a partir del cargo mensual, permanencia y descuento. |
| `SupportTickets` | Cantidad sintética de tickets de soporte. |
| `Complaints` | Cantidad sintética de reclamaciones. |
| `CustomerContacts` | Cantidad sintética de contactos del cliente. |
| `ServiceOutages` | Cantidad sintética de interrupciones del servicio. |
| `NPS` | Indicador sintético de experiencia tipo Net Promoter Score, generado en una escala de 0 a 10. |
| `CSAT` | Indicador sintético de satisfacción, generado en una escala de 1 a 5. |
| `EstimatedCLV` | Valor estimado sintético del cliente, calculado con cargo mensual, permanencia y un factor aleatorio. |
| `AvgTicketCost` | Costo promedio sintético de atención por ticket. |
| `Churn` | Variable objetivo: indica si se genera fuga (Yes) o no fuga (No). |
| `ChurnReason` | Motivo sintético asociado al resultado de fuga, generado mediante reglas condicionales. |

## 3. Variables utilizadas en el modelamiento

La configuración central del proyecto (`src/telecom_churn/config.py`) define:

- Variable objetivo: `Churn`.
- Identificador: `CustomerID`.
- Columnas excluidas: `CustomerID` y `ChurnReason`.
- Semilla de experimentación: 42.
- Tamaño de prueba: 0.20.
- Umbral inicial de clasificación: 0.35.

La función `prepare_features()` separa la variable objetivo y excluye las columnas indicadas. Las variables predictoras restantes se clasifican automáticamente como numéricas o categóricas.

## 4. Valores faltantes

El generador selecciona 150 registros y asigna valores faltantes coincidentes en cuatro columnas:

- `PaymentMethod`
- `MonthlyCharges`
- `NPS`
- `CSAT`

Esto representa 150 registros afectados y 600 celdas con valores faltantes en esas cuatro columnas, según las instrucciones del generador.

En el preprocesamiento, las variables numéricas se imputan con la mediana y las categóricas con el valor constante `missing`.

## 5. Limitaciones y uso previsto

Este dataset está destinado exclusivamente a ejercicios académicos de exploración, limpieza, visualización y modelamiento.

Sus principales limitaciones son:

- No contiene información real de clientes.
- No representa estadísticamente el comportamiento de una empresa o del mercado.
- Las relaciones entre variables dependen de las reglas utilizadas para generar los datos.
- No permite establecer relaciones causales.
- No contiene una fecha calendario que permita realizar análisis cronológicos reales.

Por estas razones, los resultados de los modelos no deben interpretarse como evidencia de capacidad predictiva sobre clientes reales ni utilizarse directamente para tomar decisiones comerciales.

## 6. Archivo fuente

`data/telecom_churn_10000.csv`

**Tamaño:** 10.000 registros y 35 variables.

