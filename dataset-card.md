# 📊 Dataset Card: Telecom Churn (10,000 Registros)

## Descripción General
Este dataset contiene información detallada de 10,000 clientes de una compañía de telecomunicaciones, estructurado para predecir la fuga de clientes (*Churn*) mediante técnicas de Aprendizaje Automático.

## Estructura y Variables
- **Variables Demográficas**: Género, Edad, Adulto Mayor (`SeniorCitizen`), Pareja, Dependientes.
- **Servicios Contratados**: Antigüedad (`TenureMonths`), Servicio telefónico, Internet, Tipo de contrato (`Contract`), Facturación electrónica, Método de pago.
- **Métricas de Satisfacción y Operativas**: Cargos mensuales y totales, Tickets de soporte, Quejas registradas, NPS (Net Promoter Score) y CSAT (Customer Satisfaction).
- **Variable Objetivo (Target)**: `Churn` (Binaria: 1 para clientes que cancelaron, 0 para clientes activos).

## Consideraciones de Privacidad y Preprocesamiento
- Se excluyó la columna `CustomerID` por tratarse de un identificador único sin valor predictivo.
- Se excluyó `ChurnReason` durante el entrenamiento para prevenir fugas de datos (*Data Leakage*).
- Los valores nulos se tratan mediante imputación por mediana en variables numéricas y constantes en categóricas dentro del pipeline de Scikit-Learn.