# Ficha del dataset Telecom

10.000 clientes y 35 columnas. Churn es la etiqueta Yes/No. 4.169 abandonos y 5.831 permanencias. No hay fechas ni horizonte de predicción documentado.

## Procedencia y licencia
El archivo se recibió del repositorio del equipo. La fuente externa original, autor, licencia y posible generación sintética NO están verificados. No atribuir a IBM o Kaggle sin enlace verificable. Es un bloqueo documental pendiente de respuesta del equipo.

## Calidad y tratamiento
588 filas tienen algún faltante. PaymentMethod, MonthlyCharges, NPS y CSAT tienen 150 faltantes cada una. No se eliminan filas. Se imputa mediana numérica y moda categórica solo con entrenamiento. CustomerID y ChurnReason se excluyen de los predictores. Quedan 32 variables predictoras antes de codificación.

## Diccionario provisional
Los significados se infieren de los nombres y deben confirmarse con el proveedor.

- **CustomerID**: Identificador
- **Age**: Edad
- **SeniorCitizen**: Indicador senior; umbral pendiente
- **Gender**: Género
- **Partner**: Pareja
- **Dependents**: Dependientes
- **TenureMonths**: Antigüedad en meses
- **TenureCohort**: Grupo de antigüedad
- **PhoneService**: Servicio telefónico
- **MultipleLines**: Líneas múltiples
- **InternetService**: Internet
- **OnlineSecurity**: Seguridad
- **OnlineBackup**: Backup
- **DeviceProtection**: Protección de dispositivo
- **TechSupport**: Soporte técnico
- **StreamingTV**: TV
- **StreamingMovies**: Películas
- **ServiceCount**: Número de servicios
- **Contract**: Contrato
- **PaperlessBilling**: Factura electrónica
- **AutoPay**: Pago automático
- **PaymentMethod**: Método de pago
- **MonthlyCharges**: Cargo mensual; moneda pendiente
- **DiscountRate**: Descuento
- **TotalCharges**: Cargo acumulado; moneda pendiente
- **SupportTickets**: Tickets; período pendiente
- **Complaints**: Quejas; período pendiente
- **CustomerContacts**: Contactos; período pendiente
- **ServiceOutages**: Interrupciones; período pendiente
- **NPS**: Puntuación 0–10; definición pendiente
- **CSAT**: Satisfacción 1–5
- **EstimatedCLV**: Valor estimado; fórmula pendiente
- **AvgTicketCost**: Costo de ticket; fórmula y moneda pendientes
- **Churn**: Abandono observado
- **ChurnReason**: Motivo posterior al abandono; fuga de información