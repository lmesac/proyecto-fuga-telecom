# Ficha del dataset Telecom

10.000 registros de clientes sintéticos y 35 columnas. Churn es la etiqueta Yes/No. 4.169 abandonos y 5.831 permanencias. No hay fechas ni horizonte de predicción documentado.

## Procedencia y licencia
Según el [README del equipo, revisión d2f501b](https://github.com/lmesac/proyecto-fuga-telecom/blob/d2f501b/README.md), el archivo fue generado sintéticamente desde cero en Python. [IBM Telco Customer Churn en Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), con 7.043 registros y 21 variables, sirvió de referencia conceptual y estructural; no se incorporaron sus registros originales.

El equipo declara una semilla `np.random.seed(42)`, distribuciones aleatorias y reglas de simulación, una función logística con aleatoriedad para generar `Churn`, reglas condicionales para `ChurnReason` y faltantes controlados de aproximadamente 1,5% en cuatro columnas.

Las 14 variables adicionales declaradas son Age, TenureCohort, ServiceCount, AutoPay, DiscountRate, SupportTickets, Complaints, CustomerContacts, ServiceOutages, NPS, CSAT, EstimatedCLV, AvgTicketCost y ChurnReason.

Esta descripción documenta lo comunicado por el equipo; aún no verifica la generación. En la revisión consultada no aparece el script original. Se necesita ese script, sus dependencias y una comparación con el CSV para reproducir las reglas y verificar el archivo. La semilla por sí sola no basta. Autoría concreta y licencia del conjunto sintético siguen pendientes; la referencia a Kaggle no establece su licencia.

## Alcance
Datos completamente sintéticos, sin clientes reales ni representatividad estadística de una empresa. Los patrones del EDA y las métricas reflejan la simulación, no evidencia de comportamiento del mercado ni causalidad. La falta de fechas impide validar disponibilidad temporal de los predictores.

## Reproducción pendiente
El CSV todavía está versionado en Git. Para cumplir el requisito del curso, incorporar y comprobar el generador original antes de retirar los datos crudos del seguimiento; conservar una copia local. No sustituirlo por la descarga de Kaggle, pues sería otro conjunto de datos.

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
- **Churn**: Abandono simulado
- **ChurnReason**: Motivo posterior al abandono; fuga de información