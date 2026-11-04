# Evaluación explicada

Análisis complementario sobre desarrollo. Test final excluido. No modifica el modelo ni el umbral operativo.

## Estabilidad
Cinco pliegues estratificados sobre entrenamiento. Cada pliegue reajusta imputación y codificación.

| Modelo | AUC media | Desviación |
|---|---|---|
| Logistic Regression | 0.7419 | 0.0075 |
| Random Forest | 0.7158 | 0.0079 |
| HistGradientBoosting | 0.7219 | 0.0099 |

Conclusión: regresión logística mantiene la mayor AUC media. La dispersión mide variación entre particiones, no un intervalo de confianza ni garantía de rendimiento futuro.

## Umbrales en validación
Bajar el umbral señala más clientes; hay que considerar falsas alarmas y capacidad de atención.

| Umbral | Señalados | Precisión | Recall | F1 |
|---|---|---|---|---|
| 0.2 | 1657 | 0.482 | 0.958 | 0.642 |
| 0.3 | 1334 | 0.542 | 0.867 | 0.667 |
| 0.4 | 1060 | 0.597 | 0.759 | 0.668 |
| 0.5 | 745 | 0.652 | 0.583 | 0.616 |
| 0.6 | 487 | 0.725 | 0.423 | 0.534 |
| 0.7 | 266 | 0.782 | 0.249 | 0.378 |

Decisión: mantener 0,5 hasta acordar criterio de capacidad/coste. Esta tabla es exploratoria; no escoger por test.

## Capacidad de contacto
Al priorizar 400 de los 2.000 clientes de validación (20%), el 74,75% del grupo seleccionado corresponde a abandonos observados y se captura el 35,85% de todos los abandonos. Esto mide priorización, no clientes salvados ni ahorro.

## Importancia por permutación
Se altera cada columna en validación y se mide caída de AUC, con tres repeticiones. Variables correlacionadas pueden compartir importancia.
- TenureMonths: caída media 0.0558, desviación 0.0051.
- Contract: caída media 0.0515, desviación 0.0030.
- TenureCohort: caída media 0.0291, desviación 0.0035.
- MonthlyCharges: caída media 0.0201, desviación 0.0010.
- AutoPay: caída media 0.0079, desviación 0.0011.

Consultar importance.html para el gráfico interactivo. No interpretar estas relaciones como causas.
