# Ficha del modelo de abandono

## Uso previsto
Demostración académica de priorización de revisión de clientes. No utilizar para decisiones operativas sin validar origen, tiempos de medición y generalización.

## Modelo y preparación
Regresión logística con imputación por mediana/moda, escalado numérico y OneHotEncoder. 32 predictores originales; se excluyen CustomerID, Churn y ChurnReason. Pipeline completo en models/churn.joblib.

## Evaluación
Semilla 42. Separación aleatoria estratificada 60/20/20. Selección por ROC-AUC en validación. Reajuste del ganador con 80%, prueba sobre 2.000 clientes. Umbral 0,5.

Matriz de prueba: 933 permanencias correctas, 233 falsas alarmas, 368 abandonos omitidos y 466 abandonos detectados. Exactitud 69,95%; precisión 66,67%; recall 55,88%; ROC-AUC 0,7509. El recall indica que todavía se omite aproximadamente el 44% de los abandonos.

## Limitaciones
Sin horizonte temporal, calibración operativa ni evaluación causal de retención. Datos generados sintéticamente según el README del equipo; script original y licencia pendientes de verificar. El rendimiento mide aprendizaje de la simulación, no generalización a clientes reales. La importancia de una variable no demuestra causa. No se ha demostrado ahorro comercial. Las métricas aplican a este archivo y esta partición.

## Reproducción
Ejecutar `uv run python -m telecom_churn.train` y `uv run pytest -q`. Evaluación complementaria: `uv run python -m telecom_churn.evaluate`. No cambiar parámetros mirando repetidamente test.
