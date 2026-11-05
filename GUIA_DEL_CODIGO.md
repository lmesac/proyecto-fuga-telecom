# Cómo entender el proyecto

1. data.py carga el CSV y valida identificadores y etiquetas. No rellena faltantes.
2. features.py separa X (características) e y (respuesta), excluye fuga directa y construye ColumnTransformer sin ajustarlo. Numéricas: mediana y escalado. Categóricas: moda y OneHotEncoder; categorías nuevas no rompen la predicción.
3. model.py declara las tres alternativas. Declarar un modelo no significa entrenarlo.
4. train.py separa 60/20/20, ajusta cada pipeline con entrenamiento, compara en validación y selecciona por ROC-AUC. Reentrena el ganador con 80% y evalúa en test. Guardar el pipeline evita procesar distinto en la app.
5. predict.py carga el artefacto propio, valida columnas y devuelve probabilidad y clase. No vuelve a entrenar.
6. app/streamlit_app.py presenta datos y llama a predict.py. La probabilidad no está calibrada para uso operativo.
7. tests comprueba datos, categorías desconocidas, faltantes, entrenamiento, guardado, inferencia y arranque de interfaz. Una prueba de arranque no sustituye revisar visualmente la demo.

## Cómo leer las métricas
Exactitud: porcentaje total de aciertos. Precisión: de los señalados, cuántos abandonan. Recall: de los abandonos, cuántos detectamos. F1 combina precisión y recall. ROC-AUC mide ordenación entre clases a través de umbrales. Umbral es el punto a partir del que se señala riesgo; 50% es una decisión inicial, no una constante universal.

## Qué falta verificar
Generador original, autoría y licencia del dataset sintético; generación reproducible; validación visual de la app; revisión del equipo y PR. No usar test repetidamente para escoger mejoras.

## Evaluación complementaria
evaluate.py ejecuta cinco pliegues sobre las 6.000 filas de entrenamiento, compara umbrales sobre validación y simula revisar el 20% de mayor riesgo. También permuta variables para medir su aporte. No modifica el pipeline guardado ni el umbral de la app. La desviación entre folds no es un intervalo de confianza.
