
import streamlit as st
import pandas as pd

from telecom_churn.config import (
    MODEL_PATH,
    CLASSIFICATION_THRESHOLD
)
from telecom_churn.predict import predict_dataframe


st.set_page_config(
    page_title="Predicción de fuga de clientes",
    page_icon="📊",
    layout="wide"
)

st.title("Sistema de predicción de fuga de clientes")
st.write(
    "Herramienta de análisis del riesgo de fuga de clientes "
    "mediante un modelo de Machine Learning."
)

st.info(
    "Cargue un archivo CSV que contenga las 32 variables predictoras "
    "utilizadas durante el entrenamiento. Puede incluir columnas "
    "adicionales como CustomerID o Churn."
)

if not MODEL_PATH.exists():
    st.error(
        "No se encuentra el modelo entrenado. "
        "Ejecute primero: uv run python -m telecom_churn.train"
    )
    st.stop()

threshold = st.sidebar.slider(
    "Umbral de clasificación de fuga",
    min_value=0.10,
    max_value=0.90,
    value=float(CLASSIFICATION_THRESHOLD),
    step=0.05
)

st.sidebar.write(
    f"Umbral seleccionado: {threshold:.0%}"
)

uploaded_file = st.file_uploader(
    "Cargar archivo de clientes (CSV)",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("Datos cargados")
        st.write(f"Registros: {len(df)}")
        st.dataframe(df.head(10), use_container_width=True)

        if st.button("Evaluar riesgo de fuga"):

            results = predict_dataframe(
                df,
                threshold=threshold
            )

            st.subheader("Resultados de la predicción")

            total = len(results)
            high_risk = int(
                (results["Prediccion_Fuga"] == 1).sum()
            )
            low_risk = total - high_risk

            col1, col2, col3 = st.columns(3)

            col1.metric("Clientes evaluados", total)
            col2.metric("Posibles fugas", high_risk)
            col3.metric("Sin alerta de fuga", low_risk)

            st.dataframe(
                results,
                use_container_width=True
            )

            st.subheader("Distribución de resultados")

            st.bar_chart(
                results["Resultado"].value_counts()
            )

            csv = results.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Descargar resultados CSV",
                data=csv,
                file_name="predicciones_fuga.csv",
                mime="text/csv"
            )

    except Exception as error:
        st.error(f"No fue posible procesar el archivo: {error}")