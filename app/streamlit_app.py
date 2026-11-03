"""Demo académica: puntuar clientes del dataset o un CSV compatible."""
import pandas as pd
import streamlit as st
from telecom_churn.data import load_data
from telecom_churn.predict import predict_churn
st.title("Riesgo de abandono de clientes")
st.caption("Demostración académica. Sin horizonte temporal validado ni efecto causal de retención.")
upload = st.file_uploader("CSV de clientes", type="csv")
# La selección de un cliente permite probar la demo sin preparar archivos externos.
df = pd.read_csv(upload) if upload else load_data()
row = st.number_input("Fila del cliente (desde 0)", min_value=0, max_value=max(0, len(df)-1), value=0)
if not df.empty:
    st.dataframe(df.iloc[[row]].drop(columns=["Churn", "ChurnReason"], errors="ignore"))
    if st.button("Estimar riesgo"):
        try:
            result = predict_churn(df.iloc[[row]])
            st.metric("Probabilidad estimada", f"{result.iloc[0,0]:.1%}")
            st.write("Clasificación con umbral 50%:", "Abandono" if result.iloc[0,1] else "Permanencia")
        except (FileNotFoundError, ValueError) as error:
            st.error(str(error))
