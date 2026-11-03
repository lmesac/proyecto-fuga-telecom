import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Sistema de Predicción de Fuga (Churn)", layout="wide")

st.title("📉 Sistema de Predicción de Fuga de Clientes (Telecom)")
st.write("Ingrese las características del cliente para evaluar el riesgo real de cancelación del servicio usando el modelo entrenado.")

# Cargar el modelo entrenado y una plantilla de datos para asegurar que existan todas las columnas
@st.cache_resource
def load_model_and_data():
    model_path = Path("models/best_model.pkl")
    model = joblib.load(model_path) if model_path.exists() else None
    
    data_path = Path("data/telecom_churn_10000.csv")
    df_template = pd.read_csv(data_path).head(1) if data_path.exists() else None
    return model, df_template

model, df_template = load_model_and_data()

if model is None:
    st.warning("⚠️ No se encontró el modelo entrenado en `models/best_model.pkl`. Asegúrate de ejecutar primero el entrenamiento.")

# Formulario de entrada
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Datos Demográficos")
    age = st.number_input("Edad", min_value=18, max_value=100, value=35)
    gender = st.selectbox("Género", ["Female", "Male"])
    dependents = st.selectbox("Tiene Dependientes", ["Yes", "No"])

with col2:
    st.subheader("Servicios Contratados")
    tenure = st.number_input("Antigüedad (Meses)", min_value=0, max_value=180, value=12)
    internet = st.selectbox("Servicio Internet", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Tipo de Contrato", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Factura Electrónica", ["Yes", "No"])
    payment = st.selectbox("Método de Pago", [
        "Virtual", "Tienda", "Banco", "Credit card (automatic)"
    ])

with col3:
    st.subheader("Métricas y Satisfacción")
    monthly_charges = st.number_input("Cargo Mensual ($)", min_value=0.0, value=75.0)
    nps = st.slider("Puntaje NPS (0 a 10)", 0, 10, 7)
    csat = st.slider("Puntaje CSAT (1 a 5)", 1, 5, 4)
    support_tickets = st.number_input("Tickets de Soporte", min_value=0, value=1)
    complaints = st.number_input("Quejas Registradas", min_value=0, value=0)

# Botón para predecir con el modelo real
if st.button("Evaluar Riesgo de Fuga"):
    if model is None:
        st.error("Error: Imposible predecir sin un modelo cargado en `models/best_model.pkl`.")
    elif df_template is None:
        st.error("Error: No se encontró el archivo de datos base en `data/`.")
    else:
        # Copiamos la plantilla base para garantizar que todas las 35 columnas estén presentes
        input_data = df_template.copy()
        
        # Limpiamos las columnas que no van como features
        for col in ['Churn', 'CustomerID', 'ChurnReason']:
            if col in input_data.columns:
                input_data = input_data.drop(columns=[col])
                
        # Actualizamos con los valores ingresados en el formulario de Streamlit
        input_data['Age'] = age
        input_data['Gender'] = gender
        input_data['Dependents'] = dependents
        input_data['TenureMonths'] = tenure
        input_data['InternetService'] = internet
        input_data['Contract'] = contract
        input_data['PaperlessBilling'] = paperless
        input_data['PaymentMethod'] = payment
        input_data['MonthlyCharges'] = monthly_charges
        input_data['NPS'] = nps
        input_data['CSAT'] = csat
        input_data['SupportTickets'] = support_tickets
        input_data['Complaints'] = complaints
        
        try:
            # Inferencia real utilizando el Pipeline y el modelo entrenado
            prob_fuga = model.predict_proba(input_data)[0][1]
            
            st.divider()
            if prob_fuga >= 0.5:
                st.error(f"⚠️ **ALTO RIESGO DE FUGA DETECTADO**\n\nProbabilidad estimada: **{prob_fuga:.1%}**")
                st.info("Recomendación: Ofrecer plan de fidelización o descuento en renovación.")
            else:
                st.success(f"✅ **CLIENTE ESTABLE**\n\nProbabilidad estimada de fuga: **{prob_fuga:.1%}**")
                
        except Exception as e:
            st.error(f"Ocurrió un error al procesar la predicción: {e}")