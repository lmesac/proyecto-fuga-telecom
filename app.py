import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Sistema de Predicción de Fuga (Churn)", layout="wide")

st.title("📉 Sistema de Predicción de Fuga de Clientes (Telecom)")
st.write("Ingrese las características del cliente para evaluar el riesgo de cancelación del servicio.")

# Formulario de entrada
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Datos Demográficos")
    age = st.number_input("Edad", min_value=18, max_value=100, value=35)
    gender = st.selectbox("Género", ["Female", "Male"])
    #senior = st.selectbox("Adulto Mayor", [0, 1])
    #partner = st.selectbox("Tiene Pareja", ["Yes", "No"])
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

# Botón para predecir
if st.button("Evaluar Riesgo de Fuga"):
    # Simulación de inferencia basada en lógica del modelo entrenado
    # En un entorno real se invoca el modelo guardado .pkl
    score_riesgo = (monthly_charges / 150) * 0.3 + (complaints * 0.2) + (1 if contract == "Month-to-month" else 0) * 0.3
    prob_fuga = min(max(score_riesgo, 0.05), 0.95)
    
    st.divider()
    if prob_fuga >= 0.5:
        st.error(f"⚠️ **ALTO RIESGOS DE FUGA DETECTADO**\n\nProbabilidad estimada: **{prob_fuga:.1%}**")
        st.info("Recomendación: Ofrecer plan de fidelización o descuento en renovación.")
    else:
        st.success(f"✅ **CLIENTE ESTABLE**\n\nProbabilidad estimada de fuga: **{prob_fuga:.1%}**")