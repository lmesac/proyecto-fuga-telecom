import pandas as pd
import streamlit as st

from telecom_churn.data import load_data
from telecom_churn.services.churn_service import evaluate_customers


# =========================================================
# CONFIGURACIÓN
# =========================================================

st.set_page_config(
    page_title="Centro de Riesgo de Churn",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Centro de Riesgo de Fuga de Clientes")

st.write(
    "Complete algunos datos clave del cliente y el sistema "
    "estimará su probabilidad de fuga."
)


# =========================================================
# DATOS DE REFERENCIA
# =========================================================

df_reference = load_data()


def category_options(column: str) -> list:
    return sorted(
        df_reference[column]
        .dropna()
        .unique()
        .tolist()
    )


def default_value(column: str):
    """
    Devuelve un valor de referencia para completar
    automáticamente variables no ingresadas.
    """

    if pd.api.types.is_numeric_dtype(
        df_reference[column]
    ):
        return df_reference[column].median()

    return df_reference[column].mode().iloc[0]


def build_base_customer() -> dict:
    """
    Construye un cliente base usando valores típicos
    del dataset.
    """

    excluded_columns = {
        "CustomerID",
        "Churn",
        "ChurnReason",
    }

    customer = {}

    for column in df_reference.columns:

        if column in excluded_columns:
            continue

        customer[column] = default_value(column)

    return customer


# =========================================================
# FORMULARIO DE EVALUACIÓN RÁPIDA
# =========================================================

st.subheader("Evaluación rápida")

st.caption(
    "Las variables no ingresadas se completan automáticamente "
    "con valores de referencia del dataset."
)

with st.form("quick_customer_form"):

    col1, col2 = st.columns(2)

    tenure_months = col1.number_input(
        "Meses de permanencia",
        min_value=0,
        value=12,
    )

    contract = col2.selectbox(
        "Tipo de contrato",
        category_options("Contract"),
    )

    col1, col2 = st.columns(2)

    monthly_charges = col1.number_input(
        "Cargo mensual",
        min_value=0.0,
        value=float(
            default_value("MonthlyCharges")
        ),
    )

    internet_service = col2.selectbox(
        "Servicio de internet",
        category_options("InternetService"),
    )

    col1, col2 = st.columns(2)

    payment_method = col1.selectbox(
        "Método de pago",
        category_options("PaymentMethod"),
    )

    auto_pay = col2.selectbox(
        "Pago automático",
        category_options("AutoPay"),
    )

    col1, col2 = st.columns(2)

    support_tickets = col1.number_input(
        "Tickets de soporte",
        min_value=0,
        value=int(
            default_value("SupportTickets")
        ),
    )

    complaints = col2.number_input(
        "Quejas",
        min_value=0,
        value=int(
            default_value("Complaints")
        ),
    )

    col1, col2 = st.columns(2)

    nps = col1.number_input(
        "NPS",
        value=float(
            default_value("NPS")
        ),
    )

    csat = col2.number_input(
        "CSAT",
        value=float(
            default_value("CSAT")
        ),
    )

    submitted = st.form_submit_button(
        "🔍 Evaluar riesgo",
        use_container_width=True,
    )


# =========================================================
# PREDICCIÓN
# =========================================================

if submitted:

    customer = build_base_customer()

    # Reemplazar únicamente las variables
    # ingresadas por el usuario.

    customer["TenureMonths"] = tenure_months
    customer["Contract"] = contract
    customer["MonthlyCharges"] = monthly_charges
    customer["InternetService"] = internet_service
    customer["PaymentMethod"] = payment_method
    customer["AutoPay"] = auto_pay
    customer["SupportTickets"] = support_tickets
    customer["Complaints"] = complaints
    customer["NPS"] = nps
    customer["CSAT"] = csat

    customer_df = pd.DataFrame([customer])

    try:

        result = evaluate_customers(customer_df)

        probability = float(
            result.loc[
                0,
                "Probabilidad_Fuga",
            ]
        )

        risk = result.loc[
            0,
            "Nivel_Riesgo",
        ]

        recommendation = result.loc[
            0,
            "Recomendacion",
        ]

        st.divider()

        st.header("Resultado de la evaluación")

        col1, col2 = st.columns(2)

        col1.metric(
            "Probabilidad de fuga",
            f"{probability:.1%}",
        )

        col2.metric(
            "Nivel de riesgo",
            risk,
        )

        st.progress(probability)

        if risk == "Alto":

            st.error(
                f"🔴 Riesgo ALTO\n\n{recommendation}"
            )

        elif risk == "Medio":

            st.warning(
                f"🟡 Riesgo MEDIO\n\n{recommendation}"
            )

        else:

            st.success(
                f"🟢 Riesgo BAJO\n\n{recommendation}"
            )

        with st.expander(
            "Ver datos enviados al modelo"
        ):

            st.dataframe(
                customer_df,
                use_container_width=True,
            )

    except Exception as error:

        st.error(
            "No fue posible generar la predicción."
        )

        st.exception(error)