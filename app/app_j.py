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
    "Herramienta para estimar el riesgo de fuga de clientes "
    "utilizando el modelo de Machine Learning entrenado en el proyecto."
)


# =========================================================
# DATOS DE REFERENCIA
# =========================================================

df_reference = load_data()


def category_options(column: str) -> list:
    """
    Obtiene las categorías válidas de una variable
    directamente desde el dataset.
    """
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
    Construye un cliente base utilizando valores típicos
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


def complete_batch_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Completa las variables faltantes de un archivo masivo
    utilizando valores de referencia del dataset.
    """

    completed_df = df.copy()

    base_customer = build_base_customer()

    for column, value in base_customer.items():

        if column not in completed_df.columns:
            completed_df[column] = value

    return completed_df


# =========================================================
# EVALUACIÓN INDIVIDUAL
# =========================================================

st.header("👤 Evaluación individual")

st.caption(
    "Ingrese algunos datos clave del cliente. "
    "Las demás variables requeridas por el modelo se completan "
    "automáticamente con valores de referencia del dataset."
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


if submitted:

    customer = build_base_customer()

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

        st.subheader("Resultado de la evaluación")

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


# =========================================================
# EVALUACIÓN MASIVA
# =========================================================

st.divider()

st.header("📂 Evaluación masiva de clientes")

st.write(
    "Cargue un archivo CSV para evaluar múltiples clientes "
    "y priorizar aquellos con mayor riesgo de fuga."
)

st.caption(
    "Si el archivo no contiene todas las variables requeridas, "
    "el sistema completa las variables faltantes con valores "
    "de referencia del dataset."
)

uploaded_file = st.file_uploader(
    "Seleccionar archivo CSV",
    type=["csv"],
    key="batch_file",
)


if uploaded_file is not None:

    try:

        batch_df = pd.read_csv(uploaded_file)

        if batch_df.empty:
            st.error(
                "El archivo cargado no contiene registros."
            )

        else:

            st.success(
                f"Archivo cargado correctamente: "
                f"{len(batch_df)} clientes."
            )

            with st.expander(
                "Vista previa del archivo"
            ):

                st.dataframe(
                    batch_df.head(10),
                    use_container_width=True,
                )

            if st.button(
                "📊 Analizar clientes",
                use_container_width=True,
            ):

                # Completar columnas faltantes.
                completed_batch = complete_batch_data(
                    batch_df
                )

                # Generar predicciones.
                batch_results = evaluate_customers(
                    completed_batch
                )

                # Mantener CustomerID cuando exista.
                if (
                    "CustomerID" in batch_df.columns
                    and "CustomerID"
                    not in batch_results.columns
                ):

                    batch_results.insert(
                        0,
                        "CustomerID",
                        batch_df[
                            "CustomerID"
                        ].values,
                    )

                # Ordenar clientes desde mayor riesgo.
                batch_results = (
                    batch_results
                    .sort_values(
                        by="Probabilidad_Fuga",
                        ascending=False,
                    )
                    .reset_index(drop=True)
                )

                # =========================================
                # KPIs
                # =========================================

                total_clients = len(
                    batch_results
                )

                high_risk = int(
                    (
                        batch_results[
                            "Nivel_Riesgo"
                        ]
                        == "Alto"
                    ).sum()
                )

                medium_risk = int(
                    (
                        batch_results[
                            "Nivel_Riesgo"
                        ]
                        == "Medio"
                    ).sum()
                )

                low_risk = int(
                    (
                        batch_results[
                            "Nivel_Riesgo"
                        ]
                        == "Bajo"
                    ).sum()
                )

                st.subheader(
                    "Resumen de riesgo"
                )

                col1, col2, col3, col4 = (
                    st.columns(4)
                )

                col1.metric(
                    "Clientes evaluados",
                    total_clients,
                )

                col2.metric(
                    "🔴 Riesgo alto",
                    high_risk,
                )

                col3.metric(
                    "🟡 Riesgo medio",
                    medium_risk,
                )

                col4.metric(
                    "🟢 Riesgo bajo",
                    low_risk,
                )

                # =========================================
                # RANKING
                # =========================================

                st.subheader(
                    "Clientes priorizados por riesgo"
                )

                ranking_columns = [
                    column
                    for column in [
                        "CustomerID",
                        "Probabilidad_Fuga",
                        "Nivel_Riesgo",
                        "Recomendacion",
                    ]
                    if column
                    in batch_results.columns
                ]

                st.dataframe(
                    batch_results[
                        ranking_columns
                    ],
                    use_container_width=True,
                )

                # =========================================
                # TOP 10
                # =========================================

                st.subheader(
                    "🚨 Top 10 clientes con mayor riesgo"
                )

                top_10 = batch_results.head(10)

                st.dataframe(
                    top_10[
                        ranking_columns
                    ],
                    use_container_width=True,
                )

                # =========================================
                # DISTRIBUCIÓN
                # =========================================

                st.subheader(
                    "Distribución de niveles de riesgo"
                )

                risk_distribution = (
                    batch_results[
                        "Nivel_Riesgo"
                    ]
                    .value_counts()
                    .reindex(
                        [
                            "Alto",
                            "Medio",
                            "Bajo",
                        ],
                        fill_value=0,
                    )
                )

                st.bar_chart(
                    risk_distribution
                )

                # =========================================
                # DESCARGA
                # =========================================

                csv_results = (
                    batch_results
                    .to_csv(index=False)
                    .encode("utf-8")
                )

                st.download_button(
                    label=(
                        "⬇️ Descargar resultados"
                    ),
                    data=csv_results,
                    file_name=(
                        "clientes_riesgo_churn.csv"
                    ),
                    mime="text/csv",
                    use_container_width=True,
                )

    except Exception as error:

        st.error(
            "No fue posible procesar el archivo."
        )

        st.exception(error)