import numpy as np
import pandas as pd

# Fijar semilla para reproducibilidad exacta
np.random.seed(42)
n_samples = 10000

# 1. CustomerID
customer_ids = [f"TELCO-{i:05d}" for i in range(1, n_samples + 1)]

# 2. Datos Demográficos (Alineados al original)
age = np.random.randint(18, 81, size=n_samples)
senior_citizen = np.where(age >= 65, 1, 0)
gender = np.random.choice(["M", "F"], size=n_samples, p=[0.51, 0.49])
partner = np.random.choice(["Yes", "No"], size=n_samples, p=[0.42, 0.58])
dependents = np.random.choice(["Yes", "No"], size=n_samples, p=[0.28, 0.72])

# 3. Permanencia (Tenure)
tenure_months = np.random.choice(
    np.arange(1, 73),
    size=n_samples,
    p=np.linspace(0.03, 0.01, 72) / np.sum(np.linspace(0.03, 0.01, 72)),
)


def get_cohort(months):
  if months <= 12:
    return "1-12 months"
  elif months <= 24:
    return "13-24 months"
  elif months <= 36:
    return "25-36 months"
  elif months <= 48:
    return "37-48 months"
  elif months <= 60:
    return "49-60 months"
  elif months <= 72:
    return "61-72 months"
  else:
    return ">72 months"


tenure_cohort = [get_cohort(m) for m in tenure_months]

# 4. Servicios Contratados
phone_service = np.random.choice(["Yes", "No"], size=n_samples, p=[0.9, 0.1])
multiple_lines = np.where(
    phone_service == "Yes",
    np.random.choice(
        ["Yes", "No", "No phone service"], size=n_samples, p=[0.45, 0.45, 0.1]
    ),
    "No phone service",
)
internet_service = np.random.choice(
    ["Fiber", "DSL", "Mobile", "No"], size=n_samples, p=[0.42, 0.35, 0.15, 0.08]
)


def get_addon(internet):
  if internet == "No":
    return "No internet service"
  return np.random.choice(["Yes", "No"], p=[0.35, 0.65])


online_security = [get_addon(i) for i in internet_service]
online_backup = [get_addon(i) for i in internet_service]
device_protection = [get_addon(i) for i in internet_service]
tech_support = [get_addon(i) for i in internet_service]
streaming_tv = [get_addon(i) for i in internet_service]
streaming_movies = [get_addon(i) for i in internet_service]

# Conteo de servicios activos
service_count = (
    (np.array(phone_service) == "Yes").astype(int)
    + (np.array(multiple_lines) == "Yes").astype(int)
    + (np.array(internet_service) != "No").astype(int)
    + (np.array(online_security) == "Yes").astype(int)
    + (np.array(online_backup) == "Yes").astype(int)
    + (np.array(device_protection) == "Yes").astype(int)
    + (np.array(tech_support) == "Yes").astype(int)
    + (np.array(streaming_tv) == "Yes").astype(int)
    + (np.array(streaming_movies) == "Yes").astype(int)
)

# 5. Contrato y Facturación
contract = np.random.choice(
    ["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.25, 0.20]
)
paperless_billing = np.random.choice(["Yes", "No"], size=n_samples, p=[0.68, 0.32])
auto_pay = np.random.choice(["Yes", "No"], size=n_samples, p=[0.57, 0.43])
payment_method = np.random.choice(
    ["Electronic check", "Mailing check", "Bank transfer", "Credit card"],
    size=n_samples,
    p=[0.37, 0.22, 0.21, 0.20],
)

# 6. Financiero y Cargos
monthly_charges = np.round(np.random.normal(loc=87.5, scale=23.0, size=n_samples), 2)
monthly_charges = np.clip(monthly_charges, 18.0, 140.0)
discount_rate = np.random.choice([0.0, 0.05, 0.10, 0.15, 0.20], size=n_samples)
total_charges = np.round(
    monthly_charges * tenure_months * (1 - discount_rate * 0.5), 2
)

# 7. Métricas Operativas y de Soporte
support_tickets = np.random.poisson(lam=1.8, size=n_samples)
complaints = np.random.poisson(lam=0.5, size=n_samples)
customer_contacts = np.random.poisson(lam=3.9, size=n_samples)
service_outages = np.random.poisson(lam=1.1, size=n_samples)

# 8. Experiencia (NPS y CSAT)
nps = np.round(np.random.normal(loc=6.1, scale=1.7, size=n_samples), 1)
nps = np.clip(nps, 0, 10)
csat = np.round(np.random.normal(loc=3.45, scale=0.68, size=n_samples), 0)
csat = np.clip(csat, 1, 5)

# 9. Valor de Vida del Cliente (CLV) y Costos
estimated_clv = np.round(
    monthly_charges * tenure_months * np.random.uniform(0.8, 1.2, size=n_samples),
    2,
)
avg_ticket_cost = np.round(
    np.random.exponential(scale=0.85, size=n_samples), 2
)

# 10. Churn y Razones de Fuga
churn_prob = 1 / (
    1
    + np.exp(
        -(
            -2.5
            + 0.03 * monthly_charges
            - 0.04 * tenure_months
            + 0.3 * support_tickets
            + 0.4 * complaints
        )
    )
)
churn_bin = np.random.binomial(1, churn_prob, size=n_samples)
churn = np.where(churn_bin == 1, "Yes", "No")

churn_reason = np.select(
    [
        churn == "No",
        support_tickets > 3,
        service_outages > 2,
        monthly_charges > 110,
        complaints > 1,
    ],
    [
        "No churn",
        "Poor customer support response",
        "Network speed/reliability issues",
        "Price increase",
        "Competitor offered higher discount",
    ],
    default="Moved to another region",
)

# Construir DataFrame
df = pd.DataFrame(
    {
        "CustomerID": customer_ids,
        "Age": age,
        "SeniorCitizen": senior_citizen,
        "Gender": gender,
        "Partner": partner,
        "Dependents": dependents,
        "TenureMonths": tenure_months,
        "TenureCohort": tenure_cohort,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "ServiceCount": service_count,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "AutoPay": auto_pay,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "DiscountRate": discount_rate,
        "TotalCharges": total_charges,
        "SupportTickets": support_tickets,
        "Complaints": complaints,
        "CustomerContacts": customer_contacts,
        "ServiceOutages": service_outages,
        "NPS": nps,
        "CSAT": csat,
        "EstimatedCLV": estimated_clv,
        "AvgTicketCost": avg_ticket_cost,
        "Churn": churn,
        "ChurnReason": churn_reason,
    }
)

# Inyección exacta de los 150 valores nulos coincidentes
nan_indices = np.random.choice(n_samples, size=150, replace=False)
df.loc[nan_indices, "PaymentMethod"] = np.nan
df.loc[nan_indices, "MonthlyCharges"] = np.nan
df.loc[nan_indices, "NPS"] = np.nan
df.loc[nan_indices, "CSAT"] = np.nan

# Guardar el resultado final idéntico
df.to_csv("telecom_churn_10000.csv", index=False)
print(f"Dataset replicado correctamente. Dimensiones: {df.shape}")