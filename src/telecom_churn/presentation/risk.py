"""
Clasificación e interpretación del nivel de riesgo de churn.
"""

LOW_RISK_LIMIT = 0.35
HIGH_RISK_LIMIT = 0.65


def classify_risk(probability: float) -> str:
    if not 0 <= probability <= 1:
        raise ValueError("La probabilidad debe estar entre 0 y 1.")

    if probability < LOW_RISK_LIMIT:
        return "Bajo"

    if probability < HIGH_RISK_LIMIT:
        return "Medio"

    return "Alto"


def get_recommendation(risk_level: str) -> str:
    recommendations = {
        "Bajo": "Mantener seguimiento normal del cliente.",
        "Medio": (
            "Revisar la experiencia del cliente y considerar "
            "una acción preventiva de retención."
        ),
        "Alto": "Priorizar al cliente para una acción de retención.",
    }

    if risk_level not in recommendations:
        raise ValueError(f"Nivel de riesgo no válido: {risk_level}")

    return recommendations[risk_level]
