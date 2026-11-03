"""Inferencia usando el pipeline completo guardado por train.py."""
import joblib
import pandas as pd
from .config import BASE_DIR

def predict_churn(data, model_path=None):
    # Cargar únicamente artefactos propios: joblib no es seguro para archivos desconocidos.
    model = joblib.load(model_path or BASE_DIR / "models/churn.joblib")
    frame = pd.read_csv(data) if isinstance(data, (str, type(BASE_DIR))) else data.copy()
    missing = set(model.feature_names_in_) - set(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {sorted(missing)}")
    frame = frame[list(model.feature_names_in_)]
    return pd.DataFrame({"probabilidad_abandono": model.predict_proba(frame)[:, 1],
                         "prediccion": model.predict(frame)})
