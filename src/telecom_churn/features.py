"""Transformaciones ajustadas únicamente con los datos de entrenamiento."""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from .data import load_data

def preprocess_data(df, target_col="Churn", id_col="CustomerID"):
    # ChurnReason revela el resultado; el identificador no representa comportamiento.
    X = df.drop(columns=[target_col, id_col, "ChurnReason"])
    y = df[target_col].map({"No": 0, "Yes": 1})
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X if c not in numeric]
    # Se devuelve un transformador SIN ajustar: Pipeline hará fit después del split.
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ])
    return X, y, preprocessor

def load_and_preprocess_data(filepath="data/telecom_churn_10000.csv"):
    # Compatibilidad con las llamadas existentes del equipo.
    df = load_data(filepath)
    X, y, _ = preprocess_data(df)
    return df, X, y
