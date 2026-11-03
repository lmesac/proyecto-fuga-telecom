import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_preprocess_data(filepath: str = "data/telecom_churn_10000.csv"):
    """
    Carga el dataset de telecomunicaciones, remueve columnas innecesarias 
    (como ChurnReason y CustomerID), maneja valores nulos e imputa datos faltantes.
    """
    df = pd.read_csv(filepath)
    
    target_col = "Churn"
    id_col = "CustomerID"
    
    drop_cols = [id_col]
    if 'ChurnReason' in df.columns:
        drop_cols.append('ChurnReason')
        
    X = df.drop(columns=[target_col] + [c for c in drop_cols if c in df.columns])
    y = df[target_col].apply(lambda val: 1 if str(val).strip().lower() in ['yes', '1', 'true'] else 0)
    
    # 1. Convertir variables categóricas de texto a numéricas (One-Hot Encoding)
    X = pd.get_dummies(X, drop_first=True)
    
    # 2. Imputar valores nulos (NaN) usando la mediana para columnas numéricas
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    return df, X_imputed, y