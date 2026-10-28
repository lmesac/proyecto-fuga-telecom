import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_preprocess_data(filepath: str = "data/telecom_churn_10000.csv"):
    """
    Carga el dataset de telecomunicaciones y remueve columnas innecesarias 
    (como ChurnReason y CustomerID) para evitar data leakage.
    """
    df = pd.read_csv(filepath)
    
    target_col = "Churn"
    id_col = "CustomerID"
    
    drop_cols = [id_col]
    if 'ChurnReason' in df.columns:
        drop_cols.append('ChurnReason')
        
    X = df.drop(columns=[target_col] + [c for c in drop_cols if c in df.columns])
    y = df[target_col].apply(lambda val: 1 if str(val).strip().lower() in ['yes', '1', 'true'] else 0)
    
    return df, X, y