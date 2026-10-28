import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(file_path):
    """Carga el dataset de las 35 variables desde la ruta especificada."""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df, target_col, id_col):
    """
    Realiza la limpieza, imputación de nulos y transformación de variables 
    para las 35 columnas del dataset sin generar fuga de datos (data leakage).
    """
    # Separar variables predictoras (X) y la variable objetivo (y)
    # Convertimos el target 'Yes'/'No' a binario (1 y 0)
    #X = df.drop(columns=[target_col, id_col])
    # Excluimos el Churn (target), el ID y el motivo de fuga para evitar fuga de datos (data leakage)
    X = df.drop(columns=[target_col, id_col, 'ChurnReason'])
    y = df[target_col].apply(lambda val: 1 if str(val).strip().lower() in ['yes', '1', 'true'] else 0)
    
    # Identificar tipos de columnas automáticamente
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Pipeline para variables numéricas (imputa con la mediana y escala)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline para variables categóricas (imputa con valor más frecuente y aplica One-Hot Encoding)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Unir ambos preprocesadores en un ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
    return X, y, preprocessor