import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

def train_and_save_full_pipeline(df_raw, target_col="Churn", model_name="Logistic Regression"):
    """
    Crea un Pipeline completo que incluye preprocesamiento (imputación, escalado, codificación)
    y el modelo de Machine Learning, asegurando que ninguna variable crítica se pierda.
    """
    id_col = "CustomerID"
    drop_cols = [id_col]
    if 'ChurnReason' in df_raw.columns:
        drop_cols.append('ChurnReason')
        
    X = df_raw.drop(columns=[target_col] + [c for c in drop_cols if c in df_raw.columns])
    y = df_raw[target_col].apply(lambda val: 1 if str(val).strip().lower() in ['yes', '1', 'true'] else 0)
    
    # Identificar columnas numéricas y categóricas automáticamente
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Preprocesador para numéricas (Imputar + Escalar para que los tickets y quejas tengan el peso correcto)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Preprocesador para categóricas (Imputar + OneHotEncoder)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Definir modelos
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42)
    }
    
    if model_name not in models:
        raise ValueError(f"Modelo no válido: {model_name}")
        
    # Construir el Pipeline completo
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', models[model_name])
    ])
    
    print(f"Entrenando el Pipeline completo con {model_name}...")
    full_pipeline.fit(X, y)
    
    # Guardar el pipeline completo en models/best_model.pkl
    Path("models").mkdir(parents=True, exist_ok=True, mode=0o755)
    model_path = "models/best_model.pkl"
    joblib.dump(full_pipeline, model_path)
    print(f"¡Pipeline entrenado y guardado exitosamente en '{model_path}'!")
    
    return full_pipeline