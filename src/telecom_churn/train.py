from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from telecom_churn.config import DATA_FILE, TARGET_COLUMN, ID_COLUMN
from telecom_churn.features import load_data, preprocess_data
from telecom_churn.model import get_models

def main():
    print("1. Cargando datos...")
    df = load_data(DATA_FILE)
    print(f"Dimensiones del dataset: {df.shape} (Debe tener 35 columnas)")
    
    print("2. Preprocesando variables...")
    X, y, preprocessor = preprocess_data(df, TARGET_COLUMN, ID_COLUMN)
    
    # Dividir en entrenamiento (80%) y prueba (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("3. Entrenando y evaluando los 3 modelos...")
    models = get_models()
    results = {}
    
    for name, model in models.items():
        # Crear un pipeline que une el preprocesador y el modelo (evita fuga de datos)
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Entrenar modelo
        pipeline.fit(X_train, y_train)
        
        # Predecir sobre conjunto de prueba
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else y_pred
        
        # Calcular métricas clave
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        
        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc
        }
        
        print(f"\n--- Resultados para: {name} ---")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {auc:.4f}")

    print("\n¡Entrenamiento completado exitosamente con las 35 variables!")

if __name__ == "__main__":
    main()