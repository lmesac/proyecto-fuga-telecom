from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

def get_models():
    """
    Define los 3 modelos a evaluar:
    1. Regresión Logística (Modelo lineal base)
    2. Random Forest (Modelo basado en árboles)
    3. HistGradientBoosting (Gradient Boosting nativo de Scikit-Learn, 
       de alto rendimiento y sin requerir dependencias externas de Mac)
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42)
    }
    return models