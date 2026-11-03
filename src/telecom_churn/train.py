"""Compara modelos en validación y reserva test para evaluar el ganador."""
import json
import joblib
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from .data import load_data
from .features import preprocess_data
from .model import get_models
from .config import BASE_DIR

def metrics(y, prediction, probability):
    return {"accuracy": accuracy_score(y, prediction),
            "precision": precision_score(y, prediction, zero_division=0),
            "recall": recall_score(y, prediction, zero_division=0),
            "f1": f1_score(y, prediction, zero_division=0),
            "roc_auc": roc_auc_score(y, probability),
            "confusion_matrix": confusion_matrix(y, prediction).tolist()}

def main():
    X, y, preprocessor = preprocess_data(load_data())
    # Sin fechas disponibles: partición aleatoria estratificada 60/20/20.
    # No permite demostrar rendimiento futuro; se documenta esa limitación.
    Xdev, Xt, ydev, yt = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    Xa, Xv, ya, yv = train_test_split(Xdev, ydev, test_size=.25, stratify=ydev, random_state=42)
    candidates = {"Baseline": DummyClassifier(strategy="prior"), **get_models()}
    results = {}
    for name, model in candidates.items():
        pipeline = Pipeline([("preprocessor", clone(preprocessor)), ("classifier", clone(model))])
        pipeline.fit(Xa, ya)
        results[name] = metrics(yv, pipeline.predict(Xv), pipeline.predict_proba(Xv)[:, 1])
    # El criterio se fija antes de mirar test: mayor ROC-AUC de validación.
    winner = max((n for n in results if n != "Baseline"), key=lambda n: results[n]["roc_auc"])
    pipeline = Pipeline([("preprocessor", clone(preprocessor)), ("classifier", clone(candidates[winner]))])
    pipeline.fit(Xdev, ydev)
    report = {"seed": 42, "split": [len(Xa), len(Xv), len(Xt)], "threshold": .5,
              "features": list(X.columns), "validation": results, "winner": winner,
              "test": metrics(yt, pipeline.predict(Xt), pipeline.predict_proba(Xt)[:, 1])}
    # Guardamos el procesamiento junto al modelo para reutilizar exactamente la misma lógica.
    (BASE_DIR / "models").mkdir(exist_ok=True)
    (BASE_DIR / "reports").mkdir(exist_ok=True)
    joblib.dump(pipeline, BASE_DIR / "models/churn.joblib")
    (BASE_DIR / "reports/metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
