"""Evaluación complementaria: no utiliza el test final ni cambia el modelo publicado."""
import json
import numpy as np
import plotly.express as px
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.inspection import permutation_importance
from .data import load_data
from .features import preprocess_data
from .model import get_models
from .train import metrics
from .config import BASE_DIR

def main():
    X, y, prep = preprocess_data(load_data())
    # Reproducimos la partición original y apartamos el test: no se consulta aquí.
    Xdev, _, ydev, _ = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    Xa, Xv, ya, yv = train_test_split(Xdev, ydev, test_size=.25, stratify=ydev, random_state=42)
    result = {"cv": {}, "thresholds": [], "note": "Análisis exploratorio sobre desarrollo; no modifica el umbral operativo."}
    for name, model in get_models().items():
        pipe = Pipeline([("prep", clone(prep)), ("model", clone(model))])
        # El procesamiento se reajusta dentro de cada pliegue: evita fuga entre folds.
        cv = cross_validate(pipe, Xa, ya, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                            scoring=["roc_auc", "average_precision", "recall"], n_jobs=1)
        result["cv"][name] = {key: {"mean": float(np.mean(value)), "std": float(np.std(value))}
                               for key, value in cv.items() if key.startswith("test_")}
    # Analizamos la regresión logística ya seleccionada; no elegimos otra mirando test.
    pipe = Pipeline([("prep", clone(prep)), ("model", get_models()["Logistic Regression"])])
    pipe.fit(Xa, ya)
    probability = pipe.predict_proba(Xv)[:, 1]
    for threshold in [.2, .3, .4, .5, .6, .7]:
        pred = probability >= threshold
        result["thresholds"].append({"threshold": threshold, "flagged": int(pred.sum()), **metrics(yv, pred, probability)})
    # Simulación de capacidad: ordenar por riesgo y revisar únicamente el 20% superior.
    top = np.argsort(-probability, kind="stable")[:int(len(yv)*.2)]
    result["top20"] = {"clients": len(top), "precision": float(yv.iloc[top].mean()),
                        "recall": float(yv.iloc[top].sum()/yv.sum())}
    # Permutar una columna mide cuánto cae AUC. Correlaciones pueden repartir su importancia.
    importance = permutation_importance(pipe, Xv, yv, scoring="roc_auc", n_repeats=3, random_state=42)
    result["importance"] = sorted([{"feature": c, "mean": float(m), "std": float(s)}
        for c,m,s in zip(X.columns,importance.importances_mean,importance.importances_std)], key=lambda v:v["mean"], reverse=True)
    output = BASE_DIR / "reports"
    output.mkdir(exist_ok=True)
    (output / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    px.bar(result["importance"][:12], x="mean", y="feature", orientation="h", title="Caída de ROC-AUC al permutar variables (validación)").write_html(output / "importance.html", include_plotlyjs=True)
    print(json.dumps(result["cv"], indent=2))
    print("Top 20%:", result["top20"])

if __name__ == "__main__":
    main()
